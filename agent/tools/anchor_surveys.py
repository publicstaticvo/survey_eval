import re
import json
import math
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sentence_transformers.util import cos_sim

from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .paper_download import PaperDownload
from .utils import extract_json, extract_list
from .openalex import openalex_search_paper, OPENALEX_SELECT, to_openalex
from .prompts import TOPIC_AGGREGATION_PROMPT, ANCHOR_SURVEY_SELECT, ANCHOR_PAPER_SELECT


class SurveyDownload(PaperDownload):

    STRUCTURE_TITLES = ["introduction", "background", "conclusion", "discussion", "experiment", "result", "method", "limitation"]

    def _check_title(self, title: str):
        for x in self.STRUCTURE_TITLES: title = title.replace(x, "")
        return len(title) >= 10
    
    def _post_hook(self, xml_content: str):
        try:
            titles = self.paper_parser.get_titles(xml_content)
            titles = set(x for x in titles if self._check_title(x))
            print(f"This survey has {len(titles)} titles")
            return list(titles), xml_content
        except Exception as e:
            print(f"Fatal: no survey parser {e}")
            return [], None
        

class AnchorPaperSelect(AsyncChat):

    PROMPT: str = ANCHOR_PAPER_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [x['title'] for x in results['selected_papers']]
        title_to_paper = {x['title']: x for x in context['papers'].values()}
        return [title_to_paper[x] for x in titles if x in title_to_paper]
    
    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(query=inputs['query'], titles="\n".join(f"- {t['title']}" for t in inputs['papers'].values()))
        return prompt, {"papers": inputs['papers']}


class AnchorSurveySelect(AsyncChat):
    """Cross-survey topic aggregate"""

    PROMPT: str = ANCHOR_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        titles = [x['title'] for x in results['surveys']]
        title_to_paper = {x['title']: x for x in context['surveys']}
        return [title_to_paper[x] for x in titles if x in title_to_paper]
    
    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(query=inputs['query'], titles="\n".join(f"- {t['title']}" for t in inputs['surveys']))
        return prompt, {"surveys": inputs['surveys']}
        

class TopicAggregateLLMClient(AsyncChat):
    """Cross-survey topic aggregate"""

    PROMPT: str = TOPIC_AGGREGATION_PROMPT

    def _availability(self, response: str, context: dict):
        topics = extract_json(response)
        #  if len(set(x['representative_papers'])) >= 2
        return [x['topic_name'] for x in topics['topics']]
    
    def _organize_inputs(self, inputs):
        # 高置信度证据 == anchor survey的子标题
        has_titles = {k: v for k, v in inputs['surveys'].items() if v is not None}
        # 低置信度证据 == 其他survey
        other_papers = [k for k, v in inputs['surveys'].items() if v is None] + [x['title'] for x in inputs['papers'] if x['title'] not in inputs['surveys']]
        high = [f"Survey title: {k}\nSection titles:\n{'\n'.join(f'- {t}' for t in v['titles'])}\n" for k, v in has_titles.items()]
        prompt = self.PROMPT.format(query=inputs['query'], anchors='\n'.join(high), surveys="\n".join(f'- {x}' for x in other_papers))
        # print(prompt)
        return prompt, {}


class AnchorSurveyFetch:
    
    SELECT: str = f"{OPENALEX_SELECT},best_oa_location,locations"
    
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.survey_download = SurveyDownload(config.grobid_url)
        self.paper_select = AnchorPaperSelect(config.llm_server_info)
        self.survey_select = AnchorSurveySelect(config.llm_server_info)
        self.topic_aggregate = TopicAggregateLLMClient(config.llm_server_info)
    
    def _citation_count_by_eval_date(self, paper: dict):
        """
        Calculate the cited by count on evaluation date.
        """
        eval_year = int(self.eval_date.year)
        if (citation_count := paper.get("cited_by_count", 0)):
            for x in paper.get("counts_by_year", []):
                if x['year'] > eval_year:
                    citation_count -= x['cited_by_count']
            return citation_count
        citation_count = 0
        for x in paper.get("counts_by_year", []):
            if x['year'] <= eval_year:
                citation_count += x['cited_by_count']
        return citation_count
    
    def _is_survey(self, paper: dict):
        if len(paper['referenced_works']) < 40: return False
        if self._citation_count_by_eval_date(paper) < 20: return False
        return True
    
    async def _download_surveys(self, papers: list[dict]):
        not_survey_id = set()
        tasks = [asyncio.create_task(self.survey_download.download_single_paper(x)) for x in papers]
        downloaded_surveys = {paper['title']: None for paper in papers}
        try:
            for paper, task in zip(papers, tasks):
                try:
                    titles, survey = await task
                    if titles:
                        if len(titles) > 5: 
                            downloaded_surveys[paper['title']] = {"titles": titles, "survey": survey}
                        else: 
                            del downloaded_surveys[paper['title']]
                            not_survey_id.add(paper['id'])
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    print(f"_download_surveys {e} {type(e)}")
        finally:
            # 确保所有任务都被清理
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        print(f"We have {len(downloaded_surveys)} downloaded surveys:")
        for x in downloaded_surveys: 
            if downloaded_surveys[x]: print(f"- {x}, {len(downloaded_surveys[x]['titles'])} titles")
            else: print(f"- {x}, no OA")
        return downloaded_surveys, not_survey_id
  
    async def _get_paper_meta_by_id(self, papers: list[str], batch_size: int = 50):
        paper_meta = {}
        for i in range(0, len(papers), batch_size):
            try:
                batch = papers[i: i + batch_size]
                results = await openalex_search_paper("works", filter={"openalex": "|".join(batch)}, per_page=batch_size)
                for x in results['results']: paper_meta[x['id']] = x
            except Exception as e:
                print(f"Critical: no paper meta in page {i}: {e}")
                continue
        return paper_meta
    
    async def _anchor_papers(self, papers: dict, survey_title: str) -> tuple[dict, list]:  
        """
        papers: {"OpenalexID": {metadata}, "OpenalexID": {metadata}}
        """
        # 第二步：叫LLM过滤重点文献。
        try:
            domain_papers = await self.paper_select.call(inputs={"query": survey_title, "papers": papers})
        except Exception as e:
            print(f"domain_papers {e}")
            domain_papers = None
        if not domain_papers:
            print(f"Critical: no domain papers.")
            return {}, []
        print(f"Get {len(domain_papers)} domain papers")
        return domain_papers
    
    async def _anchor_survey(self, real_surveys: list[dict], survey_title: str) -> dict:
        # 第二步：从返回的文章中找到真正的综述。
        if not real_surveys: return {}, {}
        print(f"Get {len(real_surveys)} real surveys")
        anchor_surveys = await self.survey_select.call(inputs={"query": survey_title, "surveys": real_surveys})
        survey_to_subtitles, not_survey_ids = await self._download_surveys(anchor_surveys)
        # 分类讨论。若文章数量小于等于2，则没有anchor papers。
        anchor_surveys = [x for x in anchor_surveys if x['id'] not in not_survey_ids]
        # anchor papers
        anchor_papers = {}
        if len(anchor_surveys) >= 3:
            cited_papers = {}  # id -> metadata
            for x in anchor_surveys:
                for w in x['referenced_works']:
                    cited_papers[w] = cited_papers.get(w, 0) + 1
            cited_by_threshold = max(3, math.ceil(0.6 * len(anchor_surveys)))
            anchor_papers = {k: v for k, v in cited_papers.items() if v >= cited_by_threshold}  # id -> metadata
            print(f"Get {len(anchor_papers)} anchor papers")
        # 通过openalex获取其他信息。
        anchor_paper_meta = await self._get_paper_meta_by_id(list(anchor_papers))  # id -> metadata
        for k in anchor_paper_meta:
            anchor_paper_meta[k]['survey_cited_by_count'] = anchor_papers[k]
        print(f"Get {len(survey_to_subtitles)} anchor surveys: {list(survey_to_subtitles)}")
        return anchor_papers, survey_to_subtitles
    
    async def __call__(self, survey_papers: list[dict], library: dict, survey_title: str): 
        domain_papers = await self._anchor_papers(library, survey_title)
        anchor_papers, anchor_surveys = await self._anchor_survey(survey_papers, survey_title)
        # 第四步：确定要使用的topic列表。
        topics = await self.topic_aggregate.call(inputs={"query": survey_title, "surveys": anchor_surveys, "papers": domain_papers})
        return {"anchor_papers": anchor_papers, "golden_topics": topics, "surveys": domain_papers, "downloaded": anchor_surveys}
