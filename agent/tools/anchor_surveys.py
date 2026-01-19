import re
import json
import math
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .tool_config import ToolConfig
from .llmclient import AsyncLLMClient
from .paper_download import PaperDownload
from .utils import extract_json, extract_list
from .openalex import openalex_search_paper, OPENALEX_SELECT
from .prompts import TOPIC_AGGREGATION_PROMPT, ANCHOR_SURVEY_SELECT


SURVEY_KEYWORDS = r"(survey|summary|review|overview|synthesis|taxonomy|study)"
SURVEY_PATTERN = re.compile(f"a (systematic |comprehensive |literature )?{SURVEY_KEYWORDS}|{SURVEY_KEYWORDS} of", re.IGNORECASE)


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
        

class AnchorSurveySelect(AsyncLLMClient):
    """Cross-survey topic aggregate"""

    PROMPT: str = ANCHOR_SURVEY_SELECT

    def _availability(self, response: str):
        results = extract_json(response)
        titles = [x['title'] for x in results['surveys']]
        title_to_paper = {x['title']: x for x in self._context}
        return [title_to_paper[x] for x in titles if x in title_to_paper]
    
    def _organize_inputs(self, inputs):
        self._context = inputs['surveys']
        prompt = self.PROMPT.format(query=inputs['query'], titles="\n".join("\n".join(f"- {t['title']}" for t in self._context)))
        return prompt
        

class TopicAggregateLLMClient(AsyncLLMClient):
    """Cross-survey topic aggregate"""

    PROMPT: str = TOPIC_AGGREGATION_PROMPT

    def _availability(self, response: str):
        topics = extract_json(response)
        #  if len(set(x['representative_papers'])) >= 2
        return [x['topic_name'] for x in topics['topics']]
    
    def _organize_inputs(self, inputs):
        # 高置信度证据 == anchor survey的子标题
        high = [f"Survey title: {k}\nSection titles:\n{'\n'.join(f'- {t}' for t in v['titles'])}\n" for k, v in inputs['anchors'].items()]
        # 低置信度证据 == 其他survey
        surveys_str = "\n".join(f'- {x['title']}' for x in inputs['surveys'] if x['title'] not in inputs['anchors'])
        prompt = self.PROMPT.format(query=inputs['query'], anchors='\n'.join(high), surveys=surveys_str)
        print(prompt)
        return prompt


class AnchorSurveyFetch:
    
    SELECT: str = f"{OPENALEX_SELECT},type,concepts,best_oa_location,locations"
    
    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.survey_download = SurveyDownload()
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
    
    async def _get_domain_to_filter_survey(self, query, survey_papers):
        to_date = self.eval_date.strftime("%Y-%m-%d")
        try:
            search = {"default.search": query, "to_publication_date": to_date}
            field_overview = await openalex_search_paper("works", search, per_page=200, select="id,cited_by_count,references")
        except Exception as e:
            print("Did not get field")
            field_overview = survey_papers
        reference_count = [len(x['referenced_works']) for x in field_overview['results'] if x['referenced_works']]
        if not reference_count:
            reference_count = [len(x['referenced_works']) for x in survey_papers['results'] if x['referenced_works']]
        reference_count.sort()
        reference_threshold = reference_count[int(len(reference_count) * 0.8)]
        cited_by_count = [x['cited_by_count'] for x in field_overview['results']]
        cited_by_count.sort()
        cited_by_threshold = cited_by_count[int(len(cited_by_count) * 0.75)]
        real_surveys = []
        for x in survey_papers['results']:
            if x['type']['display_name'].lower() == "review": 
                real_surveys.append(x)
            elif any(y['display_name'].lower() in self.GOLDEN_SURVEY_CONCEPTS for y in x['concepts']):
                real_surveys.append(x)
            elif len(x['referenced_works']) >= reference_threshold and self._citation_count_by_eval_date(x) >= cited_by_threshold and re.search(SURVEY_PATTERN, x['title']):
                real_surveys.append(x)
    
    async def _download_surveys(self, papers):
        tasks = [asyncio.create_task(self.survey_download.download_single_paper(x)) for x in papers]
        downloaded_surveys = {}
        try:
            for paper, task in zip(papers, tasks):
                try:
                    titles, survey = await task
                    if titles:
                        downloaded_surveys[paper['title']] = {"titles": titles, "survey": survey}
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    continue
        finally:
            # 确保所有任务都被清理
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        print(f"We have {len(downloaded_surveys)} downloaded surveys:")
        for x in downloaded_surveys: print(f"- {x}")
        return downloaded_surveys
  
    async def _get_paper_meta_by_id(self, papers: list[str], batch_size: int = 50):
        paper_meta = {}
        for i in range(0, len(papers), batch_size):
            try:
                batch = papers[i: i + batch_size]
                results = await openalex_search_paper("works", filter={"openalex": "|".join(batch)}, per_page=batch_size)
                paper_meta |= results['results']
            except Exception as e:
                print(f"Critical: no paper meta in page {i}: {e}")
                continue
        return paper_meta
    
    async def __call__(self, query: str, survey_title: str):        
        # 第一步：用“领域关键词+综述”搜索最近出现过的综述。
        from_date = (self.eval_date - relativedelta(years=4)).strftime("%Y-%m-%d")
        to_date = self.eval_date.strftime("%Y-%m-%d")
        try:
            search = [
                ("default.search", query),
                ("default.search", "survey|summary|overview|comprehensive study|synthesis|review"),
                ("to_publication_date", to_date),
                # ("from_publication_date", from_date)
            ]
            task = await openalex_search_paper("works", search, per_page=50, select=self.SELECT)
            survey_papers = task['results']
        except Exception as e:
            print(f"Critical: no survey papers: {e}")
            return {"anchor_papers": {}, "golden_topics": [], "surveys": []}
        if not survey_papers:
            print(f"Critical: no survey papers. The request args is {search}")
            return {"anchor_papers": {}, "golden_topics": [], "surveys": []}
        # 第二步：从返回的文章中找到真正的综述。
        real_surveys = [x for x in survey_papers if self._is_survey(x)]
        if not real_surveys:
            print(f"Critical: no survey papers after filtering. The request args is {search}. We got {len(survey_papers)} surveys but filtered none.")
            return {"anchor_papers": {}, "golden_topics": [], "surveys": survey_papers}
        # 第三步：分析综述引用情况。
        cited_papers = {}
        for x in real_surveys:
            for w in x['referenced_works']:
                cited_papers[w] = cited_papers.get(w, 0) + 1
        # cited_papers = sorted(cited_papers.items(), key=lambda x: x[1], reverse=True)
        cited_by_threshold = max(3, math.ceil(0.5 * len(real_surveys)))
        anchor_papers = {k: v for k, v in cited_papers.items() if v >= cited_by_threshold}
        # 通过openalex获取其他信息。
        anchor_paper_meta = await self._get_paper_meta_by_id(list(anchor_papers))
        for k in anchor_paper_meta:
            anchor_paper_meta[k]['survey_cited_by_count'] = anchor_papers[k]
        # 第四步：确定要使用的topic列表。
        anchor_surveys = await self.survey_select.call(inputs={"query": survey_title, "surveys": real_surveys, "survey_title": survey_title})
        survey_to_subtitles = await self._download_surveys(anchor_surveys)
        topics = await self.topic_aggregate.call(inputs={"query": survey_title, "anchors": survey_to_subtitles, "surveys": real_surveys})
        return {"anchor_papers": anchor_paper_meta, "golden_topics": topics, "surveys": real_surveys, "downloaded": survey_to_subtitles}
