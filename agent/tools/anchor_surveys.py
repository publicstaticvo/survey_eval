import re
import json
import math
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .tool_config import ToolConfig
from .llmclient import AsyncLLMClient
from .paper_download import PaperDownload
from .openalex import openalex_search_paper, OPENALEX_SELECT
from .utils import extract_json, extract_list


SURVEY_KEYWORDS = r"(survey|summary|review|overview|synthesis|taxonomy|study)"
SURVEY_PATTERN = re.compile(f"a (systematic |comprehensive |literature )?{SURVEY_KEYWORDS}|{SURVEY_KEYWORDS} of", re.IGNORECASE)


class SurveyDownload(PaperDownload):
    
    def _post_hook(self, xml_content: str):
        try:
            return self.paper_parser.get_titles(xml_content)
        except Exception as e:
            return []
        

class TopicAggregateLLMClient(AsyncLLMClient):
    """Cross-survey topic aggregate"""

    PROMPT: str

    def _availability(self, response: str):
        topics = extract_json(response)
        return [x['topic'] for x in topics['topics'] if len(set(x['surveys'])) >= 2]
    
    def _organize_inputs(self, inputs):
        titles_str = [f"S{i + 1}:\n{'\n'.join(f'- {x}' for x in t)}" for i, t in enumerate(inputs['titles'])]
        return self.PROMPT.format(query=inputs['query'], titles="\n".join(titles_str))


class AnchorSurveyFetch:
    
    SELECT: str = f"{OPENALEX_SELECT},type,concepts,best_oa_location,locations"
    GOLDEN_SURVEY_CONCEPTS: dict[str, str] = {"review article": "C140608501", "systematic review": "C189708586", "narrative review": "C3020000205"}
    SILVER_SURVEY_CONCEPTS: dict[str, str] = {"meta-analysis": "C95190672"}
    
    def __init__(self, config: ToolConfig, eval_date: datetime = datetime.now()):
        self.eval_date = eval_date
        self.survey_download = SurveyDownload()
        self.topic_aggregate = TopicAggregateLLMClient(config.llm_server_info)
        # self.spacy_model = spacy.load("en_core_web_sm")
        # self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
    
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
        if (cited_by_count := self._citation_count_by_eval_date(paper)) < 20: return False
        for x in paper['concepts']:
            if x['display_name'] in self.GOLDEN_SURVEY_CONCEPTS and x['score'] > 0.3: return True
            elif x['display_name'] in self.SILVER_SURVEY_CONCEPTS and x['score'] > 0.3:
                if len(paper['referenced_works']) >= 60 and cited_by_count >= 30: return True
        return False
    
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
                    titles = await task
                    downloaded_surveys[paper['title']] = titles
                    if len(downloaded_surveys) >= 5:
                        for other_task in tasks:
                            if not other_task.done():
                                other_task.cancel()
                        break
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
        return downloaded_surveys
  
    async def _get_paper_meta_by_id(self, papers: list[str], batch_size: int = 50):
        paper_meta = {}
        for i in range(0, len(papers), batch_size):
            try:
                batch = papers[i: i + batch_size]
                results = await openalex_search_paper("works", filter={"openalex": "|".join(batch)}, per_page=batch_size)
                paper_meta |= results['results']
            except Exception as e:
                print(f"Critical: no paper meta in page {i}")
                continue
        return paper_meta
    
    async def __call__(self, query: str):        
        # 第一步：用“领域关键词+综述”搜索最近出现过的综述。
        from_date = (self.eval_date - relativedelta(years=2)).strftime("%Y-%m-%d")
        to_date = self.eval_date.strftime("%Y-%m-%d")
        try:
            search = {
                "default.search": query,
                "concept.id": "|".join(self.GOLDEN_SURVEY_CONCEPTS | self.SILVER_SURVEY_CONCEPTS), 
                "to_publication_date": to_date, 
                "from_publication_date": from_date
            }
            survey_papers = await openalex_search_paper("works", search, per_page=50, select=self.SELECT)['results']
        except Exception as e:
            print("Critical: no survey papers")
            return
        # 第二步：从返回的文章中找到真正的综述。
        real_surveys = [x for x in survey_papers if self._is_survey(x)]
        # 第三步：分析综述引用情况。
        cited_papers = {}
        for x in real_surveys:
            for w in x['referenced_works']:
                cited_papers[w] = cited_papers.get(w, 0) + 1
        # cited_papers = sorted(cited_papers.items(), key=lambda x: x[1], reverse=True)
        cited_by_threshold = min(max(3, math.ceil(0.25 * len(real_surveys))), 8)
        anchor_papers = {k: v for k, v in cited_papers if v >= cited_by_threshold}
        # 通过openalex获取其他信息。
        anchor_paper_meta = await self._get_paper_meta_by_id(list(anchor_papers))
        for k in anchor_paper_meta:
            anchor_paper_meta[k]['survey_cited_by_count'] = anchor_papers[k]
        # 第四步：确定要使用的topic列表。
        survey_to_subtitles = await self._download_surveys(real_surveys)
        topics = await self.topic_aggregate.call(inputs={"query": query, "titles": survey_to_subtitles})
        return {"anchor_papers": anchor_paper_meta, "golden_topics": topics}


async def main():
    config = ToolConfig()
    query = ""
    results = await AnchorSurveyFetch(config)(query)
    with open("debug/anchor_papers.json") as f:
        json.dump(results['anchor_papers'], f, ensure_ascii=False)
    with open("debug/topics.txt") as f:
        f.write("\n".join(results['golden_topics']))


if __name__ == "__main__":
    asyncio.run(main())
