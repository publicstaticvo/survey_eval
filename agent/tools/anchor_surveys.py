import json
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .utils import split_content_to_paragraph, extract_json
from .request_utils import AsyncLLMClient, openalex_search_paper, OPENALEX_SELECT
from .prompts import CLAIM_SEGMENTATION_PROMPT
from .tool_config import ToolConfig


class ClaimSegmentationLLMClient(AsyncLLMClient):

    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def _availability(self, response: json):
        response = response["choices"][0]["message"]["content"]
        claim = extract_json(response)
        assert all(x in claim for x in ['claim', 'claim_type', 'requires'])
        
        citations = self._context['citations']
        assert set(claim['requires'].keys()) == set(citations.keys())
        claim['citations'] = citations
        # 将FULL_TEXT、TITLE_AND_ABSTRACT、TITLE_ONLY转化为0/1/2，与status对应。
        for x in claim['requires']:
            if claim['requires'][x].upper() == "TITLE_ONLY": claim['requires'][x] = 2
            elif claim['requires'][x].upper() == "TITLE_AND_ABSTRACT": claim['requires'][x] = 1
            # 默认为FULL_TEXT
            else: claim['requires'][x] = 0
        claim['paragraph_id'] = self._context['paragraph_id']
        return claim
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(text=inputs['text'], range=inputs['range'])


class AnchorSurveyFetch:
    
    SELECT: str = f"{OPENALEX_SELECT},type,concepts"
    SURVEY_CONCEPTS: set[str] = {"review article", "survey", "meta-analysis"}
    SURVEY_KEYWORDS: set[str] = {'survey', 'summary', 'review', 'comprehensive', 'overview', 
                                 'systematic', 'taxonomy', 'synthesis', 'comprehensive study'}
    
    def __init__(self, config: ToolConfig, eval_date: datetime = datetime.now()):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers)
        self.eval_date = eval_date

    def _count_survey_keywords(self, paper):
        text = f"{paper['title']}. {paper['abstract']}".lower()
        return sum(int(x in text) for x in self.SURVEY_KEYWORDS)
    
    async def __call__(self, query: str):        
        # 第一步：用“领域关键词+综述”搜索最近出现过的综述。
        default_search = f"{query},default.search:summary|survey|review|overview|tutorial|synthesis"
        from_date = (self.eval_date - relativedelta(years=2)).strftime("%Y-%m-%d")
        to_date = self.eval_date.strftime("%Y-%m-%d")
        try:
            search = {"default.search": default_search, "to_publication_date": to_date, "from_publication_date": from_date}
            survey_papers = await openalex_search_paper("works", search, per_page=100, select=self.SELECT)
        except Exception as e:
            print("Critical: no survey papers")
            return
        # 第二步：从返回的文章中找到真正的综述。
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
        num_reference_threshold = reference_count[int(len(reference_count) * 0.8)]
        # cited_by_count = [x['cited_by_count'] for x in field_overview['results']]
        # cited_by_count.sort()
        # num_cited_by_threshold = cited_by_count[int(len(cited_by_count) * 0.75)]
        real_surveys = []
        for x in survey_papers['results']:
            if x['type']['display_name'].lower() == "review": 
                real_surveys.append(x)
            elif any(y['display_name'].lower() in self.SURVEY_CONCEPTS for y in x['concepts']):
                real_surveys.append(x)
            elif len(x['referenced_works']) >= num_reference_threshold and self._count_survey_keywords(x) >= 2:
                real_surveys.append(x)                
        # 第三步：分析综述引用情况。
        cited_papers = {}
        for x in real_surveys:
            for w in x['referenced_works']:
                cited_papers[w] = cited_papers.get(w, 0) + 1
        # cited_papers = sorted(cited_papers.items(), key=lambda x: x[1], reverse=True)
        cited_by_threshold = max(3, math.ceil(0.3 * len(real_surveys)))
        anchor_papers = {k: v for k, v in cited_papers if v >= cited_by_threshold}
        return {"anchor_papers": anchor_papers}
