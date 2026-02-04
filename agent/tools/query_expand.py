from typing import List, Dict, Any
from datetime import datetime
import jsonschema

from .utils import extract_json
from .llmclient import AsyncChat
from .tool_config import ToolConfig
from .prompts import QUERY_EXPANSION_PROMPT, QUERY_SCHEMA, SURVEY_SPECIFIED_QUERY_EXPANSION
from .openalex import to_openalex, OPENALEX_SELECT, openalex_search_paper

SURVEY_KEYWORDS = "survey|summary|review|overview|synthesis|taxonomy|study"
    

class QueryExpansionLLMClient(AsyncChat):
    PROMPT: str = QUERY_EXPANSION_PROMPT
    def _availability(self, response, context):
        queries = extract_json(response)
        jsonschema.validate(queries, QUERY_SCHEMA)
        return queries
    

class SurveySpecifiedLLMClient(AsyncChat):
    PROMPT: str = SURVEY_SPECIFIED_QUERY_EXPANSION
    def _availability(self, response, context):
        return extract_json(response)['query']
    

class QueryExpand:
    
    def __init__(self, config: ToolConfig):
        self.llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.survey_llm = SurveySpecifiedLLMClient(config.llm_server_info, config.sampling_params)
        self.eval_date = config.evaluation_date

    async def _request_for_papers(self, query, uplimit, select=f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        queries = to_openalex(query)
        search = [
            *[("default.search", q) for q in queries],
            ("to_publication_date", self.eval_date.strftime("%Y-%m-%d"))
        ]
        oracle = []
        # first query
        for page in range(1, (uplimit - 1) // 200 + 2):
            try:
                results = await openalex_search_paper("works", search, per_page=min(200, uplimit), select=select, page=page)
            except Exception as e:
                print(f"An {e} cause oracle data miss in query {query}.")
                return []
            uplimit = results['count']
            oracle.extend(results['results'])

        return oracle

    async def _request_for_surveys(self, query, uplimit, select=f"{OPENALEX_SELECT},relevance_score") -> List[Dict[str, Any]]:
        queries = to_openalex(query)
        search = [
            *[("title.search", q) for q in queries],
            ("title.search", SURVEY_KEYWORDS)
            ("to_publication_date", self.eval_date.strftime("%Y-%m-%d"))
        ]
        oracle = []
        # first query
        for page in range(1, (uplimit - 1) // 200 + 2):
            try:
                results = await openalex_search_paper("works", search, per_page=min(200, uplimit), select=select, page=page)
            except Exception as e:
                print(f"An {e} cause oracle data miss in query {query}.")
                return []
            uplimit = results['count']
            oracle.extend(results['results'])

        return oracle

    async def __call__(self, query: str, papers_for_each_query: int = 50):        
        # 1. fetch 4 queries
        try:
            queries = await self.llm.call(inputs={"query": query})
        except Exception as e:
            print(f"QueryExpansion {e}")
            queries = {}      
        try:
            survey_query = await self.survey_llm.call(inputs={"query": query})
        except Exception as e:
            print(f"QueryExpansion2 {e}")
            survey_query = ""
        # 2 request papers 要串行
        lqueries = [*queries['core_anchor'], queries['theoretical_bridge'], queries['methodological_bridge']]
        print(f"Get {len(lqueries)} queries: {lqueries}\nAnd survey_specified: {survey_query}")
        prev_queries, library = [], {}
        for q in lqueries:                
            try:
                oracle = await self._request_for_papers(q, papers_for_each_query)
                if oracle:
                    prev_queries.append(q)
                    for x in oracle:
                        if x['id'] not in library: 
                            library[x['id']] = x
                            library[x['id']]['query'] = q
                        elif x['relevance_score'] > library[x['id']]['relevance_score']:
                            library[x['id']]['query'] = q
                            library[x['id']]['relevance_score'] = x['relevance_score']
            except Exception as e:
                print(f"Query {q} has an {e}")
        # survey specified
        oracle = []
        if survey_query:
            try:
                oracle = await self._request_for_surveys(survey_query, papers_for_each_query, f"{OPENALEX_SELECT},best_oa_location,locations")
            except Exception as e:
                print(f"Query {q} has an {e}")
        return {"queries": prev_queries, "core": oracle, "library": library}
