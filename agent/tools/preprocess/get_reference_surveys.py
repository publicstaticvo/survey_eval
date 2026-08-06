import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import jsonschema

from .sentences import SentenceClassification
from ..prompts import REFERENCE_SURVEY_SCHEMA, REFERENCE_SURVEY_SELECT
from ..utility.academic_engine import get_academic_engine
from ..utility.llmclient import AsyncChat
from ..utility.openalex import OPENALEX_SELECT, get_openalex_client
from ..utility.paper_download import (
    PaperDownload,
    S2PaperDownload,
    yield_location,
)
from ..utility.content_walk import iter_heading
from ..utility.paper_elements import Paper
from ..utility.s2 import get_semantic_scholar_client
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


ORACLE_SELECT = f"{OPENALEX_SELECT},locations,best_oa_location,relevance_score"
SURVEY_TITLE_KEYWORDS = ("survey", "summary", "review", "overview", "comprehensive study")
S2_ENGINE_NAMES = {"semantic scholar", "semantic_scholar", "semanticscholar", "s2"}


class ReferenceSurveySelect(AsyncChat):
    PROMPT: str = REFERENCE_SURVEY_SELECT

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        jsonschema.validate(results, REFERENCE_SURVEY_SCHEMA)
        title_to_paper = {item["title"]: item for item in context["surveys"]}
        selected = []
        for item in results['reference_surveys']:
            paper = dict(title_to_paper[item["title"]])
            paper["reference_survey_reason"] = item["reason"]
            paper["covered_subtopics"] = item["covered_subtopics"]
            selected.append(paper)
        return selected

    def _organize_inputs(self, inputs):
        candidates = "\n".join(f"{i + 1:02d}. Title: {paper['title']}\n    Abstract: {paper['abstract']}" for i, paper in enumerate(inputs["surveys"]))
        prompt = self.PROMPT.format(query=inputs["query"], candidates=candidates)
        return prompt, {"surveys": inputs["surveys"]}


class GetReferenceSurveys:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.config = config
        self.eval_date = config.evaluation_date
        self.openalex_downloader = PaperDownload(config)
        self.semantic_scholar_downloader = S2PaperDownload(config)
        self.survey_select = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)
        self.openalex = get_openalex_client(config)
        self.semantic_scholar = get_semantic_scholar_client(config)
        self.academic_engine = get_academic_engine(config)
        self.sentence_llm = SentenceClassification(config)
        self.academic_engine_type = config.default_academic_search_engine
        self.limit = config.reference_survey_search_limit

    def _uses_semantic_scholar_engine(self) -> bool:
        return (self.academic_engine_type or "").strip().lower() in S2_ENGINE_NAMES

    async def _search_surveys(self, query: str, limit: int = 50):
        if self.academic_engine_type == "openalex": 
            payload = await self.academic_engine.search_works(
                search=query,
                filter={
                    "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
                    "title.search": 'survey|summary|review|overview|"comprehensive study"'
                },
                per_page=limit,
                select=self.SELECT,
            )
        else:
            search_query = f'{query} AND (survey OR summary OR review OR overview OR "comprehensive study")'
            print(f"Search keywords: {search_query}")
            payload = await self.academic_engine.search_works(
                search=search_query,
                filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")},
                per_page=limit,
                select=self.SELECT,
            )
        candidates = [item for item in payload.get("results", []) if (item.get("title") or "").strip()]
        if len(candidates) > limit:
            print(f"referenceSurveySearch clipped {len(candidates)} candidates to limit={limit}")
        return candidates[:limit]

    def _is_review_like(self, paper: dict) -> bool:
        title = (paper.get("title") or "").lower()
        if any(keyword in title for keyword in SURVEY_TITLE_KEYWORDS): return paper["cited_by_count"] > 10
        if self.academic_engine_type != "openalex":
            publication_types = {str(item).lower() for item in paper.get("publicationTypes", []) or []}
            publication_types |= {str(item).lower() for item in paper.get("publication_types", []) or []}
            if "review" in publication_types: return paper["cited_by_count"] > 10
        return False

    async def __call__(self, query: str):
        try:
            surveys_raw = await self._search_surveys(query, self.limit)
        except Exception as exc:
            print(f"referenceSurveySearch {exc}")
            surveys_raw = []

        review_like = [x for x in surveys_raw if self._is_review_like(x)]
        print(f"referenceSurvey: {len(review_like)} candidate surveys")
        if not review_like:
            return {"reference_surveys": []}
        try:
            selected_candidates = await self.survey_select.call(inputs={"query": query, "surveys": review_like})
        except Exception as exc:
            print(f"referenceSurveySelect {exc}")
            selected_candidates = []
        prints = "\n".join([
            f'- [{x.get("reference_survey_tier", "")}] {x["title"]}'
            for x in selected_candidates
        ]) if selected_candidates else "0"
        print(f"Selected referenceSurvey = {prints}")
        if not selected_candidates:
            return {"reference_surveys": []}
        return {"reference_surveys": selected_candidates}
