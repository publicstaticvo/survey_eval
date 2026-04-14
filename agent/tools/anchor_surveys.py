import re
import json
import math
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

    def _flatten_title_paths(self, paper_skeleton: dict):
        titles = []

        def _walk(section: dict, parent_titles: list[str]):
            title = (section.get("title") or "").strip()
            if title:
                title_path = " > ".join([*parent_titles, title])
                lowered = title_path.lower()
                if self._check_title(lowered):
                    titles.append(title_path)
                parent_titles = [*parent_titles, title]
            for child in section.get("sections", []):
                _walk(child, parent_titles)

        for section in paper_skeleton.get("sections", []):
            _walk(section, [])
        return titles
    
    def _post_hook(self, xml_content: str):
        try:
            paper = self.paper_parser.parse(xml_content, mode="strict")
            paper_skeleton = paper.get_skeleton()
            titles = set(self._flatten_title_paths(paper_skeleton))
            print(f"This survey has {len(titles)} titles")
            return list(titles), paper_skeleton
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
        high = []
        for k, v in has_titles.items():
            titles = "\n".join(f"- {title}" for title in v["titles"])
            high.append(f"Survey title: {k}\nSection titles:\n{titles}\n")
        prompt = self.PROMPT.format(query=inputs['query'], anchors='\n'.join(high), surveys="\n".join(f'- {x}' for x in other_papers))
        # print(prompt)
        return prompt, {}


class AnchorSurveyFetch:
    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"

    def __init__(self, config: ToolConfig):
        self.eval_date = config.evaluation_date
        self.survey_download = SurveyDownload(config.grobid_url)
        self.paper_select = AnchorPaperSelect(config.llm_server_info, config.sampling_params)
        self.survey_select = AnchorSurveySelect(config.llm_server_info, config.sampling_params)

    def _citation_count_by_eval_date(self, paper: dict):
        eval_year = int(self.eval_date.year)
        citation_count = paper.get("cited_by_count", 0) or 0
        if citation_count:
            for item in paper.get("counts_by_year", []):
                if item["year"] > eval_year:
                    citation_count -= item["cited_by_count"]
            return citation_count
        return sum(item["cited_by_count"] for item in paper.get("counts_by_year", []) if item["year"] <= eval_year)

    def _is_survey(self, paper: dict):
        return len(paper.get("referenced_works", [])) >= 40 and self._citation_count_by_eval_date(paper) >= 20

    async def _download_surveys(self, papers: list[dict]):
        downloaded = {}
        for paper in papers:
            try:
                survey = await self.survey_download.download_single_paper(paper)
            except Exception:
                survey = None
            if isinstance(survey, tuple):
                titles, paper_skeleton = survey
                if titles and paper_skeleton:
                    downloaded[paper["title"]] = {"titles": titles, "skeleton": paper_skeleton, "paper": paper}
        return downloaded

    async def _anchor_papers(self, papers: dict, survey_title: str) -> list[dict]:
        try:
            selected = await self.paper_select.call(inputs={"query": survey_title, "papers": papers})
        except Exception:
            selected = list(papers.values())[:40]
        return selected or list(papers.values())[:40]

    async def _get_paper_meta_by_id(self, papers: list[str], batch_size: int = 50):
        paper_meta = {}
        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            if not batch:
                continue
            try:
                results = await openalex_search_paper("works", filter={"openalex": "|".join(batch)}, per_page=batch_size)
            except Exception:
                continue
            for paper in results.get("results", []):
                paper_meta[paper["id"]] = paper
        return paper_meta

    async def _anchor_survey(self, survey_candidates: list[dict], survey_title: str):
        real_surveys = [paper for paper in survey_candidates if self._is_survey(paper)]
        if not real_surveys:
            return {}, {}
        try:
            selected = await self.survey_select.call(inputs={"query": survey_title, "surveys": real_surveys})
        except Exception:
            selected = real_surveys[:5]
        downloaded = await self._download_surveys(selected[:10])
        selected_titles = set(downloaded)
        selected = [paper for paper in selected if paper["title"] in selected_titles]
        citation_counter = {}
        for paper in selected:
            for ref in paper.get("referenced_works", []):
                citation_counter[ref] = citation_counter.get(ref, 0) + 1
        threshold = max(2, math.ceil(0.6 * max(1, len(selected))))
        anchor_ids = [paper_id for paper_id, count in citation_counter.items() if count >= threshold]
        anchor_meta = await self._get_paper_meta_by_id(anchor_ids)
        for paper_id, metadata in anchor_meta.items():
            metadata["survey_cited_by_count"] = citation_counter[paper_id]
        return anchor_meta, downloaded

    async def __call__(self, survey_papers: list[dict], library: dict, survey_title: str):
        domain_papers = await self._anchor_papers(library, survey_title)
        anchor_papers, anchor_surveys = await self._anchor_survey(survey_papers, survey_title)
        return {
            "anchor_papers": anchor_papers,
            "surveys": domain_papers,
            "downloaded": anchor_surveys,
        }
