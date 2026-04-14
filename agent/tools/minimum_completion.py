from typing import List, Dict, Any

from .prompts import *
from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .fact_check import FactCheckLLMClient
from .utils import extract_json, section_to_text, get_top_level_section_titles


class IntegrateClient(FactCheckLLMClient):

    PROMPT: str = INTERGRATION_INTENT
    KEY: str = 'integration_intent'
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(text=inputs), {"text": inputs}


class StructureClient(AsyncChat):
    PROMPT: str = STRUCTURE_AND_DISCUSSION

    def _availability(self, response, context):
        data = extract_json(response)
        discussion_sections = [title for title in data.get("discussion_sections", []) if title in context["titles"]]
        return {
            "topic_driven": bool(data.get("topic_driven")),
            "has_discussion": bool(discussion_sections),
            "discussion_sections": discussion_sections,
        }

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(titles="\n".join(inputs)), {"titles": inputs}


class MinimalCompletionCheck:
    def __init__(self, config: ToolConfig):
        self.integration_intent_llm = IntegrateClient(config)
        self.structure_llm = StructureClient(config.llm_server_info, config.sampling_params)

    async def _integration_intent(self, paper: Dict[str, Any]) -> bool:
        text_blocks = []
        if paper.get("title"):
            text_blocks.append(f"Title: {paper['title']}")
        if paper.get("abstract"):
            text_blocks.append(section_to_text({"paragraphs": paper["abstract"], "sections": []}))
        first_section = paper.get("sections", [])[:1]
        if first_section:
            text_blocks.append(section_to_text(first_section[0]))
        combined = "\n\n".join(block for block in text_blocks if block)
        if not combined:
            return False
        try:
            return bool(await self.integration_intent_llm.call(inputs=combined))
        except Exception:
            return False

    def _title_structure_check(self, paper: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        title = (paper.get("title") or "").lower()
        survey_markers = ["survey", "review", "overview", "taxonomy", "synthesis"]
        has_title_marker = any(marker in title for marker in survey_markers)
        titles = [t.lower() for t in get_top_level_section_titles(paper)]
        has_intro_like = any("intro" in t or "background" in t or "overview" in t for t in titles)
        return has_title_marker or has_intro_like, {"title_marker": has_title_marker, "top_level_titles": titles}

    def _flatten_sections(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        flattened = []

        def _walk(section: Dict[str, Any], parent_titles: List[str]):
            title = section.get("title", "").strip()
            title_path = [*parent_titles, title] if title else list(parent_titles)
            flattened.append(
                {
                    "section_id": section.get("section_id"),
                    "title": title,
                    "title_path": " > ".join(x for x in title_path if x),
                    "depth": len(title_path),
                }
            )
            for child in section.get("sections", []):
                _walk(child, title_path)

        for section in paper.get("sections", []):
            _walk(section, [])
        return flattened

    def _discussion_candidate_sections(self, flattened_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keywords = [
            "discussion",
            "future",
            "open problem",
            "open problems",
            "open question",
            "open questions",
            "challenge",
            "challenges",
            "limitation",
            "limitations",
            "conclusion",
            "conclusions",
            "outlook",
            "research agenda",
        ]
        candidates = []
        seen = set()
        for section in flattened_sections:
            haystack = f"{section.get('title', '')} {section.get('title_path', '')}".lower()
            if any(keyword in haystack for keyword in keywords):
                key = (section.get("section_id"), section.get("title_path"))
                if key not in seen:
                    seen.add(key)
                    candidates.append(section)
        return candidates

    async def __call__(self, paper: Dict[str, Any]) -> Dict[str, List]:
        structure_ok, structure_details = self._title_structure_check(paper)
        if not structure_ok:
            return {"minimum_check": {"status": "fail", "stage": "title_structure", "details": structure_details}}
        if not await self._integration_intent(paper):
            return {"minimum_check": {"status": "fail", "stage": "abstract_intent", "details": None}}
        flattened_sections = self._flatten_sections(paper)
        try:
            structure_judge = await self.structure_llm.call(inputs=[x["title_path"] or x["title"] for x in flattened_sections if x["title_path"] or x["title"]])
        except Exception:
            structure_judge = {"topic_driven": False, "has_discussion": False, "discussion_sections": []}
        if not structure_judge.get("topic_driven"):
            return {"minimum_check": {"status": "fail", "stage": "topic_driven", "details": None}}
        discussion_candidates = self._discussion_candidate_sections(flattened_sections)
        llm_discussion_titles = set(structure_judge.get("discussion_sections", []))
        if llm_discussion_titles:
            llm_candidates = [x for x in flattened_sections if (x["title_path"] in llm_discussion_titles or x["title"] in llm_discussion_titles)]
            by_key = {(x["section_id"], x["title_path"]): x for x in discussion_candidates}
            for candidate in llm_candidates:
                by_key[(candidate["section_id"], candidate["title_path"])] = candidate
            discussion_candidates = list(by_key.values())
        return {
            "minimum_check": {
                "status": "pass", 
                "details": {
                    "has_discussion": structure_judge.get("has_discussion", False),
                    "discussion_section_candidates": discussion_candidates,
                }
            }
        }
