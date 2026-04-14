from typing import Any, Dict, List

from .evidence_check import EvidenceCheck
from .llmclient import AsyncChat
from .prompts import CORE_ARGUMENT_PROMPT, MAIN_CONTRIBUTION_PROMPT, RESEARCH_GAP_PROMPT
from .tool_config import ToolConfig
from .utils import extract_json, get_first_section, safe_text, section_to_text


class EvidenceBackedExtractionClient(AsyncChat):
    PROMPT = ""

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _normalize_item(self, item: Dict[str, Any], source_text: str) -> Dict[str, Any] | None:
        if not item or not item.get("statement"):
            return None
        evidence = item.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        verified, score = self.check.verify(evidence, source_text)
        if not verified:
            return None
        return {
            "statement": item["statement"].strip(),
            "evidence": evidence,
            "score": score,
        }


class CoreArgumentClient(EvidenceBackedExtractionClient):
    PROMPT = CORE_ARGUMENT_PROMPT

    def _availability(self, response, context):
        data = extract_json(response)
        return self._normalize_item(data.get("item"), context["source_text"])

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(text=inputs["text"], few_shot_examples=inputs.get("few_shot_examples", ""))
        return prompt, {"source_text": inputs["text"]}


class MainContributionClient(EvidenceBackedExtractionClient):
    PROMPT = MAIN_CONTRIBUTION_PROMPT

    def _availability(self, response, context):
        data = extract_json(response)
        items = []
        for item in data.get("items", []):
            normalized = self._normalize_item(item, context["source_text"])
            if normalized:
                items.append(normalized)
        return items

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(text=inputs["text"], few_shot_examples=inputs.get("few_shot_examples", ""))
        return prompt, {"source_text": inputs["text"]}


class ResearchGapClient(EvidenceBackedExtractionClient):
    PROMPT = RESEARCH_GAP_PROMPT

    def _availability(self, response, context):
        data = extract_json(response)
        research_gaps, future_directions = [], []
        for item in data.get("research_gaps", []):
            normalized = self._normalize_item(item, context["source_text"])
            if normalized:
                research_gaps.append(normalized)
        for item in data.get("future_directions", []):
            normalized = self._normalize_item(item, context["source_text"])
            if normalized:
                future_directions.append(normalized)
        return {"research_gaps": research_gaps, "future_directions": future_directions}

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(text=inputs["text"], few_shot_examples=inputs.get("few_shot_examples", ""))
        return prompt, {"source_text": inputs["text"]}


class ArgumentStructureEvaluator:
    def __init__(self, config: ToolConfig):
        self.core_argument_client = CoreArgumentClient(config)
        self.main_contribution_client = MainContributionClient(config)
        self.research_gap_client = ResearchGapClient(config)

    def _abstract_intro_text(self, paper: Dict[str, Any]) -> str:
        parts = []
        if paper.get("title"):
            parts.append(f"Title: {paper['title']}")
        if paper.get("abstract"):
            parts.append("Abstract:\n" + safe_text(paper["abstract"]))
        first_section = get_first_section(paper)
        if first_section:
            parts.append(f"Introduction Candidate ({first_section.get('title', '')}):\n{section_to_text(first_section)}")
        return "\n\n".join(part for part in parts if part)

    def _find_section_by_id(self, paper: Dict[str, Any], section_id: Any) -> Dict[str, Any] | None:
        target = str(section_id)

        def _walk(section: Dict[str, Any]) -> Dict[str, Any] | None:
            if str(section.get("section_id")) == target:
                return section
            for child in section.get("sections", []):
                found = _walk(child)
                if found is not None:
                    return found
            return None

        for section in paper.get("sections", []):
            found = _walk(section)
            if found is not None:
                return found
        return None

    def _gap_text(self, paper: Dict[str, Any], discussion_section_candidates: List[Dict[str, Any]] | None = None) -> str:
        parts = [self._abstract_intro_text(paper)]
        included_ids = set()

        top_level_sections = paper.get("sections", [])
        if top_level_sections:
            last_section = top_level_sections[-1]
            last_section_id = str(last_section.get("section_id", ""))
            included_ids.add(last_section_id)
            parts.append(f"Last Chapter ({last_section.get('title', '')}):\n{section_to_text(last_section)}")

        for candidate in discussion_section_candidates or []:
            candidate_id = str(candidate.get("section_id", ""))
            if not candidate_id or candidate_id in included_ids: continue
            section = self._find_section_by_id(paper, candidate_id)
            if section is None: continue
            included_ids.add(candidate_id)
            parts.append(f"Candidate Discussion Section ({candidate.get('title_path') or candidate.get('title', '')}):\n{section_to_text(section)}")
        return "\n\n".join(part for part in parts if part)

    async def __call__(
        self,
        paper: Dict[str, Any],
        minimum_check_details: Dict[str, Any] | None = None,
        few_shot_examples: Dict[str, str] | None = None,
    ):
        few_shot_examples = few_shot_examples or {}
        minimum_check_details = minimum_check_details or {}
        abstract_intro = self._abstract_intro_text(paper)
        gap_text = self._gap_text(paper, minimum_check_details.get("discussion_section_candidates", []))

        core_argument = await self.core_argument_client.call(
            inputs={"text": abstract_intro, "few_shot_examples": few_shot_examples.get("core_argument", "")}
        )
        contributions = await self.main_contribution_client.call(
            inputs={"text": abstract_intro, "few_shot_examples": few_shot_examples.get("contributions", "")}
        )
        gaps = await self.research_gap_client.call(
            inputs={"text": gap_text, "few_shot_examples": few_shot_examples.get("research_gaps", "")}
        )

        return {
            "argument_evals": {
                "core_argument": core_argument,
                "main_contributions": contributions,
                "research_gaps": gaps["research_gaps"],
                "future_directions": gaps["future_directions"],
                "discussion_section_candidates": minimum_check_details.get("discussion_section_candidates", []),
            }
        }
