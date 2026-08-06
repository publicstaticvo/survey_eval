from __future__ import annotations

import asyncio
import logging
from typing import Any

import jsonschema

from ..utility.content_walk import paragraph_to_text
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json


SECTION_LABELS = {"SCOPE", "BACKGROUND", "CONTENT", "FUTURE_WORK", "CONCLUSION"}
SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "functional_type": {"enum": sorted(SECTION_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["functional_type", "confidence"],
    "additionalProperties": False,
}

PROMPT = """### Task
Classify the rhetorical function of one section of an academic survey. Use the section title, its position in the document, its parent section type, and its opening text. A subsection normally inherits the rhetorical context of its parent, but may receive a different label when its own purpose clearly differs.

### Labels
- SCOPE: the survey's literature search, inclusion/exclusion, filtering, selection, or coverage boundary.
- BACKGROUND: prerequisites, definitions, motivation, or field context that prepares the reader; an Introduction subsection remains BACKGROUND unless it actually reviews a research area.
- CONTENT: organized review of specific methods, systems, datasets, tasks, or research subareas.
- FUTURE_WORK: open problems, limitations of the field, or future research directions.
- CONCLUSION: a closing synthesis of the survey without substantial new review content.

Do not label a section CONTENT merely because its title is technical. In an Introduction or Background branch, use BACKGROUND unless the opening text shows that the section systematically reviews prior work. Do not infer SCOPE from a topic boundary such as “bounded domains”; SCOPE requires a boundary of this survey's literature selection.

### Input
Document title: "{DOCUMENT_TITLE}"
Section path: "{SECTION_PATH}"
Parent type: "{PARENT_TYPE}"
Section title: "{SECTION_TITLE}"
Opening text: "{PREAMBLE}"

### Output
Return JSON only: {{"functional_type": "SCOPE|BACKGROUND|CONTENT|FUTURE_WORK|CONCLUSION", "confidence": 0.0}}"""


class HierarchicalSectionClient(AsyncChat):
    """Use the parent section type as explicit context for section classification."""

    def _availability(self, response: str, context: dict[str, Any]) -> dict[str, Any]:
        result = extract_json(response)
        jsonschema.validate(result, SECTION_SCHEMA)
        return result

    def _organize_inputs(self, inputs: dict[str, Any]):
        prompt = PROMPT.format(
            DOCUMENT_TITLE=inputs.get("document_title", ""),
            SECTION_PATH=" > ".join(inputs.get("section_path", [])),
            PARENT_TYPE=inputs.get("parent_type", ""),
            SECTION_TITLE=inputs.get("section_title", ""),
            PREAMBLE=inputs.get("preamble", ""),
        )
        return prompt, {}


class HierarchicalSectionClassification:
    """Classify sections breadth-first so each child receives its parent's result."""

    def __init__(self, config: ToolConfig):
        self.llm = HierarchicalSectionClient(config.llm_server_info, config.sampling_params)
        self.last_report = {"module": "section_hierarchical", "success_count": 0, "error_count": 0, "errors": []}

    @staticmethod
    def _preamble(section: Section) -> str:
        parts = []
        for paragraph in section.paragraphs[:2]:
            text = paragraph_to_text(paragraph, include_environments=False)
            if text:
                parts.append(text)
        return " ".join(parts)[:4000]

    def _children(self, paper: Paper) -> list[tuple[Section, list[str], str]]:
        return [(section, [section.name], "") for section in paper.children if section.name]

    async def __call__(self, paper: Paper, only_missing: bool = False) -> Paper:
        pending = self._children(paper)
        successes, layer = 0, 0
        errors = []
        while pending:
            layer += 1
            active = [item for item in pending if not only_missing or not item[0].functional_type]
            # skipped = [item for item in pending if item not in active]
            tasks = [
                asyncio.create_task(self.llm.call(inputs={
                    "document_title": paper.title,
                    "section_path": path,
                    "parent_type": parent_type,
                    "section_title": section.name,
                    "preamble": self._preamble(section),
                }))
                for section, path, parent_type in active
            ]
            logging.info(f"Section Classify Layer {layer} with {len(tasks)} tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
            next_pending = []
            for (section, path, parent_type), result in zip(active, results):
                if isinstance(result, Exception):
                    errors.append({"section": " > ".join(path), "error": repr(result)})
                    continue
                functional_type = result["functional_type"]
                if len(path) == 2 and path[0].strip().lower() == "introduction" and functional_type == "CONTENT":
                    functional_type = "BACKGROUND"
                section.functional_type = functional_type
                successes += 1
            for section, path, _ in pending:
                if section.children:
                    next_pending.extend(
                        (child, [*path, child.name], section.functional_type)
                        for child in section.children
                        if child.name
                    )
            pending = next_pending
        self.last_report = {"module": "section_hierarchical", "success_count": successes, "error_count": len(errors), "errors": errors}
        logging.info("hierarchical section classification: %d successes, %d errors", successes, len(errors))
        return paper

