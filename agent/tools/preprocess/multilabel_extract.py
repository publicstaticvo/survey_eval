from __future__ import annotations

import asyncio
import json
import tqdm
import logging
from typing import Any

import jsonschema

from ..utility.content_walk import split_content_to_paragraph
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json

LABELS = ("CONTRIBUTION", "TEXTUAL", "SCOPE", "SYNTHESIS", "COMPARISON", "FUTURE_WORK")

_PROMPT = r'''### Task
You are an academic survey-structure analyst. Read the complete paragraph below and independently extract every expressed instance of the six requested functions. A paragraph may contain several functions, and one function may span several items; preserve all relevant item_ids in one match. Do not extract SUMMARY or BACKGROUND unless the passage also performs SYNTHESIS. Do not infer an instance that is not expressed.

### Evidence and identifiers
Every match must contain item_id as an array of one or more exact input identifiers. Every match must contain verbatim_evidence copied exactly from the supplied items; do not paraphrase, normalize, or return only a trigger phrase. For a multi-item match, join the copied evidence with a newline. The evidence must be sufficient for an independent reader to inspect the decision.

### Functions and fields
1. CONTRIBUTION: a document-level statement of what the current survey or paper contributes, proposes, defines, categorizes, introduces, analyzes, or demonstrates. Extract only one field: promised_scope, the complete contribution promise, including its action and object. Do not extract a statement about a prior work. A statement about a table or section is TEXTUAL unless it explicitly claims that the current paper introduces the underlying contribution.
2. TEXTUAL: a statement about what a concrete paper component does or contains, including a section, subsection, figure, table, equation, appendix, or list. Return component, component_target, and function. Cross-section references are TEXTUAL when the sentence asserts that another component contains or develops information; the evidence is the complete sentence, not only the section number. A table or caption is TEXTUAL when it describes how the table or figure organizes, summarizes, or illustrates the paper. Do not label a metric, method, dataset, task, or research object as TEXTUAL unless the passage explicitly states that a paper component contains, presents, or refers to it.
3. SCOPE: a statement about the survey's literature selection, retrieval, filtering, inclusion, exclusion, or coverage boundary. Return polarity, dimension, boundary, and search_method. polarity is inclusion or exclusion. dimension is one or more of time, venue, database, language, topic, document_type, or selection_criterion. A domain property such as bounded domain, open domain, or specialized domain is not SCOPE unless it is explicitly tied to which literature this survey selects.
4. SYNTHESIS: an integrative or interpretive statement that combines multiple prior works, methods, topics, results, or background facts into a higher-level relation or insight. Return synthesis_type, integrated_objects, and insight. integrated_objects must contain only the objects that participate in the integration, not every noun phrase. Valid synthesis_type values include trend, tradeoff, taxonomy, generalization, evaluation, mechanism, field_framing, and other. A single-work summary, a list of facts, or ordinary background is not SYNTHESIS.
5. COMPARISON: a substantive comparison that states how two or more research objects differ, relate, or perform along an explicit dimension. Return comparison_targets, comparison_dimensions, and comparison_relation. Mere juxtaposition, historical change, the word contrast, or a sentence mentioning a comparative experiment without stating its relation is not COMPARISON.
6. FUTURE_WORK: a prospective research direction or open problem that calls for future investigation or development at the field, subfield, task, or topic level. Return future_work_scope, future_work_type, and target. Do not extract a factual limitation of one work, a criticism with no future direction, or a bare word such as without. Use future_work_scope values field, subfield, task, topic, or work; work-level limitations are normally excluded unless the passage explicitly turns them into a future research direction.

### Positive examples
CONTRIBUTION: "We introduce a multilingual benchmark for evaluating factual consistency in generated summaries." -> {"label":"CONTRIBUTION","item_id":["ex-c1"],"verbatim_evidence":"We introduce a multilingual benchmark for evaluating factual consistency in generated summaries.","promised_scope":"introduce a multilingual benchmark for evaluating factual consistency in generated summaries"}
TEXTUAL: "Figure 2 illustrates how retrieved evidence flows through the verification pipeline." -> {"label":"TEXTUAL","item_id":["ex-t1"],"verbatim_evidence":"Figure 2 illustrates how retrieved evidence flows through the verification pipeline.","component":"figure","component_target":"Figure 2","function":"illustrates the verification pipeline"}
SCOPE: "We include studies published between 2018 and 2024 retrieved from ACL Anthology." -> {"label":"SCOPE","item_id":["ex-s1"],"verbatim_evidence":"We include studies published between 2018 and 2024 retrieved from ACL Anthology.","polarity":"inclusion","dimension":["time","database"],"boundary":"studies published between 2018 and 2024","search_method":"retrieval from ACL Anthology"}
SYNTHESIS: "Across retrieval-based and generative systems, greater response diversity improves engagement but makes factual verification more difficult." -> {"label":"SYNTHESIS","item_id":["ex-y1"],"verbatim_evidence":"Across retrieval-based and generative systems, greater response diversity improves engagement but makes factual verification more difficult.","synthesis_type":"tradeoff","integrated_objects":["retrieval-based systems","generative systems","response diversity","user engagement","factual verification"],"insight":"Greater response diversity can improve engagement while making factual verification more difficult."}
COMPARISON: "Retrieval-based systems are more controllable than generative systems, but they provide less diverse responses." -> {"label":"COMPARISON","item_id":["ex-m1"],"verbatim_evidence":"Retrieval-based systems are more controllable than generative systems, but they provide less diverse responses.","comparison_targets":["retrieval-based systems","generative systems"],"comparison_dimensions":["controllability","response diversity"],"comparison_relation":"Retrieval-based systems are more controllable, whereas generative systems provide more diverse responses."}
FUTURE_WORK: "Future work should develop evaluation protocols for long-term user adaptation in conversational agents." -> {"label":"FUTURE_WORK","item_id":["ex-f1"],"verbatim_evidence":"Future work should develop evaluation protocols for long-term user adaptation in conversational agents.","future_work_scope":"field","future_work_type":"future_direction","target":"evaluation protocols for long-term user adaptation"}

### Negative examples
Do not label "Task-oriented systems operate in bounded domains" as SCOPE: bounded domains describe the research object, not the survey's literature selection. Do not label "Model X fails to preserve long-range coherence" as FUTURE_WORK unless the passage explicitly proposes future research to address it. Do not label "Table 4 presents a taxonomy" as CONTRIBUTION unless the passage states that the current paper introduces or proposes that taxonomy. Do not label "Early systems used pattern matching without semantic understanding" as FUTURE_WORK: it is a description of a prior work.

### Input
{paragraph}

### Output
Return JSON only with this shape: {"matches":[{"label":"CONTRIBUTION|TEXTUAL|SCOPE|SYNTHESIS|COMPARISON|FUTURE_WORK","item_id":["p0-s0"],"verbatim_evidence":"...", "...label-specific fields..."}]}. Return an empty matches array when no requested function is expressed. Do not include source_text because the input paragraph is already stored by its parent document.
'''

FIELD_SCHEMA = {
    "CONTRIBUTION": {"promised_scope": "string"},
    "TEXTUAL": {"component": "string", "component_target": "string", "function": "string"},
    "SCOPE": {"polarity": "string", "dimension": "array", "boundary": "string", "search_method": "string"},
    "SYNTHESIS": {"synthesis_type": "string", "integrated_objects": "array", "insight": "string"},
    "COMPARISON": {"comparison_targets": "array", "comparison_dimensions": "array", "comparison_relation": "string"},
    "FUTURE_WORK": {"future_work_scope": "string", "future_work_type": "string", "target": "string"},
}


def _schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "label": {"type": "string", "enum": list(LABELS)},
        "item_id": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "verbatim_evidence": {"type": "string", "minLength": 1},
        "evidence_valid": {"type": "boolean"},
        "evidence_confidence": {"type": "number"},
        "environment_type": {"type": "string"},
    }
    for fields in FIELD_SCHEMA.values():
        for name, kind in fields.items():
            properties[name] = {"type": "array", "items": {"type": "string"}} if kind == "array" else {"type": "string"}
    return {"type": "object", "required": ["matches"], "additionalProperties": False, "properties": {"matches": {"type": "array", "items": {"type": "object", "required": ["label", "item_id", "verbatim_evidence"], "additionalProperties": False, "properties": properties}}}}


class ParagraphMultiLabelExtractor(AsyncChat):
    """Extract all six rhetorical functions from one complete paragraph."""

    def __init__(self, config: ToolConfig):
        sampling_params = dict(config.sampling_params)
        sampling_params["max_tokens"] = min(int(sampling_params.get("max_tokens", 4096)), 4096)
        super().__init__(config.llm_server_info, sampling_params)
        self.evidence_check = EvidenceCheck(config)
        self.schema = _schema()

    def _organize_inputs(self, inputs: dict[str, Any]):
        return _PROMPT.replace("{paragraph}", inputs["paragraph"]), {"items": inputs["items"]}

    def _availability(self, response: str, context: dict[str, Any]):
        result = extract_json(response)
        array_fields = {"dimension", "integrated_objects", "comparison_targets", "comparison_dimensions"}
        for match in result.get("matches", []):
            if isinstance(match.get("item_id"), str):
                match["item_id"] = [match["item_id"]]
            for field in array_fields:
                if isinstance(match.get(field), str):
                    match[field] = [match[field]]
                elif isinstance(match.get(field), dict):
                    match[field] = [f"{key}: {value}" for key, value in match[field].items()]
        item_map = {item["item_id"]: item for item in context["items"]}
        valid_matches = []
        for match in result.get("matches", []):
            ids = [str(item_id).split("|", 1)[0].strip() for item_id in match.get("item_id", [])]
            match["item_id"] = ids
            label = match.get("label")
            if label not in FIELD_SCHEMA or not ids or any(item_id not in item_map for item_id in ids):
                logging.warning("discarded match with unavailable or invalid ids: %s", ids)
                continue
            for name in FIELD_SCHEMA[label]:
                if name not in match:
                    raise jsonschema.ValidationError(f"missing field {name} for {label}")
            evidence = match["verbatim_evidence"]
            source = "\n".join(item_map[item_id]["text"] for item_id in ids)
            valid, confidence = self.evidence_check.verify([evidence], source, min_char_len=8)
            match["evidence_valid"] = valid
            match["evidence_confidence"] = confidence
            match["environment_type"] = ",".join(dict.fromkeys(item_map[item_id]["environment_type"] for item_id in ids))
            valid_matches.append(match)
        result["matches"] = valid_matches
        jsonschema.validate(result, self.schema)
        return result


class MultiLabelExtraction:
    """Run one complete-paragraph six-label extraction per paragraph."""

    def __init__(self, config: ToolConfig):
        self.extractor = ParagraphMultiLabelExtractor(config)
        self.last_report = {"module": "multilabel_extract", "success_count": 0, "error_count": 0, "errors": []}

    @staticmethod
    def _items(paragraph: list[Any], paragraph_index: int) -> list[dict[str, str]]:
        items = []
        for index, sentence in enumerate(paragraph):
            text = (sentence.text or "").strip()
            if not text or sentence.environment_type == "paragraph_name":
                continue
            items.append({"item_id": f"p{paragraph_index}-s{index}", "text": text, "environment_type": sentence.environment_type})
        return items

    async def _run_paragraph(self, paragraph: list[Any], paragraph_index: int):
        items = self._items(paragraph, paragraph_index)
        if not items:
            return paragraph_index, {}, []
        rendered = "\n".join(f"[{item['item_id']}|{item['environment_type']}] {item['text']}" for item in items)
        try:
            result = await self.extractor.call(inputs={"paragraph": rendered, "items": items})
        except Exception as exc:
            return paragraph_index, {}, [{"paragraph": paragraph_index, "error": repr(exc)}]
        extracted: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
        for match in result.get("matches", []):
            extracted[match["label"]].append(match)
        return paragraph_index, extracted, []

    async def __call__(self, paper, only_missing: bool = False):
        paragraphs = split_content_to_paragraph(paper, include_abstract=True, include_appendix=False)
        errors: list[dict[str, Any]] = []
        success = 0
        tasks = [asyncio.create_task(self._run_paragraph(paragraph, index)) for index, paragraph in enumerate(paragraphs)]
        checkpoint_path = getattr(self, "checkpoint_path", None)
        completed_count = 0
        for completed in tqdm.tqdm(asyncio.as_completed(tasks), desc="MultiLabelExtract", total=len(tasks)):
            paragraph_index, extracted, paragraph_errors = await completed
            errors.extend(paragraph_errors)
            paragraph = paragraphs[paragraph_index]
            by_id = {f"p{paragraph_index}-s{index}": sentence for index, sentence in enumerate(paragraph)}
            for label, matches in extracted.items():
                for match in matches:
                    target_sentences = [by_id[item_id] for item_id in match["item_id"] if item_id in by_id]
                    if not target_sentences:
                        errors.append({"paragraph": paragraph_index, "label": label, "error": "unknown item after extraction"})
                        continue
                    if only_missing and any(label in sentence.label_extractions for sentence in target_sentences):
                        continue
                    for sentence in target_sentences:
                        sentence.label_extractions.setdefault(label, []).append(match)
                    success += 1
            completed_count += 1
            if checkpoint_path and (completed_count % 8 == 0 or completed_count == len(tasks)):
                with open(checkpoint_path, "w", encoding="utf-8") as handle:
                    json.dump(paper.get_skeleton(), handle, ensure_ascii=False, indent=2, default=str)
            # logging.info("multilabel paragraph %d complete: %d matches", paragraph_index, sum(len(v) for v in extracted.values()))
        self.last_report = {"module": "multilabel_extract", "success_count": success, "error_count": len(errors), "errors": errors}
        logging.info("multilabel extraction: %d matches, %d errors", success, len(errors))
        return paper
