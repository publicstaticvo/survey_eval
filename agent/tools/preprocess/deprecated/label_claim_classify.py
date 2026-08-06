from __future__ import annotations

import asyncio
import jsonschema

from ..prompts import (
    COMPARISON_CLASSIFICATION,
    COMPARISON_SCHEMA,
    GAP_CLASSIFICATION,
    GAP_SCHEMA,
)
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import split_content_to_paragraph


class ComparisonClassificationClient(AsyncChat):
    PROMPT = COMPARISON_CLASSIFICATION

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.evidence_check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, COMPARISON_SCHEMA)
        if not result["excluded"]:
            for item in result["comparisons"]:
                verified, _ = self.evidence_check.verify(
                    [item["verbatim_evidence"]], context["sentence"], min_char_len=8
                )
                assert verified, "Comparison evidence must be copied verbatim from the input sentence"
        return result

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(S=inputs["sentence"], CONTEXT=inputs.get("context", "")), {
            "sentence": inputs["sentence"],
        }


class GapClassificationClient(AsyncChat):
    PROMPT = GAP_CLASSIFICATION

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.evidence_check = EvidenceCheck(config)

    def _availability(self, response, context):
        result = extract_json(response)
        jsonschema.validate(result, GAP_SCHEMA)
        if not result["excluded"]:
            verified, _ = self.evidence_check.verify(
                [result["verbatim_evidence"]], context["sentence"], min_char_len=8
            )
            assert verified, "Gap evidence must be copied verbatim from the input sentence"
        return result

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(S=inputs["sentence"], CONTEXT=inputs.get("context", "")), {
            "sentence": inputs["sentence"],
        }


class _SentenceFieldClassification:
    label = ""

    def __init__(self, config: ToolConfig, client_type):
        self.client = client_type(config)
        self.last_report = {"module": self.label.casefold(), "success_count": 0, "error_count": 0, "errors": []}

    async def __call__(self, paper, only_missing: bool = False):
        targets = []
        field = self.label.casefold()
        for paragraph in split_content_to_paragraph(paper, include_abstract=True):
            sentences = [
                sentence
                for sentence in paragraph
                if sentence.environment_type == "text"
                and sentence.label == self.label
                and sentence.text.strip()
                and (not only_missing or field not in sentence.classified_fields)
            ]
            for index, sentence in enumerate(sentences):
                context = " ".join(item.text.strip() for item in sentences[max(0, index - 2):index + 3] if item is not sentence)
                targets.append((sentence, context))
        results = await asyncio.gather(*(
            self.client.call(inputs={"sentence": sentence.text, "context": context})
            for sentence, context in targets
        ), return_exceptions=True)
        successful = []
        errors = []
        for index, ((sentence, _), result) in enumerate(zip(targets, results)):
            if isinstance(result, Exception):
                errors.append({"index": index, "sentence": sentence.text[:200], "error": repr(result)})
            else:
                successful.append({"sentence": sentence.text, "result": result})
                if field not in sentence.classified_fields:
                    sentence.classified_fields.append(field)
        self.last_report = {"module": self.label.casefold(), "success_count": len(successful), "error_count": len(errors), "errors": errors}
        return successful


class ComparisonClassification(_SentenceFieldClassification):
    label = "COMPARISON"

    def __init__(self, config: ToolConfig):
        super().__init__(config, ComparisonClassificationClient)


class GapClassification(_SentenceFieldClassification):
    label = "GAP"

    def __init__(self, config: ToolConfig):
        super().__init__(config, GapClassificationClient)
