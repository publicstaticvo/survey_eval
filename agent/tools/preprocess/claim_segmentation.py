import asyncio
import jsonschema
from typing import Any, List, Dict

from ..prompts import CLAIM_SEGMENTATION, CLAIM_SCHEMA, CLAIMS_SCHEMA
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from ..fact.utils import split_content_to_paragraph, extract_json, paragraph_to_text

TARGET_LABELS = {"BACKGROUND", "SUMMARY", "COMPARISON", "EVALUATION"}


class ClaimSegmentationLLMClient(AsyncChat):
    PROMPT: str = CLAIM_SEGMENTATION

    def _availability(self, response: str, context: dict):
        claims = extract_json(response)
        try:
            jsonschema.validate(claims, CLAIMS_SCHEMA)
        except jsonschema.ValidationError:
            jsonschema.validate(claims, CLAIM_SCHEMA)
            claims = {"claims": [claims]}
        marker_map = context["citations"]
        for claim in claims["claims"]:            
            markers = claim["citation_keys"]
            claim["citations"] = {
                int(marker): marker_map[int(marker)]
                for marker in markers
                if int(marker) in marker_map
            }
            if not claim["citations"]:
                claim["verifiable"] = False
            claim["paragraph_id"] = context["paragraph_id"]
        return [claim for claim in claims["claims"] if claim["verifiable"]]

    def _organize_inputs(self, inputs):
        paragraph_text = inputs["paragraph_text"]
        prompt = self.PROMPT.format(range=paragraph_text)
        return prompt, {"citations": inputs["citations"], "paragraph_id": inputs["paragraph_id"]}


class ClaimSegmentation:
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params)

    async def __call__(self, paper_content: dict[str, Any]):
        paragraphs = split_content_to_paragraph(paper_content)
        tasks, claims, count = [], [], 0
        for i, paragraph in enumerate(paragraphs):
            paragraph_citations = {}
            for s in paragraph['sentences']:
                if isinstance(s, dict) and s.get('environment_type', 'text') == 'text':
                    paragraph_citations.update(s.get('citations', {}))
            if not paragraph_citations: continue
            paragraph_text = paragraph_to_text(paragraph)
            tasks.append(asyncio.create_task(self.llm.call(inputs={"paragraph_text": paragraph_text, "citations": paragraph_citations, "paragraph_id": i})))
        for task in asyncio.as_completed(tasks):
            try:
                x = await task
                if isinstance(x, list) and x:
                    claims.extend(x)
                else:
                    count += 1
            except Exception as e:
                print(f"Claim {e} {type(e)}")
                count += 1
        print(f"We have {len(claims)} claims.")
        return {"claims": claims, "errors": count}