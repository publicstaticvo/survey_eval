import re
import asyncio
import jsonschema
from typing import Any, List, Dict

from .utils import split_content_to_paragraph, extract_json
from .prompts import CLAIM_SEGMENTATION_PROMPT, CLAIM_SCHEMA
from .tool_config import ToolConfig
from .llmclient import AsyncChat


def range_check(claim: str, paragraph: List[Dict[str, Any]], anchor_id: int) -> bool:
    """
    check if the claim's tokens are 
    - 100% from the paragraph and 
    - 95% from the anchor sentence and its adjacent sentences.
    - allow claim starting with "This paper ..."
    """

    def _tokenize(sequence: str) -> list[str]:
        raw = re.split(r"[\s]+", sequence)
        tokens = []
        for t in raw:        
            if t := re.sub(r"^[^\w]+|[^\w]+$", "", t.lower()): tokens.append(t)
        return tokens
    
    claim = claim.lower().replace("this paper", "").strip()
    claim_tokens = _tokenize(claim)
    paragraph_tokens = [_tokenize(s['text']) for s in paragraph]
    if not all(token in sum(paragraph_tokens, []) for token in claim_tokens): return False
    anchor_range = sum(paragraph_tokens[max(0, anchor_id - 1): anchor_id + 2], [])
    print(sum(1 for token in claim_tokens if token in anchor_range))
    return sum(1 for token in claim_tokens if token in anchor_range) / len(claim_tokens) < 0.95


class ClaimSegmentationLLMClient(AsyncChat):

    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def _availability(self, response: str, context: dict):
        claims = extract_json(response)
        # 1. check schema
        jsonschema.validate(claim, CLAIM_SCHEMA)
        # 2. range_check if claim is in paragraph
        for claim in claims:
            assert range_check(claim, context['paragraph'], context['sentence_id'])
            # 3. after prompt validation and improvement:
            #    ensure each claim has exactly 1 citation
            claim['citations'] = {k: context['citations'][k] for k in claim['citation_markers']}
        return claims
    
    def _organize_inputs(self, inputs):
        paragraph_text = " ".join(s['text'] for s in inputs['range'])
        prompt = self.PROMPT.format(text=inputs['text'], range=paragraph_text, keys=list(inputs['citations']))
        return prompt, {"citations": inputs['citations'], "paragraph": inputs['range'], "sentence_id": inputs['sentence_id']}


class ClaimSegmentation:
    
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params)
    
    async def __call__(self, paper_content: dict[str, Any]):
        """
        :param paper_content: This is the result of `Paper.get_skeleton()`
        Use split_content_to_paragraph to convert into list of paragraphs
        Each paragraph is a list of sentences
        Each sentence is {"text": "sentence_text", "citations": {"key1": {}, "key2": {}}}
        """
        paragraphs = split_content_to_paragraph(paper_content)
        tasks = []
        for i, p in enumerate(paragraphs):
            for j, s in enumerate(p): 
                if s['citations']:
                    inputs = {"text": s['text'], "citations": s['citations'], "range": p, "sentence_id": j}
                    tasks.append(asyncio.create_task(self.llm.call(inputs=inputs, context={"paragraph_id": i})))
        print(f"We have {len(tasks)} claims.")
        claims = []
        count = 0
        for task in asyncio.as_completed(tasks):
            try:
                x = await task
                if isinstance(x, list) and x: claims.extend(x)
                else: count += 1
            except Exception as e:
                print(f"Claim {e} {type(e)}")
                count += 1
        return {"claims": claims, "errors": count}
