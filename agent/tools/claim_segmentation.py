import re
import asyncio
import jsonschema
from typing import Any, List, Dict

from .prompts import CLAIM_SEGMENTATION_PROMPT, CLAIM_SCHEMA, CLAIMS_SCHEMA
from .utils import split_content_to_paragraph, extract_json
from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig
from .llmclient import AsyncChat


def range_check(claim: str, paragraph: List[Dict[str, Any]], anchor_id: int) -> bool:
    """
    check if the claim's tokens are 
    - 100% from the paragraph and 
    - 90% from the anchor sentence and its adjacent sentences.
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
    # print(sum(1 for token in claim_tokens if token in anchor_range), len(claim_tokens), anchor_range)
    return sum(1 for token in claim_tokens if token in anchor_range) / len(claim_tokens) >= 0.9


class ClaimSegmentationLLMClient(AsyncChat):

    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def __init__(self, llm, sampling_params):
        super().__init__(llm, sampling_params)

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        if result["is_verifiable_performance_claim"]: return context['claim']
    
    def _organize_inputs(self, inputs):
        paragraph_text = "\n".join(s['text'] for s in inputs['range'])
        prompt = self.PROMPT.format(text=inputs['text'], range=paragraph_text, keys=list(x['key'] for x in inputs['citations']))
        return prompt, {'claim': inputs}


class ClaimSegmentation:
    
    def __init__(self, config: ToolConfig):
        sbert = SentenceTransformerClient(config.sbert_server_url)
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params)
    
    async def __call__(self, paper_content: dict[str, Any]):
        """
        :param paper_content: This is the result of `Paper.get_skeleton()`
        Use split_content_to_paragraph to convert into list of paragraphs
        Each paragraph is a list of sentences
        Each sentence is {"text": "sentence_text", "citations": [{"title": "title1", "key": "key1"}, {"title": "title1", "key": "key1"}]}
        """
        paragraphs = split_content_to_paragraph(paper_content)
        tasks, claims = [], []
        for i, p in enumerate(paragraphs):
            for j, s in enumerate(p): 
                # 在现在的实现逻辑下，只保留含1个引用的句子。
                if len(s['citations']) == 1:
                    inputs = {"text": s['text'], "citations": s['citations'], "range": p, "sentence_id": j}
                    tasks.append(asyncio.create_task(self.llm.call(inputs=inputs, context={"paragraph_id": i})))
        for task in asyncio.as_completed(tasks):
            try:
                x = await task
                if isinstance(x, dict) and x: claims.append(x)
            except Exception as e:
                print(f"Claim {e} {type(e)}")
        return {"claims": claims}
