import asyncio
import json

from .utils import split_content_to_paragraph, extract_json
from .request_utils import AsyncLLMClient
from .prompts import CLAIM_SEGMENTATION_PROMPT
from .tool_config import ToolConfig


class ClaimSegmentationLLMClient(AsyncLLMClient):

    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def _availability(self, response: str):
        claim = extract_json(response)
        assert all(x in claim for x in ['claim', 'claim_type', 'requires'])
        
        citations = self._context['citations']
        assert set(claim['requires'].keys()) == set(citations.keys())
        claim['citations'] = citations
        # 将FULL_TEXT、TITLE_AND_ABSTRACT、TITLE_ONLY转化为0/1/2，与status对应。
        for x in claim['requires']:
            if claim['requires'][x].upper() == "TITLE_ONLY": claim['requires'][x] = 2
            elif claim['requires'][x].upper() == "TITLE_AND_ABSTRACT": claim['requires'][x] = 1
            # 默认为FULL_TEXT
            else: claim['requires'][x] = 0
        claim['paragraph_id'] = self._context['paragraph_id']
        return claim
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(text=inputs['text'], range=inputs['range'])


class ClaimSegmentation:
    
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers)
    
    async def __call__(self, paper_content: dict[str, str]):
        # 第一步：分割成段落
        paragraphs = split_content_to_paragraph(paper_content)
        # 第二步：提取段落中的引用及句子
        tasks = []
        for i, p in enumerate(paragraphs):
            p_text = " ".join(s['text'] for s in p)
            for s in p: 
                inputs = {"text": s['text'], "citations": s['citations'], "range": p_text}
                tasks.append(asyncio.create_task(self.llm.call(inputs=inputs, citations=input['citations'], context={"paragraph_id": i})))
        # 第三步：并行调用
        claims = []
        count = 0
        for task in asyncio.as_completed(tasks):
            try:
                x = await task
                if isinstance(x, dict): claims.append(x)
                else: count += 1
            except Exception as e:
                count += 1
        return {"claims": claims, "errors": count}
