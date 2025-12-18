import asyncio
import json

from .utils import split_content_to_paragraph, extract_json
from .request_utils import AsyncLLMClient
from .prompts import CLAIM_SEGMENTATION_PROMPT
from .tool_config import ToolConfig


class ClaimSegmentationLLMClient(AsyncLLMClient):

    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def _availability(self, response: json):
        response = response["choices"][0]["message"]["content"]
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
        return claim

    async def format_and_call(self, inputs):
        message = self.PROMPT.format(text=inputs['text'], range=inputs['range'])
        return await self.call(messages=message, citations=input['citations'])


class ClaimSegmentation:
    
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params, config.llm_num_workers)
    
    async def __call__(self, paper_content: dict[str, str]):
        # 第一步：分割成段落
        paragraphs = split_content_to_paragraph(paper_content)
        # 第二步：提取段落中的引用及句子
        citations = []
        for p in paragraphs:
            p_text = " ".join(s['text'] for s in p)
            for s in p: 
                inputs = {"text": s['text'], "citations": s['citations'], "range": p_text}
                citations.append(asyncio.create_task(self.llm.format_and_call(inputs)))
        # 第三步：并行调用
        results = await asyncio.gather(*citations, return_exceptions=True)
        claims = []
        count = 0
        for x in results:
            if isinstance(x, BaseException):
                count += 1
            elif isinstance(x, dict):
                claims.append(x)
        return {"claims": claims, "errors": count}
