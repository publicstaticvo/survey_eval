import re
import json
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dynamic_oracle_generator import DynamicOracleGenerator

from utils import split_content_to_paragraph
from llm_server import ConcurrentLLMClient
from pydantic import BaseModel
from prompts import CLAIM_SEGMENTATION_PROMPT


class ClaimSegmentationLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\{.+?\}", re.DOTALL | re.IGNORECASE)
    PROMPT: str = CLAIM_SEGMENTATION_PROMPT

    def __init__(self, llm, sampling_params, n_workers, retry = 5):
        super().__init__(llm, sampling_params, n_workers, retry)

    def _pattern_check(self, output: str, citations: dict):
        try:
            pattern = self.format_pattern.findall(output)[-1]
            claim = json.loads(pattern)
            assert all(x in claim for x in ['claim', 'claim_type', 'requires'])
            assert set(claim['requires'].keys()) == set(citations.keys())
            return claim
        except:
            return

    def run_llm(self, inputs):
        retry = 5
        message = self.PROMPT.format(text=inputs['text'], range=inputs['range'])
        while retry and (claim := self._pattern_check(super().run_llm(message)), inputs['citations']) is None: 
            retry -= 1
        if not claim: return
        claim['citations'] = inputs['citations']
        # 将FULL_TEXT、TITLE_AND_ABSTRACT、TITLE_ONLY转化为0/1/2，与status对应。
        for x in claim['requires']:
            if claim['requires'][x].upper() == "TITLE_ONLY": claim['requires'][x] = 2
            elif claim['requires'][x].upper() == "TITLE_AND_ABSTRACT": claim['requires'][x] = 1
            # 默认为FULL_TEXT
            else: claim['requires'][x] = 0
        return claim


class SegmentationInput(BaseModel):
    review_text: list[dict[str, str]] = Field(description="The full text of the literature review to be evaluated.")


class ClaimSegmentation(BaseTool):
    name = "claim_segmentation_classification"
    description = (
        "Segments the review text into atomic claim blocks and classifies them. "
        "Classifies each as 'SINGLE_CLAIM' (one source), 'SYNTHESIS_CLAIM' (multi-source relation), "
        "or 'SERIAL_CLAIMS' (list of independent facts). "
        "Returns a structured list of claims with their citations."
    )
    args_schema: type[BaseModel] = SegmentationInput
    
    def __init__(self, llm: ClaimSegmentationLLMClient, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
    
    def _run(self, paper_content: dict[str, str]):
        # 第一步：分割成段落
        paragraphs = split_content_to_paragraph(paper_content)
        # 第二步：提取段落中的引用及句子
        citations = []
        for p in paragraphs:
            p_text = " ".join(s['text'] for s in p)
            for s in p: citations.append({"text": s['text'], "citations": s['citations'], "range": p_text})
        # 第三步：并行调用
        return self.llm.run_parallel(citations)
    
    async def _arun(self, paper_content: dict[str, str]) -> str:
        return self._run(paper_content)
