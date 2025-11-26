import re
import json
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dynamic_oracle_generator import DynamicOracleGenerator

from utils import split_content_to_paragraph
from llm_server import ConcurrentLLMClient
from pydantic import BaseModel


class ClaimSegmentationLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\{.+?\}", re.DOTALL | re.IGNORECASE)
    PROMPT: str = """..."""

    def __init__(self, llm, sampling_params, n_workers, retry = 5):
        super().__init__(llm, sampling_params, n_workers, retry)

    def _pattern_check(self, output):
        try:
            pattern = self.format_pattern.findall(output)[-1]
            claim = json.loads(pattern)
            assert 'claim_text' in claim and 'claim_type' in claim
            return claim
        except:
            return

    def run_llm(self, inputs):
        retry = 3
        message = self.PROMPT.format(text=inputs['text'], range=inputs['range'])
        while retry and (claim := self._pattern_check(super().run_llm(message))) is None: retry -= 1
        claim['citations'] = inputs['citations']
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
        # {
        #     "citation_ids": ["p1", "p3"],
        #     "claims": [
        #         {"text": "A did X [p1]", "type": "SINGLE_CLAIM", "citations": ["p1"]},
        #         {"text": "A implies B [p1, p3]", "type": "SYNTHESIS_CLAIM", "citations": ["p1", "p3"]},
        #         {"text": "Lists: A [p1], B [p3]", "type": "SERIAL_CLAIMS", "sub_claims": [
        #             {"text": "A [p1]", "citation": "p1"},
        #             {"text": "B [p3]", "citation": "p3"}
        #         ]}
        #     ]
        # }
    
    async def _arun(self, paper_content: dict[str, str]) -> str:
        return self._run(paper_content)
