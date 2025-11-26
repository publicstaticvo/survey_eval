import re
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from llm_server import ConcurrentLLMClient


class SynthesisLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\\boxed\{(supported|refuted|not[^0-9a-zA-Z]?mentioned|neutral)\}", 
                                            re.DOTALL | re.IGNORECASE)
    PROMPT: str = """..."""

    def __init__(self, llm, sampling_params, n_workers, retry = 5):
        super().__init__(llm, sampling_params, n_workers, retry)

    def _pattern_check(self, output):
        try:
            return self.format_pattern.findall(output)[-1].lower()
        except:
            return

    def run_llm(self, inputs):
        retry = 3
        message = self.PROMPT.format(**inputs)
        while retry and (pattern := self._pattern_check(super().run_llm(message))) is None: retry -= 1
        return pattern


class SynthesisCriticInput(BaseModel):
    claim_text: str = Field(..., description="The synthesis claim statement.")
    citation_ids: List[str] = Field(..., description="List of paper IDs involved in the synthesis.")
    paper_texts: Dict[str, str] = Field(..., description="Map of {id: text} for all involved papers.")


class SynthesisCorrectnessCritic(BaseTool):
    name = "synthesis_correctness_critic"
    description = (
        "Verifies a SYNTHESIS_CLAIM connecting multiple sources. "
        "Checks if the stated relationship (e.g., contrast, extension) is supported "
        "by the content of the cited papers."
    )
    args_schema: type[BaseModel] = SynthesisCriticInput
    
    def __init__(self, llm: SynthesisLLMClient, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def _run(self, claim: str, cited_papers: Dict[str, str]):
        """
        Implementation: Directly calls LLM.
        """
        # TODO：如何将cited_papers处理成str？
        inputs = {"claim": claim, "cited_papers": cited_papers}
        result = self.llm.run_llm(inputs)
        if result == "supported": inputs['result'] = "SUPPORTED"
        elif result == "refuted": inputs['result'] = "REFUTED"
        else: inputs['result'] = "NEUTRAL"
        return inputs
    
    async def _arun(self, claim: str, paper_texts: Dict[str, str]) -> str:
        return await self._run(claim, paper_texts)
