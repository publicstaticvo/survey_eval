import re
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from survey_eval.agent.tools.citation_parser import PaperParser

from llm_server import ConcurrentLLMClient, LLMServerInfo
from utils import split_content_to_paragraph, paragraph_to_text, callLLM


class FactualLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\\boxed\{(supported|refuted|not[^0-9a-zA-Z]?mentioned|neutral)\}", 
                                            re.DOTALL | re.IGNORECASE)
    PROMPT: str = """..."""

    def __init__(
            self, 
            critic_llm: LLMServerInfo, 
            rerank_llm: LLMServerInfo, 
            sampling_params: Dict[str, Any], 
            n_workers: int, 
            num_selected_documents: int = 3,
            retry: int = 5
        ):
        super().__init__(critic_llm, sampling_params, n_workers, retry)
        self.num_selected_documents = num_selected_documents
        self.rerank_llm = rerank_llm

    def _pattern_check(self, output):
        try:
            return self.format_pattern.findall(output)[-1].lower()
        except:
            return

    def run_llm(self, claim: str, documents: list[str]) -> str:
        # 2. Rerank documents
        documents = list(set(documents))
        parameters = {"query": claim, "documents": documents, "top_n": self.num_selected_documents}
        rerank_results = callLLM(self.rerank_llm, "rerank", parameters, retry=self.retry, return_documents=True)
        # 3. request 
        retry = self.retry
        inputs = {"claim": claim, "text": rerank_results}
        message = self.PROMPT.format(**inputs)
        while retry and (pattern := self._pattern_check(super().run_llm(message))) is None: retry -= 1
        return pattern


class FactualCriticInput(BaseModel):
    claim_text: str = Field(..., description="The specific claim statement.")
    citation_id: str = Field(..., description="The ID of the cited paper.")
    paper_text: str = Field(..., description="The full text (or abstract) of the source paper.")


class FactualCorrectnessCritic(BaseTool):
    name = "factual_correctness_critic"
    description = (
        "Verifies a SINGLE_CLAIM against a single source. "
        "Retrieves relevant passages from 'paper_text' and determines if "
        "the claim is SUPPORTED, REFUTED, or NEUTRAL."
    )
    args_schema: type[BaseModel] = FactualCriticInput
    
    def __init__(self, llm: FactualLLMClient, **kwargs):
        super().__init__(**kwargs)
        self.prompt = """None"""
        self.llm = llm
    
    def _run(self, claim: str, cited_paper: Dict[str, Any]) -> str:
        """
        param cited_papers
        Key: paper_citation_key
        Value:
        - metadata: metadata
        - status: 0-3
        - abstract: abstract
        - full_content: full_content (Dict[str, str]) / abstract (str) / title (str)
        Implementation: 
        """
        # 1. Split cited_papers['full_content'] into paragraphs if it has
        documents = [cited_paper['title'], cited_paper['abstract']]        
        if isinstance(content := cited_paper['full_content'], dict):
            # Has full content
            documents += [paragraph_to_text(x) for x in split_content_to_paragraph(content)]
        # 2. Select related paragraphs from each evidence
        inputs = {"claim": claim, "documents": documents}
        result = self.llm.run_llm(claim, documents)
        if result == "supported": inputs['result'] = "SUPPORTED"
        elif result == "refuted": inputs['result'] = "REFUTED"
        else: inputs['result'] = "NEUTRAL"
        return inputs
    
    async def _arun(self, claim: str, cited_papers: Dict[str, str]) -> str:
        return await self._run(claim, cited_papers)
    

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

    def _run(self, claim: str, cited_papers: Dict[str, Dict[str, Any]]):
        """
        param cited_papers
        Key: paper_citation_key
        Value:
        - metadata: metadata
        - status: 0-3
        - abstract: abstract
        - full_content: full_content (Dict[str, str]) / abstract (str) / title (str)
        Implementation: 
        """
        # 1. Split cited_papers['full_content'] into paragraphs if it has
        cited_split = {}
        for k, v in cited_papers.items():
            if isinstance(v['full_content'], dict):
                # Has full content
                paragraphs = [paragraph_to_text(x) for x in split_content_to_paragraph(v['full_content'])]
                cited_split[k] = [v['full_content']['title'], v['full_content']['abstract']] + paragraphs
            else:
                # Does not full content
                cited_split[k] = [v['full_content']['title'], v['full_content']['abstract']]
        # 2. Select related paragraphs from each evidence
        inputs = {"claim": claim, "cited_papers": cited_split}
        result = self.llm.run_llm(inputs)
        if result == "supported": inputs['result'] = "SUPPORTED"
        elif result == "refuted": inputs['result'] = "REFUTED"
        else: inputs['result'] = "NEUTRAL"
        return inputs
    
    async def _arun(self, claim: str, paper_texts: Dict[str, str]) -> str:
        return await self._run(claim, paper_texts)
