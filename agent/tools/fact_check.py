from typing import List, Dict, Any

from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .evidence_check import EvidenceCheck
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .utils import extract_json, section_to_text, split_content_to_paragraph, paragraph_to_text


class FactCheckLLMClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT
    KEY: str = "judgment"

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        response = extract_json(response)
        if response[self.KEY]:
            assert self.check.verify(response['evidence'], context['text'])[0]
        return response[self.KEY]
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(text=inputs), {"text": inputs}


class FactualCorrectnessCritic:
    
    def __init__(self, config: ToolConfig):
        self.llm = FactCheckLLMClient(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    async def _judge_and_verify(self, claim: str, content: str, content_type: str):
        inputs = {"claim": claim, "text": content, "content_type": "title, abstract, and full text" if content_type == "full_text" else "title and abstract"}
        judgment, evidence = await self.llm.call(inputs=inputs)
        if judgment == "neutral": return judgment, evidence, 0, "insufficient information"
        verify_result, score = self.check.verify(evidence, content)
        if verify_result: return judgment, evidence, score, ""
        return judgment, evidence, score, "false judgment"
    
    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, List]:
        """
        param claim
        Each claim has exactly 1 citation, so claim is a string.
        param cited_paper
        Keys:
        - metadata: metadata
        - status: 0-3
        - abstract: abstract
        - full_content: full_content (Dict[str, str]) if has
        """
        content = f"{cited_paper['title']}. {cited_paper['abstract']}"
        material = "title_abstract"
        judgment, evidence, score, neutral_type = await self._judge_and_verify(claim, content, material)
        if judgment == "neutral" and cited_paper['full_content']:
            material = "full_text"
            for p in split_content_to_paragraph(cited_paper['full_content']):
                p = paragraph_to_text()
                content = f"Title: {cited_paper['title']}\nAbstract: {cited_paper['abstract']}\n\nFull Text:\n\n{p}"
            judgment, evidence, score, neutral_type = await self._judge_and_verify(claim, content, material)
        return {"fact_check": {
            "claim": claim, 
            "judgment": judgment,
            "evidence": evidence,
            "reason": neutral_type,
            "score": score,
            "material": material
        }}
