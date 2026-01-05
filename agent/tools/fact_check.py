import re
from typing import List, Dict, Any

from .tool_config import ToolConfig
from .llmclient import AsyncLLMClient
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .utils import split_content_to_paragraph, paragraph_to_text


class FactualReranker(AsyncLLMClient):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response):
        return [x['document']['text'] for x in response['results']]


class FactualLLMClient(AsyncLLMClient):

    format_pattern: re.Pattern = re.compile(r"\\boxed\{(supported|refuted|not[^0-9a-zA-Z]?mentioned|neutral)\}", 
                                            re.DOTALL | re.IGNORECASE)

    def _availability(self, response):
        return self.format_pattern.findall(response)[-1].lower()


class FactualCorrectnessCritic:
    
    def __init__(self, config: ToolConfig):
        self.reranker = FactualReranker(config.rerank_server_info)
        self.llm = FactualLLMClient(config.llm_server_info, config.sampling_params)
        self.rerank_n_documents = config.n_documents
    
    async def __call__(self, claim: str, cited_paper: Dict[str, Any]) -> Dict[str, List]:
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
        documents = [] 
        if isinstance(content := cited_paper['full_content'], dict):
            # Has full content
            documents = [paragraph_to_text(x) for x in split_content_to_paragraph(content)]
            # 2. Select related paragraphs from each evidence
            documents = list(set(documents))
            parameters = {"query": claim, "documents": documents, "top_n": self.rerank_n_documents}
            rerank_results = await self.reranker.call("rerank", **parameters, return_documents=True)
            # add: "This paper requires only *** to validate"
        # 3. Judge        
        text = f"Title: {cited_paper['title']}"
        if cited_paper['abstract']:
            text += f"\nAbstract: {cited_paper['abstract']}"
        if isinstance(content, dict):
            text += f"\nRetrieved related sentences: \n{'\n\n'.join(rerank_results)}"
        result = await self.llm.call(inputs={"claim": claim, "text": text})
        inputs = {"claim": claim, "documents": documents}
        if result == "supported": inputs['result'] = "SUPPORTED"
        elif result == "refuted": inputs['result'] = "REFUTED"
        else: inputs['result'] = "NEUTRAL"
        return {"fact_check": [inputs]}


class SynthesisCorrectnessCritic:
    
    def __init__(self, config: ToolConfig):
        self.reranker = FactualReranker(config.rerank_server_info)
        self.llm = FactualLLMClient(config.llm_server_info, config.sampling_params)
        self.rerank_n_documents = config.n_documents

    async def __call__(self, claim: str, cited_papers: Dict[str, Any]) -> Dict[str, List]:
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
        # TODO: cited_split是个dict，要如何变成list？
        parameters = {"query": claim, "documents": cited_split, "top_n": self.rerank_n_documents}
        rerank_results = await self.reranker.call("rerank", **parameters, return_documents=True)
        # 3. Judge            
        result = await self.llm.call(inputs={"claim": claim, "text": "\n\n".join(rerank_results)})
        inputs = {"claim": claim, "documents": cited_split}
        if result == "supported": inputs['result'] = "SUPPORTED"
        elif result == "refuted": inputs['result'] = "REFUTED"
        else: inputs['result'] = "NEUTRAL"
        return {"fact_check": [inputs]}
