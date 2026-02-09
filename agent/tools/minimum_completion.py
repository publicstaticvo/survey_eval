from typing import List, Dict, Any

from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .fact_check import FactCheckLLMClient
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .utils import extract_json


def to_sections(section: Dict[str, Any]):
    text = "\n\n".join(" ".join(x['text'] for x in p) for p in section['paragraphs'])
    for s in section['sections']:
        text += "\n\n" + to_sections(s)
    return text


class IntegrateClient(FactCheckLLMClient):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT
    KEY: str = 'integration_intent'
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(text=inputs), {"text": inputs}


class StructureClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response, context):
        response = extract_json(response)
        assert isinstance(response['topic_driven'], bool)
        for x in response['discussion_sections']:
            assert any(x in y for y in context['titles'])
        return {"topic_driven": response['topic_driven'], "has_discussion": bool(response['discussion_sections'])}
    
    def _organize_inputs(self, inputs):
        titles = '\n'.join(inputs)
        return self.PROMPT.format(titles=titles), {"titles": titles}


class MinimalCompletionCheck:
    
    def __init__(self, config: ToolConfig):
        self.integration_intent_llm = IntegrateClient(config)
        self.structure_llm = StructureClient(config.llm_server_info, config.sampling_params)

    async def _integration_intent(self, paper: Dict[str, Any]):
        abstract_text = "\n\n".join(" ".join(x['text'] for x in p) for p in paper['abstract'])
        intent = await self.integration_intent_llm.call(inputs=abstract_text)
        if intent: return True

        if paper['paragraphs']:
            paragraph_text = "\n\n".join(" ".join(x['text'] for x in p) for p in paper['paragraphs'])
            intent = await self.integration_intent_llm.call(inputs=paragraph_text)
            if intent: return True

        for s in paper['sections']:
            section_text = to_sections(s)
            intent = await self.integration_intent_llm.call(inputs=section_text)
            if intent: return True
        
        return False
    
    def _section_citations_count(self, paper: Dict[str, Any]):
        
        def _recursive_citation_count(section: Dict[str, Any]):
            cites = set()
            for p in section['paragraphs']:
                for s in p:
                    cites.update(s['citations'])
            for s in section['sections']:
                cites.update(_recursive_citation_count(s))
            return cites
        
        citation_counts = []
        for i, section in paper['sections']:
            if "conclusion" in section['title'].lower() or "appendix" in section['title'].lower(): continue
            cites = _recursive_citation_count(section)
            section_text = to_sections(section)
            citation_counts.append({
                "id": i,
                "name": section_text['title'],
                "count": len(cites), 
                "tokens": len(x for x in section_text.split() if x)
            })

        low = [x for x in citation_counts if x['count'] < x['tokens'] // 200]
        return low
    
    async def _structural_judge(self, paper: Dict[str, Any]):
        def _get_titles(paper: Dict[str, Any], title_idx: str = ""):
            titles = [f"{title_idx} {paper['title']}"] if title_idx else []
            for i, s in enumerate(paper['sections'], 1):
                titles.extend(_get_titles(s, f"{title_idx}{i}."))
            return titles

        titles = _get_titles(paper)
        results = await self.structure_llm.call(inputs=titles)
        return results
    
    async def __call__(self, paper: Dict[str, Any]) -> Dict[str, List]:
        if not await self._integration_intent(paper):
            return {"minimum_check": {"status": "fail", "stage": "integration", "details": None}}
        
        low_citation_counts = self._section_citations_count(paper)
        if len(low_citation_counts) >= 2:
            return {"minimum_check": {"status": "fail", "stage": "cites", "details": low_citation_counts}}
        
        structure_judge = self._structural_judge(paper)
        if not structure_judge['topic_driven']:
            return {"minimum_check": {"status": "fail", "stage": "topic-driven", "details": None}}
        if not structure_judge['has_discussion']:
            return {"minimum_check": {"status": "fail", "stage": "conclusion", "details": None}}
        
        return {"minimum_check": {"status": "pass"}}