import asyncio
import jsonschema
from typing import List, Dict, Any

from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .fact_check import FactCheckLLMClient
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .utils import extract_json


class LandmarkLLMClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT
    ROLE: dict = {"foundational": 3, "representative": 2, "incremental": 1, "background": 0}

    def _availability(self, response, context):
        response = extract_json(response)
        return self.ROLE[response['role']]
    

class MethodClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response, context):
        response = extract_json(response)
        s = context['section']
        methods = []
        for x in response['introducte_spans']:
            span = x['introduce_span']
            if span[1] <= span[0]: continue
            for i in range(span[0] - 1, span[1]):
                assert i >= 0, i
                if x['ref_key'] in s[i]['citations']: break
            else:
                # f"No ref_key is in this span: {span}"
                continue
            methods.append({"key": x['ref_key'], "sentences": span[0] - 1})
        methods.sort(key=lambda x: x["sentences"])
        for m in methods:
            m['sentences'] = f"{s[m['sentences']]['text']} {s[m['sentences'] + 1]['text']}"
        return {"section_name": context['name'], "methods": methods}
    
    def _organize_inputs(self, inputs):
        organized_str, refs, refs_set = [], [], set()
        for i, x in enumerate(inputs['section'], 1):
            organized_str.append(f"Sentence {i}: {x['text']}")
            for c in x['citations']:
                if c not in refs_set:
                    refs.append(f"- {c}")
                    refs_set.add(c)
        return self.PROMPT.format(text="\n".join(organized_str), keys="\n".join(refs)), {"section": inputs['section']}


class SectionOrganizeLLMClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response, context):
        response = extract_json(response)
        if response['organization_type'] != "no clear structure":
            schema = {
                "type": "object",
                "required": ['organization_type', 'selected_methods', 'justification'],
                "properties": {
                    'organization_type': {"type": "string", "enum": ["grouping by criteria", "chronological or technical progression", "explicit comparison"]},
                    'selected_methods': {"type": "array", "items": {"type": "string", "enum": [f"M{i + 1}" for i in range(context['num_works'])]}},
                    'justification': {"type": "string", "minLength": 1}
                }
            }
            jsonschema.validate(response, schema)
            if len(set(response['selected_methods'])) < context['num_works'] / 2:
                response['organization_type'] = "no clear structure"
        response['num_works'] = context['num_works']
        return response
    
    def _organize_inputs(self, inputs):
        string = []
        for i, m in enumerate(inputs['methods'], 1):
            string.append(f"- id: M{i}\n  citation: {m['key']}\n  related_text: {m['sentences']}")
        string = "\n".join(string)
        inputs = f"Section name: {inputs['section_name']}\nMethods discussed in this section:\n{string}"
        return self.PROMPT.format(text=inputs), {"num_works": len(inputs['methods'])}
    

class PaperOrganizeLLMClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response, context):
        response = extract_json(response)
        if response['organization_type'] != "no clear structure":
            schema = {
                "type": "object",
                "required": ['organization_type', 'selected_sections', 'justification'],
                "properties": {
                    'organization_type': {"type": "string", "enum": ["grouping by criteria", "chronological or technical progression", "explicit comparison"]},
                    'selected_sections': {"type": "array", "items": {"type": "string", "enum": [f"S{i + 1}" for i in range(context['num_sections'])]}},
                    'justification': {"type": "string", "minLength": 1}
                }
            }
            jsonschema.validate(response, schema)
            if len(set(response['selected_sections'])) < context['num_works'] / 2:
                response['organization_type'] = "no clear structure"
        else: assert response['justification']
        return response
    
    def _organize_inputs(self, inputs):
        string = []
        for i, s in enumerate(inputs, i):
            string.append(f"Section S{i}: {s['name']}\n- Methods count: {s['num_works']}\n- Organization type: {s['organization_type']}")
        string = "\n".join(string)
        return self.PROMPT.format(text=string), {"num_sections": len(inputs)}
    

class RefuteOrganizationLLMClient(FactCheckLLMClient):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT
    KEY: str = "refute"

    def _availability(self, response, context):
        response = extract_json(response)
        if response[self.KEY]:
            assert self.check.verify(response['evidence'], context['text'])[0]
        return response
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {"text": inputs['text']}


class MissingTopicLLMClient(FactCheckLLMClient):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT
    KEY: str = "has_claim"
    
    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {"text": inputs['text']}


class QualityPressureCheck:
    
    def __init__(self, config: ToolConfig):
        self.landmark_llm = LandmarkLLMClient(config.llm_server_info, config.sampling_params)
        self.method_llm = MethodClient(config.llm_server_info, config.sampling_params)
        self.section_organize_llm = SectionOrganizeLLMClient(config.llm_server_info, config.sampling_params)
        self.paper_organize_llm = PaperOrganizeLLMClient(config.llm_server_info, config.sampling_params)
        self.refute_llm = RefuteOrganizationLLMClient(config)
        self.missing_topic_llm = MissingTopicLLMClient(config)

    async def _landmark(self, anchor: str, paper: Dict[str, Any]):
        def _search_for_cite_context(anchor_paper: str, paper: Dict[str, Any]):
            contexts = []
            for p in paper['paragraphs']:
                for i, s in enumerate(p['sentences']):
                    for c in s['citations']:
                        if c['name'] == anchor_paper:
                            contexts.append(" ".join(p['sentences'][max(i, 0):i+2]))
                            break
            for s in paper['sections']:
                contexts.extend(_search_for_cite_context(anchor_paper, s))
            return contexts
        
        cite_contexts = _search_for_cite_context(anchor, paper)
        tasks = [asyncio.create_task(self.landmark_llm.call(inputs={"title": anchor, "text": c})) for c in cite_contexts]
        max_role = -1
        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    max_role = max(max_role, result)
                    if max_role >= 2:
                        for task in tasks:
                            if not task.done(): task.cancel()
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    continue
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if max_role > 2: return {"anchor": anchor, "landmark": True}
        return {"anchor": anchor, "landmark": False, "details": "incremental" if max_role == 1 else "background"}
    
    async def _missing_topic_claim(self, topic: str, paper: Dict[str, Any]):
        """
        Check if a missing topic is claimed in the paper. Accept if there is a claim.
        """
        def _yield_section(paper: dict):
            if 'abstract' in paper and paper['abstract']:
                abstract = "\n\n".join(" ".join(x['text'] for x in p) for p in paper['abstract'])
                yield abstract
            paragraphs = "\n\n".join(" ".join(x['text'] for x in p) for p in paper['paragraphs'])
            yield paragraphs
            for s in paper['sections']: _yield_section(s)

        for t in _yield_section(paper):
            result = await self.missing_topic_llm.call(inputs={"topic": topic, "text": t})
            if result: return
        return topic
    
    async def _structural_check(self, paper: Dict[str, List]):
        def _yield_section(paper: Dict[str, List]):
            if paper['sections']:
                for s in paper['sections']:
                    _yield_section(s)
            elif paper['paragraphs']:
                name = "" if "abstract" in paper else paper['title']
                section = []
                for p in paper['paragraphs']:
                    if p['type'] == "text": section.extend(p)
                yield name, section

        async def _single_section(name: str, section: List[Dict[str, Any]]):
            # 第一步：section内部抽取方法和介绍句。
            methods = await self.method_llm.call(inputs={"name": name, "section": section})
            # 第二步：LLM按照介绍句判断每个section内部所有方法的组织模式。
            organize = await self.section_organize_llm.call(inputs=methods)
            organize['name'] = name
            return organize
        
        def _get_introduction(paper: dict, section_idx: str = ""):
            text = []
            if not section_idx:
                text.append(f"Title: {text['title']}")
                if paper['abstract']:
                    abstract = "\n\n".join(" ".join(x['text'] for x in p) for p in paper['abstract'])
                    text.append(f"Abstract: {abstract}")
                text.extend(_get_introduction(s['sections'][0], "1"))
            else:
                text.append(f"Section {section_idx} {paper['title']}")
                for p in paper['paragraphs']:
                    text.append(" ".join(x['text'] for x in p))
                for i, s in enumerate(paper['sections'], 1): 
                    text.extend(_get_introduction(s, f"{section_idx}.{i}"))
            return text
        
        method_organize, tasks = [], []
        for n, s in _yield_section(paper):
            tasks.append(asyncio.create_task(_single_section(n, s)))
        for task in asyncio.as_completed(tasks):
            try:
                methods = await task
                if methods: method_organize.append(methods)
            except Exception as e:
                continue
        
        none_sections, total_sections = 0, 0
        for x in method_organize:
            if x['num_works'] >= 3:
                total_sections += 1
                if x['organization_type'] == "no clear structure":
                    none_sections += 1
        if total_sections > 0 and none_sections / total_sections >= 0.8:
            return {"status": False, "reason": "Sections no clear structure", "details": (none_sections, total_sections)}
        # 第三步：LLM判断整篇文章的组织形式。
        section_organize = await self.paper_organize_llm.call(inputs=method_organize)
        if section_organize['organization_type'] == "no clear structure":
            return {"status": False, "reason": "Paper no clear structure", "details": section_organize['justification']}
        # LLM判断整篇文章的组织形式是否与标题矛盾。
        introduction = "\n\n".join(_get_introduction(paper))
        refutes = self.refute_llm.call(inputs={"text": introduction, "organize": section_organize['organization_type']})
        if refutes['refute']:
            return {"status": False, "reason": "Paper refutes structure", "details": {"organization": section_organize, "evidence": refutes['evidence']}}
        return {"status": True}
    
    async def __call__(self, paper: Dict[str, List], anchors: Dict[str, List], missing_topics: List[str]) -> Dict[str, List]:
        landmarks = []
        for x in anchors:
            landmark_result = await self._landmark(x['title'], paper)
            if not landmark_result['landmark']: landmarks.append(landmark_result)

        structure = await self._structural_check(paper)

        tasks = [asyncio.create_task(self._missing_topic_claim(m, paper)) for m in missing_topics]
        real_missing_topics = []
        for task in tasks:
            result = await task
            if result: real_missing_topics.append(result)
        
        return {"quality": {
            "missing_topics": real_missing_topics,
            "structure_check": structure,
            "uncovered_landmarks": landmarks
        }}
