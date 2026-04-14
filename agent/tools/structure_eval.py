import asyncio
import jsonschema
from typing import List, Dict, Any

from .prompts import *
from .tool_config import ToolConfig
from .llmclient import AsyncChat
from .fact_check import FactCheckLLMClient
from .utils import extract_json, iter_sections, paragraph_to_text


class MethodClient(AsyncChat):
    PROMPT: str = EXTRACT_METHODS

    def _availability(self, response, context):
        data = extract_json(response)
        methods = []
        section_sentences = context["section"]
        for item in data.get("introduce_spans", []):
            span = item.get("span") or item.get("introduce_span")
            ref_key = item.get("ref_key")
            if not span or not ref_key or len(span) != 2:
                continue
            start, end = span
            if not (1 <= start < end <= len(section_sentences)):
                continue
            text = " ".join(section_sentences[idx - 1]["text"] for idx in range(start, end + 1))
            methods.append({"key": ref_key, "sentences": text})
        return {"section_name": context["name"], "methods": methods}

    def _organize_inputs(self, inputs):
        sentences = "\n".join(f"Sentence {idx}: {sentence['text']}" for idx, sentence in enumerate(inputs["section"], 1))
        citation_keys = []
        seen = set()
        for sentence in inputs["section"]:
            for citation in sentence.get("citations", []):
                if citation not in seen:
                    seen.add(citation)
                    citation_keys.append(f"- {citation}")
        prompt = self.PROMPT.format(text=sentences, keys="\n".join(citation_keys))
        return prompt, {"section": inputs["section"], "name": inputs["name"]}


class SectionOrganizeLLMClient(AsyncChat):
    PROMPT: str = SECTION_ORGANIZE

    def _availability(self, response, context):
        data = extract_json(response)
        organization_type = data.get("organization_type", "no_clear_structure")
        selected_methods = data.get("selected_methods", [])
        if organization_type != "no_clear_structure" and len(set(selected_methods)) < max(1, context["num_works"] / 2):
            organization_type = "no_clear_structure"
        return {
            "organization_type": organization_type,
            "selected_methods": selected_methods,
            "justification": data.get("justification", ""),
            "num_works": context["num_works"],
        }

    def _organize_inputs(self, inputs):
        methods_text = "\n".join(
            f"- id: M{idx}\n  reference_key: {method['key']}\n  related_text: {method['sentences']}"
            for idx, method in enumerate(inputs["methods"], 1)
        )
        prompt = self.PROMPT.format(
            text=f"Section name: {inputs['section_name']}\nMethods discussed in this section:\n{methods_text}"
        )
        return prompt, {"num_works": len(inputs["methods"])}


class PaperOrganizeLLMClient(AsyncChat):
    PROMPT: str = PAPER_ORGANIZE

    def _availability(self, response, context):
        data = extract_json(response)
        organization_type = data.get("organization_type", "no_clear_structure")
        selected_sections = data.get("selected_sections", [])
        if organization_type != "no_clear_structure" and len(set(selected_sections)) < max(1, context["num_sections"] / 2):
            organization_type = "no_clear_structure"
        return {
            "organization_type": organization_type,
            "selected_sections": selected_sections,
            "justification": data.get("justification", ""),
        }

    def _organize_inputs(self, inputs):
        sections_text = "\n".join(
            f"- Section S{idx}:\n  - title: {section['name']}\n  - methods_count: {section['num_works']}\n  - organization_type: {section['organization_type']}"
            for idx, section in enumerate(inputs, 1)
        )
        return self.PROMPT.format(text=sections_text), {"num_sections": len(inputs)}


class MissingTopicLLMClient(FactCheckLLMClient):
    PROMPT: str = MISSING_TOPIC_CLAIM

    def _availability(self, response, context):
        data = extract_json(response)
        if data.get("has_claim") and data.get("evidence"):
            evidence = data["evidence"] if isinstance(data["evidence"], list) else [data["evidence"]]
            verified, _ = self.check.verify(evidence, context["text"])
            if verified:
                return True
        return False

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(**inputs), {"text": inputs["text"]}


class StructureCheck:
    def __init__(self, config: ToolConfig):
        self.method_llm = MethodClient(config.llm_server_info, config.sampling_params)
        self.section_organize_llm = SectionOrganizeLLMClient(config.llm_server_info, config.sampling_params)
        self.paper_organize_llm = PaperOrganizeLLMClient(config.llm_server_info, config.sampling_params)
        self.missing_topic_llm = MissingTopicLLMClient(config)

    def _leaf_sections(self, paper: Dict[str, List]):
        for section in iter_sections(paper):
            if section.get("paragraphs"):
                sentences = []
                for paragraph in section.get("paragraphs", []):
                    sentences.extend(paragraph)
                if sentences:
                    yield section.get("title", ""), sentences

    async def _single_section(self, name: str, section: List[Dict[str, Any]]):
        methods = await self.method_llm.call(inputs={"name": name, "section": section})
        if not methods["methods"]:
            return None
        organize = await self.section_organize_llm.call(inputs=methods)
        organize["name"] = name
        return organize

    async def _structural_check(self, paper: Dict[str, List]):
        tasks = [asyncio.create_task(self._single_section(name, section)) for name, section in self._leaf_sections(paper)]
        method_organize = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
            except Exception:
                result = None
            if result:
                method_organize.append(result)

        strong_sections = [item for item in method_organize if item["num_works"] >= 3]
        if strong_sections:
            unclear = [item for item in strong_sections if item["organization_type"] == "no_clear_structure"]
            if len(unclear) / len(strong_sections) >= 0.8:
                return {"status": False, "reason": "sections_no_clear_structure", "details": {"unclear": len(unclear), "total": len(strong_sections)}}

        if not method_organize:
            return {"status": False, "reason": "no_method_introduction_sections", "details": None}

        try:
            paper_organize = await self.paper_organize_llm.call(inputs=method_organize)
        except Exception as exc:
            return {"status": False, "reason": "paper_organization_error", "details": str(exc)}
        if paper_organize["organization_type"] == "no_clear_structure":
            return {"status": False, "reason": "paper_no_clear_structure", "details": paper_organize.get("justification", "")}
        return {"status": True, "details": {"section_results": method_organize, "paper_organization": paper_organize}}

    async def _missing_topic_claim(self, topic: str, paper: Dict[str, Any]):
        text_blocks = []
        if paper.get("abstract"):
            text_blocks.append("\n\n".join(paragraph_to_text(p) for p in paper["abstract"]))
        for section in [paper, *iter_sections(paper)]:
            section_text = "\n\n".join(paragraph_to_text(p) for p in section.get("paragraphs", []))
            if section_text:
                text_blocks.append(section_text)
        for block in text_blocks:
            try:
                result = await self.missing_topic_llm.call(inputs={"topic": topic, "text": block})
            except Exception:
                result = False
            if result:
                return None
        return topic

    async def __call__(self, paper: Dict[str, List], missing_topics: List[str]) -> Dict[str, List]:
        structure = await self._structural_check(paper)
        tasks = [asyncio.create_task(self._missing_topic_claim(topic, paper)) for topic in missing_topics]
        real_missing_topics = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                real_missing_topics.append(result)
        return {"structure_evals": {"missing_topics": real_missing_topics, "structure_check": structure}}
