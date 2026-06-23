import asyncio
import copy
from typing import Any

import jsonschema, json

from ..prompts import (
    SENTENCE_CLASSIFICATION_PARAGRAPH, 
    SENTENCE_CLASSIFICATION_SINGLE,
    SINGLE_SCHEMA, PARAGRAPH_SCHEMA, RULES
)
from ..utility.llmclient import AsyncChat
from ..utility.tool_config import ToolConfig
from .utils import extract_json, split_content_to_paragraph


class SentenceClassificationSingle(AsyncChat):
    PROMPT: str = SENTENCE_CLASSIFICATION_SINGLE

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, SINGLE_SCHEMA)
        return result["label"], float(result["confidence"])

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(S=inputs["text"], RULE=RULES), {}


class SentenceClassificationParagraph(AsyncChat):
    PROMPT: str = SENTENCE_CLASSIFICATION_PARAGRAPH

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        jsonschema.validate(result, PARAGRAPH_SCHEMA)
        try:
            assert len(result["results"]) == len(context["sentences"])
        except AssertionError:
            print(f"{len(context['sentences'])} sentences:")
            count = 0
            for x in context['sentences']: 
                print(f"- {x['text']}")
                count += int(x['environment_type'] == 'text')
            print(f"{len(result['results'])} results")
            for x in result['results']: print(json.dumps(x, ensure_ascii=False))
            raise
        return [(item["label"], float(item["confidence"])) for item in result["results"]]

    def _organize_inputs(self, inputs):
        sentences = inputs["sentences"]
        return self.PROMPT.format(
            P="\n".join(f"- {sentence["text"]}" for sentence in sentences),
            length=len(sentences), RULE=RULES
        ), {"sentences": sentences}


class SentenceClassification:
    def __init__(self, config: ToolConfig):
        self.single_llm = SentenceClassificationSingle(config.llm_server_info, config.sampling_params)
        self.paragraph_llm = SentenceClassificationParagraph(config.llm_server_info, config.sampling_params)

    def _is_classifiable(self, sentence: dict[str, Any]) -> bool:
        return sentence.get("environment_type", "text") == "text" and bool(sentence.get("text", "").strip())

    async def _classify_single_sentence(self, sentence: dict[str, Any]) -> tuple[str, float]:
        return await self.single_llm.call(inputs={"text": sentence.get("text", "")})

    async def _classify_sentence_single_mode(self, paper: dict[str, Any]) -> dict[str, Any]:
        paragraphs = split_content_to_paragraph(paper)
        if paper.get('abstract', []): paragraphs = [*paper['abstract']['paragraphs'], *paragraphs]
        classifiable_sentences = [
            sentence
            for paragraph in paragraphs
            for sentence in paragraph
            if isinstance(sentence, dict) and self._is_classifiable(sentence)
        ]
        tasks = [
            asyncio.create_task(self._classify_single_sentence(sentence)) 
            for sentence in classifiable_sentences
        ]
        labels = await asyncio.gather(*tasks, return_exceptions=True)
        for sentence, x in zip(classifiable_sentences, labels):
            if not isinstance(x, tuple): continue
            label, confidence = x
            sentence.update({"label": label, "confidence": confidence})
        return paper

    async def _classify_paragraph(self, paragraph: list[dict[str, Any]]):
        if isinstance(paragraph, dict):
            paragraph = paragraph.get("sentences", [])
        sentences = [
            sentence
            for sentence in paragraph
            if isinstance(sentence, dict) and self._is_classifiable(sentence)
        ]
        if not sentences: return []
        labels = await self.paragraph_llm.call(inputs={"sentences": sentences})
        return [(sentence, label, confidence) for sentence, (label, confidence) in zip(sentences, labels)]

    async def _classify_sentence_paragraph_mode(self, paper: dict[str, Any]) -> dict[str, Any]:
        paragraphs = split_content_to_paragraph(paper)
        if paper.get('abstract', []): paragraphs = [*paper['abstract']['paragraphs'], *paragraphs]
        tasks = [asyncio.create_task(self._classify_paragraph(paragraph)) for paragraph in paragraphs]
        for paragraph_result in await asyncio.gather(*tasks, return_exceptions=True):
            if not isinstance(paragraph_result, list): continue
            for sentence, label, confidence in paragraph_result:
                sentence.update({"label": label, "confidence": confidence})
        return paper

    async def __call__(self, paper_content: dict[str, Any]) -> dict[str, Any]:
        paper = copy.deepcopy(paper_content)
        # return await self._classify_sentence_single_mode(paper)
        return await self._classify_sentence_paragraph_mode(paper)
