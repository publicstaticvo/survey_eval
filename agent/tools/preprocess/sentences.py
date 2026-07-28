import asyncio
import logging
from typing import Any

import jsonschema, json

from ..prompts import SENTENCE_CLASSIFICATION_PARAGRAPH, PARAGRAPH_SCHEMA, RULES
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Sentence
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json
from ..utility.content_walk import split_content_to_paragraph


class SentenceClassificationParagraph(AsyncChat):
    PROMPT: str = SENTENCE_CLASSIFICATION_PARAGRAPH

    def _availability(self, response: str, context: dict):
        result = extract_json(response)
        if not result.get('results') and result.get('result'):
            result = {'results': result['result']}
        jsonschema.validate(result, PARAGRAPH_SCHEMA)
        try:
            assert len(result["results"]) == len(context["sentences"]), "SentenceClassificationParagraph: result length mismatch"
        except AssertionError:
            print(f"{len(context['sentences'])} sentences:")
            count = 0
            for sentence in context["sentences"]:
                print(f"- {sentence.text}")
                count += int(sentence.environment_type == "text")
            print(f"{len(result['results'])} results")
            for item in result["results"]:
                print(json.dumps(item, ensure_ascii=False))
            raise
        return [(item["label"], float(item["confidence"])) for item in result["results"]]

    def _organize_inputs(self, inputs):
        sentences = inputs["sentences"]
        prompt = self.PROMPT.format(
            P="\\n".join(f"- {sentence.text}" for sentence in sentences),
            length=len(sentences), RULE=RULES
        )
        return prompt, {"sentences": sentences}


class SentenceClassification:
    def __init__(self, config: ToolConfig):
        self.paragraph_llm = SentenceClassificationParagraph(config.llm_server_info, config.sampling_params)

    def _is_classifiable(self, sentence: Sentence) -> bool:
        return sentence.environment_type == "text" and bool(sentence.text.strip())

    async def _classify_paragraph(self, paragraph: list[Sentence]):
        sentences = [sentence for sentence in paragraph if self._is_classifiable(sentence)]
        if not sentences: return []
        labels = await self.paragraph_llm.call(inputs={"sentences": sentences})
        return [(sentence, label, confidence) for sentence, (label, confidence) in zip(sentences, labels)]

    async def __call__(self, paper: Paper) -> Paper:
        paragraphs = split_content_to_paragraph(paper, include_abstract=True)
        tasks = [asyncio.create_task(self._classify_paragraph(paragraph)) for paragraph in paragraphs]
        logging.info(f"sentence classify for paper {paper.title}")
        for paragraph_result in await asyncio.gather(*tasks, return_exceptions=True):
            if not isinstance(paragraph_result, list): continue
            for sentence, label, confidence in paragraph_result:
                sentence.label = label
                sentence.confidence = confidence
        return paper
