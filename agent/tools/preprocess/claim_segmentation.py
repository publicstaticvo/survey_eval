from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import jsonschema

from ..prompts import CLAIM_SEGMENTATION, CLAIM_SCHEMA, CLAIMS_SCHEMA
from ..utility.citation_utils import citation_marker_key_map, has_citations
from ..utility.content_walk import iter_paragraphs, paragraph_sentences
from ..utility.llmclient import AsyncChat
from ..utility.paper_elements import Paper, Paragraph, Sentence
from ..utility.tool_config import ToolConfig
from ..utility.utils import extract_json, ENGLISH_STOPWORDS

try:
    import spacy
except ImportError:
    spacy = None

SPACY_NLP = None
if spacy is not None:
    try:
        SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        SPACY_NLP = None


SOURCE_LABELS = {"SUMMARY", "SYNTHESIS", "CONTRAST", "GAP"}


class ClaimSegmentationLLMClient(AsyncChat):
    PROMPT: str = CLAIM_SEGMENTATION

    def _lemma(self, token: str) -> str:
        token = token.casefold()
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ing"):
            stem = token[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(token) > 3 and token.endswith("ed"):
            stem = token[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _tokens(self, text: str) -> set[str]:
        if SPACY_NLP is not None:
            return {
                (token.lemma_ or token.text).casefold()
                for token in SPACY_NLP(text or "")
                if not token.is_stop and not token.is_punct and not token.is_space
            }
        return {
            self._lemma(token)
            for token in re.findall(r"[A-Za-z0-9]+", text or "")
            if token.casefold() not in ENGLISH_STOPWORDS
        }

    def _availability(self, response: str, context: dict):
        claims = extract_json(response)
        try:
            jsonschema.validate(claims, CLAIMS_SCHEMA)
        except jsonschema.ValidationError:
            jsonschema.validate(claims, CLAIM_SCHEMA)
            claims = {"claims": [claims]}

        source_sentences: dict[str, str] = context["source_sentences"]
        marker_map = {str(marker): key for marker, key in context["citation_marker_map"].items()}
        valid_source_ids = set(source_sentences)
        for claim in claims["claims"]:
            source_ids = [str(source_id) for source_id in claim["source_ids"]]
            assert set(source_ids) <= valid_source_ids, f"source ids: {source_ids} is not subset of valid source ids {valid_source_ids}"
            source_text = " ".join(source_sentences[source_id] for source_id in source_ids)
            source_tokens = self._tokens(source_text)
            claim_tokens = self._tokens(claim["claim"])
            # assert claim_tokens <= source_tokens, f"claim tokens: {claim_tokens} is not subset of source tokens {source_tokens}"
            if not (claim_tokens <= source_tokens): continue

            markers = [str(marker) for marker in claim["citation_keys"]]
            claim["citation_keys"] = markers
            claim["citations"] = {
                int(marker) if marker.isdigit() else marker: marker_map[marker]
                for marker in markers
                if marker in marker_map
            }
            claim["verifiable"] = bool(claim["verifiable"])
            claim["paragraph_id"] = context["paragraph_id"]
            claim["sources"] = [source_sentences[source_id] for source_id in source_ids]
            claim.pop("source_ids", None)
        return claims["claims"]

    def _citation_markers(self, sentence: Sentence) -> list[str]:
        return [str(marker) for marker in citation_marker_key_map(sentence.citations)]

    def _sentence_text(self, sentence: Sentence) -> str:
        text = (sentence.text or "").strip()
        if not text:
            return ""
        markers = self._citation_markers(sentence)
        missing_markers = [marker for marker in markers if marker not in text]
        if missing_markers:
            text = f"{text} [{', '.join(missing_markers)}]"
        if sentence.environment_type == "paragraph_name":
            return r"\paragraph{" + text + "}"
        return text

    def _should_mark_sentence(self, sentence: Sentence) -> bool:
        if has_citations(sentence.citations):
            return True
        return (sentence.label or "") in SOURCE_LABELS

    def _organize_inputs(self, inputs):
        paragraph = inputs["paragraph"]
        sentences = paragraph_sentences(paragraph)
        source_sentences: dict[str, str] = {}
        parts = []
        source_index = 1
        for sentence in sentences:
            if not isinstance(sentence, Sentence): continue
            text = self._sentence_text(sentence)
            if not text: continue
            if self._should_mark_sentence(sentence):
                source_id = str(source_index)
                source_index += 1
                source_sentences[source_id] = text
                parts.append(f'<E id="{source_id}">{text}</E>')
            else:
                parts.append(text)
        paragraph_text = " ".join(parts).strip()
        prompt = self.PROMPT.format(range=paragraph_text)
        return prompt, {
            "citation_marker_map": inputs["citation_marker_map"],
            "paragraph_id": inputs["paragraph_id"],
            "source_sentences": source_sentences,
        }


class ClaimSegmentation:
    def __init__(self, config: ToolConfig):
        self.llm = ClaimSegmentationLLMClient(config.llm_server_info, config.sampling_params)

    def _paragraph_citation_map(self, paragraph: Paragraph) -> dict[Any, str]:
        citation_map: dict[Any, str] = {}
        for sentence in paragraph.sentences:
            if sentence.environment_type == "text":
                citation_map.update(citation_marker_key_map(sentence.citations))
        return citation_map

    def _paragraph_entities(self, paragraph: Paragraph) -> list[str]:
        names = []
        for entity in paragraph.entities or []:
            if not isinstance(entity, dict):
                continue
            for name in [entity.get("name"), entity.get("alias_pairs"), *(entity.get("alternative_names", []) or [])]:
                name = str(name or "").strip()
                if name:
                    names.append(name)
        return list(dict.fromkeys(names))

    def _paragraph_has_source_sentence(self, paragraph: Paragraph) -> bool:
        for sentence in paragraph_sentences(paragraph):
            if not isinstance(sentence, Sentence):
                continue
            if self.llm._sentence_text(sentence) and self.llm._should_mark_sentence(sentence):
                return True
        return False

    def _organize_input_paragraph(self, paragraph: Paragraph) -> str:
        sentences = paragraph_sentences(paragraph)
        source_sentences: dict[str, str] = {}
        parts = []
        source_index = 1
        for sentence in sentences:
            if not isinstance(sentence, Sentence): continue
            text = self.llm._sentence_text(sentence)
            if not text: continue
            if self.llm._should_mark_sentence(sentence):
                source_id = str(source_index)
                source_index += 1
                source_sentences[source_id] = text
                parts.append(f'<E id="{source_id}">{text}</E>')
            else:
                parts.append(text)
        return " ".join(parts).strip()

    async def __call__(self, paper_content: Paper):
        paragraph_jobs = []
        for paragraph_id, paragraph in enumerate(iter_paragraphs(paper_content, include_abstract=True, include_appendix=True)):
            if not paragraph_sentences(paragraph): continue
            if not self._paragraph_has_source_sentence(paragraph): continue
            citation_map = self._paragraph_citation_map(paragraph)
            paragraph_jobs.append((paragraph_id, paragraph, citation_map))

        tasks = [
            asyncio.create_task(self.llm.call(inputs={
                "paragraph": paragraph,
                "citation_marker_map": citation_map,
                "paragraph_id": paragraph_id,
            }))
            for paragraph_id, paragraph, citation_map in paragraph_jobs
        ]

        claims, unverifiable_claims, errors = [], [], 0
        logging.info("Extract AF")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (paragraph_id, paragraph, citation_map), result in zip(paragraph_jobs, results):
            if isinstance(result, Exception):
                print(f"Claim {result} {type(result)}")
                errors += 1
                continue
            if not isinstance(result, list):
                errors += 1
                continue

            paragraph_citation_keys = list(dict.fromkeys(citation_map.values()))
            paragraph_entities = self._paragraph_entities(paragraph)
            for claim in result:
                claim["_paragraph_id"] = paragraph_id
                claim["_paragraph_citation_keys"] = paragraph_citation_keys
                claim["_paragraph_entities"] = paragraph_entities
                if claim.get("verifiable", False):
                    claims.append(claim)
                else:
                    unverifiable_claims.append(claim)

        logging.info(f"We have {len(claims)} verifiable claims and {len(unverifiable_claims)} unverifiable claims.")
        return {"claims": claims, "unverifiable_claims": unverifiable_claims, "errors": errors}
