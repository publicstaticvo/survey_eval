from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


GRAPH_ENVIRONMENT_NAMES = {"tikzpicture", "figure", "figure*", "table", "table*", "tabular", "longtable"}


@dataclass
class Sentence:
    """Represents text, a LaTeX environment block, or another sentence-like item."""
    text: str = ""
    father: "Paragraph" = None
    citations: Union[List[str], Dict[int, str]] = field(default_factory=list)
    environment_type: str = "text"
    caption: str = ""
    label: str = ""
    confidence: Optional[float] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self):
        citations = self.citation_keys()
        cite_str = f" [{', '.join(citations)}]" if citations else ""
        return f"{self.text} {cite_str}"

    def __repr__(self):
        return self.text

    def citation_keys(self) -> List[str]:
        if isinstance(self.citations, dict):
            return list(self.citations.values())
        return list(self.citations or [])

    def to_dict(self) -> Dict[str, Any]:
        return self.get_skeleton()

    def get_skeleton(self) -> Dict[str, Any]:
        result = {
            "text": re.sub(r"\s+", " ", self.text or "").strip(),
            "citations": self.citations,
            "environment_type": self.environment_type or "text",
        }
        if self.caption:
            result["caption"] = self.caption
        if self.label:
            result["label"] = self.label
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.claims:
            result["claims"] = self.claims
        return result
    
    @classmethod
    def from_skeleton(cls, data: dict[str, Any]):
        if data.get("environment_type") == "paragraph_name":
            sentence = ParagraphName(text=data.get("text", ""), citations=data.get("citations", {}))
        else:
            sentence = cls(
                text=data.get("text", ""),
                citations=data.get("citations", {}),
                environment_type=data.get("environment_type", "text"),
                caption=data.get("caption", ""),
            )
        sentence.label = data.get("label", "")
        sentence.confidence = data.get("confidence")
        sentence.claims = data.get("claims", []) or []
        return sentence


class ParagraphName(Sentence):
    r"""Represents a \paragraph{...} heading as a sentence-like item."""

    def __init__(self, text: str, citations: Union[List[str], Dict[int, str]] | None = None):
        super().__init__(text=text, citations=citations or {}, environment_type="paragraph_name")

    def __repr__(self):
        return f"\\paragraph{{{self.text}}}"


@dataclass
class Paragraph:
    """Represents a paragraph with multiple sentence-like items."""
    father: "Section" = None
    sentences: List[Sentence] = field(default_factory=list)
    name: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    alias_pairs: List[List[str]] = field(default_factory=list)

    def add_sentence(self, sentence: Sentence):
        sentence.father = self
        self.sentences.append(sentence)

    def get_skeleton(self) -> List[Dict[str, Any]]:
        return [sentence.get_skeleton() for sentence in self.sentences]

    def has_text_content(self) -> bool:
        return bool(self.sentences)

    def to_dict(self) -> Dict[str, Any]:
        result = {"sentences": [sentence.to_dict() for sentence in self.sentences]}
        if self.name:
            result["name"] = self.name
        if self.entities:
            result["entities"] = self.entities
        if self.alias_pairs:
            result["alias_pairs"] = self.alias_pairs
        return result

    def get_sentences(self) -> List[Sentence]:
        return self.sentences

    def get_next_sentence_until_citation(self, current_idx: int, sentence_number: int):
        graph_environments = {"tikzpicture", "figure", "table", "tabular", "longtable"}
        current_sentence = self.sentences[current_idx]
        if len(current_sentence.citations) >= 2:
            return []
        next_sentences = []
        i = current_idx + 1
        while i <= current_idx + sentence_number + 1:
            if i >= len(self.sentences):
                return next_sentences
            sentence = self.sentences[i]
            if sentence.environment_type == "text" and sentence.citations:
                return next_sentences
            if sentence.environment_type in graph_environments:
                sentence_number += 1
            else:
                next_sentences.append(sentence.text)
            i += 1
        return next_sentences
    
    @classmethod
    def from_skeleton(cls, data: list[dict[str, Any]] | dict[str, Any]):
        if isinstance(data, dict):
            sentence_data = data.get("sentences", []) or []
            paragraph = cls(name=data.get("name"))
            paragraph.entities = data.get("entities", []) or []
            paragraph.alias_pairs = data.get("alias_pairs", []) or []
        else:
            sentence_data = data or []
            paragraph = cls()
        for item in sentence_data:
            if isinstance(item, dict):
                paragraph.add_sentence(Sentence.from_skeleton(item))
        return paragraph

    def __repr__(self):
        return debug_sentences(self.sentences)


@dataclass
class Section:
    """Represents a section, subsection, subsubsection, or abstract."""
    name: str = ""
    father: Union["Section", "Paper"] = None
    paragraphs: List[Paragraph] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)
    functional_type: str = ""
    content_tags: List[str] = field(default_factory=list)
    parsed_contents: Dict[str, Any] = field(default_factory=dict)

    def add_paragraph(self, paragraph: Paragraph):
        paragraph.father = self
        self.paragraphs.append(paragraph)

    def add_child(self, section: "Section"):
        section.father = self
        self.children.append(section)

    def get_sentences(self) -> List[Sentence]:
        sentences = []
        for paragraph in self.paragraphs:
            sentences.extend(paragraph.get_sentences())
        for child in self.children:
            sentences.extend(child.get_sentences())
        return sentences

    def get_skeleton(self, section_id) -> Dict[str, Any]:
        result = {
            "title": self.name,
            "section_id": section_id,
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs if paragraph.has_text_content()],
            "sections": [
                section.get_skeleton(f"{section_id}.{index + 1}" if section_id else str(index + 1))
                for index, section in enumerate(self.children)
            ],
        }
        if self.functional_type:
            result["functional_type"] = self.functional_type
        if self.content_tags:
            result["content_tags"] = self.content_tags
        if self.parsed_contents:
            result["parsed_contents"] = self.parsed_contents
        return result
    
    @classmethod
    def from_skeleton(cls, data: dict[str, Any]):
        section = cls(name=data.get("title") or data.get("name", ""))
        section.functional_type = data.get("functional_type", "")
        section.content_tags = data.get("content_tags", []) or []
        section.parsed_contents = data.get("parsed_contents", {}) or {}
        for paragraph_data in data.get("paragraphs", []) or []:
            section.add_paragraph(Paragraph.from_skeleton(paragraph_data))
        for child_data in data.get("sections", []) or []:
            if isinstance(child_data, dict):
                section.add_child(Section.from_skeleton(child_data))
        return section

    def __repr__(self):
        return f"Section('{self.name}', {len(self.children) + len(self.paragraphs)} children)"


@dataclass
class Paper(Section):
    """Represents the entire academic paper."""
    title: str = ""
    author: Optional[str] = None
    abstract: Optional[Section] = None
    keywords: List[str] = field(default_factory=list)
    limitation: List[Section] = field(default_factory=list)
    appendix: List[Section] = field(default_factory=list)
    references: dict = field(default_factory=dict)
    all_citation_keys: List[str] = field(default_factory=list)
    unresolved_citation_keys: List[str] = field(default_factory=list)
    has_section_index: bool = True
    contribution_claims: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: Section):
        self.add_child(section)

    def add_keyword(self, word: str):
        self.keywords.append(word)

    def get_sentences(self) -> List[Sentence]:
        sentences = []
        if self.abstract is not None:
            sentences.extend(self.abstract.get_sentences())
        for section in [*self.children, *self.limitation, *self.appendix]:
            sentences.extend(section.get_sentences())
        return sentences

    def get_skeleton(self) -> dict:
        result = {
            "title": self.title or self.name,
            "author": self.author,
            "abstract": self.abstract.get_skeleton("") if self.abstract else "",
            "paragraphs": [paragraph.get_skeleton() for paragraph in self.paragraphs],
            "sections": [section.get_skeleton(index + 1) for index, section in enumerate(self.children)],
            "citations": self.references,
        }
        if self.limitation:
            result["limitation"] = [section.get_skeleton(index + 1) for index, section in enumerate(self.limitation)]
        if self.appendix:
            result["appendix"] = [section.get_skeleton(index + 1) for index, section in enumerate(self.appendix)]
        if self.unresolved_citation_keys:
            result["missing_citations"] = self.unresolved_citation_keys
            result["missing_citation_count"] = len(self.unresolved_citation_keys)
        if self.contribution_claims:
            result["contribution_claims"] = self.contribution_claims
        return result

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.title:
            result["title"] = self.title
        if self.author:
            result["author"] = self.author
        if self.abstract:
            result["abstract"] = self.abstract.to_dict()
        result["sections"] = [section.to_dict() for section in self.children]
        if self.all_citation_keys:
            result["citations"] = self.all_citation_keys
        return result
    
    @classmethod
    def from_skeleton(cls, data: dict[str, Any]):
        paper = cls(title=data.get("title", ""), author=data.get("author"))
        abstract = data.get("abstract")
        if isinstance(abstract, dict):
            paper.abstract = Section.from_skeleton(abstract)
        for paragraph_data in data.get("paragraphs", []) or []:
            paper.add_paragraph(Paragraph.from_skeleton(paragraph_data))
        for section_data in data.get("sections", []) or []:
            if isinstance(section_data, dict):
                paper.add_section(Section.from_skeleton(section_data))
        for section_data in data.get("limitation", []) or []:
            if isinstance(section_data, dict):
                paper.limitation.append(Section.from_skeleton(section_data))
        for section_data in data.get("appendix", []) or []:
            if isinstance(section_data, dict):
                paper.appendix.append(Section.from_skeleton(section_data))
        paper.references = data.get("citations", {}) or data.get("references", {}) or {}
        paper.unresolved_citation_keys = data.get("missing_citations", []) or []
        paper.all_citation_keys = list(paper.references.keys())
        paper.contribution_claims = data.get("contribution_claims", {}) or {}
        return paper

    def __str__(self):
        return json.dumps(self.get_skeleton(), indent=2, ensure_ascii=False)


def debug_sentences(sentences: List[Sentence], start_id: int = 0):
    output = ""
    for index, sentence in enumerate(sentences):
        if isinstance(sentence, ParagraphName):
            item = f"{index + start_id}\t{sentence.__repr__()}\n"
        else:
            sentence_dict = sentence.to_dict()
            item = f"{index + start_id}\t{sentence_dict['text']}\n"
            if sentence_dict["citations"]:
                item += f"-\tCitations: {sentence_dict['citations']}\n"
        output += item
    return output


def get_paragraph_skeleton(paragraph: Paragraph, mode: str, accumulated_sentence_id: int = 0):
    if mode == "none" or len(paragraph.sentences) == 0:
        return ""
    if mode == "first":
        return paragraph.sentences[0].text
    return debug_sentences(paragraph.sentences, accumulated_sentence_id)


__all__ = [
    "GRAPH_ENVIRONMENT_NAMES",
    "Sentence",
    "ParagraphName",
    "Paragraph",
    "Section",
    "Paper",
    "debug_sentences",
    "get_paragraph_skeleton",
]
