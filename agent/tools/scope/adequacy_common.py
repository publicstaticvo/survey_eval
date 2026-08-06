from __future__ import annotations

from typing import Any, Iterable

from ..utility.paper_elements import Paper, Section


def iter_sections(paper: Paper) -> Iterable[Section]:
    def walk(section: Section):
        yield section
        for child in section.children:
            yield from walk(child)
    for section in paper.children:
        yield from walk(section)
    for section in paper.limitation:
        yield from walk(section)
    for section in paper.appendix:
        yield from walk(section)


def iter_sentences(paper: Paper) -> Iterable[dict[str, Any]]:
    for section in iter_sections(paper):
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                text = sentence.caption or sentence.text or ""
                if text.strip():
                    yield {
                        "text": text.strip(),
                        "label": sentence.label,
                        "section": section.name,
                        "citations": list(sentence.citations or []),
                    }


def content_topics(paper: Paper) -> list[str]:
    topics: list[str] = []
    for section in iter_sections(paper):
        parsed = section.parsed_contents or {}
        if not isinstance(parsed, dict):
            continue
        for topic in parsed.get("topics", []) or []:
            label = str(topic).strip()
            if label and label not in topics:
                topics.append(label)
    return topics


def sentence_entities(sentence: dict[str, Any]) -> set[str]:
    text = sentence.get("text", "")
    # Cheap entity proxy for numeric metrics; LLM-facing detectors may replace this later.
    import re
    spans = re.findall(r"\b(?:[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,4}|[A-Z]{2,})\b", text)
    return {span.strip() for span in spans if len(span.strip()) > 1}


def top_comments(items: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    return items[: max(0, k)]
