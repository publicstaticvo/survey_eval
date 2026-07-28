from __future__ import annotations

from typing import Iterable, Iterator, List

from .paper_elements import Paper, Paragraph, Section, Sentence


def paragraph_sentences(paragraph: Paragraph | list[Sentence]) -> list[Sentence]:
    if isinstance(paragraph, Paragraph):
        return paragraph.sentences
    return paragraph


def iter_sections(content: Section, include_appendix: bool = False) -> Iterator[Section]:
    for section in content.children:
        yield section
        yield from iter_sections(section, include_appendix=False)
    if include_appendix and isinstance(content, Paper):
        for section in [*content.limitation, *content.appendix]:
            yield section
            yield from iter_sections(section, include_appendix=False)




def iter_sections_with_context(
    content: Paper,
    groups: tuple[str, ...] = ("sections", "limitation", "appendix"),
) -> Iterator[tuple[Section, list[str], str]]:
    def walk(section: Section, title_path: list[str], section_id: str):
        current_path = [*title_path, section.name]
        yield section, current_path, section_id
        for index, child in enumerate(section.children):
            child_id = f"{section_id}.{index + 1}" if section_id else str(index + 1)
            yield from walk(child, current_path, child_id)

    if "sections" in groups:
        for index, section in enumerate(content.children):
            yield from walk(section, [], str(index + 1))
    if "limitation" in groups:
        for index, section in enumerate(content.limitation):
            section_id = "Limitation" if len(content.limitation) == 1 else f"Limitation.{index + 1}"
            yield from walk(section, [], section_id)
    if "appendix" in groups:
        for index, section in enumerate(content.appendix):
            section_id = "Appendix" if len(content.appendix) == 1 else f"Appendix.{index + 1}"
            yield from walk(section, [], section_id)


def iter_paragraphs(
    content: Section,
    include_abstract: bool = False,
    include_appendix: bool = False,
) -> Iterator[Paragraph]:
    if include_abstract and isinstance(content, Paper) and content.abstract:
        yield from iter_paragraphs(content.abstract)
    yield from content.paragraphs
    for section in content.children:
        yield from iter_paragraphs(section)
    if include_appendix and isinstance(content, Paper):
        for section in [*content.limitation, *content.appendix]:
            yield from iter_paragraphs(section)


def split_content_to_paragraph(
    content: Section | list[Sentence],
    include_abstract: bool = False,
    include_appendix: bool = False,
) -> list[list[Sentence]]:
    if isinstance(content, list):
        return [content]
    return [
        paragraph.sentences
        for paragraph in iter_paragraphs(
            content,
            include_abstract=include_abstract,
            include_appendix=include_appendix,
        )
    ]


def iter_sentences(
    content: Section,
    include_abstract: bool = False,
    include_appendix: bool = False,
) -> Iterator[Sentence]:
    for paragraph in iter_paragraphs(
        content,
        include_abstract=include_abstract,
        include_appendix=include_appendix,
    ):
        yield from paragraph.sentences

def paragraph_to_text(content: Paragraph | list[Sentence], include_environments: bool = False) -> str:
    parts = []
    for sentence in paragraph_sentences(content):
        if not sentence.text:
            continue
        text = sentence.text.strip()
        if sentence.environment_type == "paragraph_name":
            parts.append(r"\paragraph{" + text + "}")
        elif sentence.environment_type == "text":
            parts.append(text)
        elif include_environments:
            parts.append(f"\\begin{{{sentence.environment_type}}} {text} \\end{{{sentence.environment_type}}}")
    return " ".join(parts).strip()


def paragraphs_to_text(paragraphs: Iterable[Paragraph | list[Sentence]], include_environments: bool = False) -> str:
    return "\n\n".join(filter(None, (paragraph_to_text(paragraph, include_environments) for paragraph in paragraphs)))


def section_to_text(section: Section | None, include_environments: bool = False) -> str:
    if section is None:
        return ""
    blocks = [paragraphs_to_text(section.paragraphs, include_environments)]
    for child in section.children:
        child_text = section_to_text(child, include_environments)
        if child_text:
            blocks.append(child_text)
    return "\n\n".join(filter(None, blocks))


def iter_heading(content: Section, include_appendix: bool = False) -> list[str]:
    headings = [content.name] if content.name and not isinstance(content, Paper) else []
    for section in content.children:
        headings.extend(iter_heading(section, include_appendix=False))
    if include_appendix and isinstance(content, Paper):
        if content.limitation:
            headings.append("Scope")
        for section in content.limitation:
            headings.extend(iter_heading(section, include_appendix=False))
        if content.appendix:
            headings.append("Appendix")
        for section in content.appendix:
            headings.extend(iter_heading(section, include_appendix=False))
    return headings


def get_top_level_section_titles(content: Section) -> List[str]:
    return [section.name for section in content.children if section.name]
