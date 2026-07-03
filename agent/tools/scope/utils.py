import json
import re
import numpy as np
from typing import Any, Iterable, Iterator, List, Dict


def extract_json(text: str) -> dict:
    """浠庢枃鏈腑鎻愬彇 JSON 瀵硅薄"""
    if not text:
        return {}
    
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced[-1])
        except Exception:
            pass
    
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    
    candidate = text[start : end + 1].replace("'", '"')
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)
    

def paragraph_sentences(paragraph):
    if isinstance(paragraph, dict):
        return paragraph.get("sentences", []) or []
    return paragraph


def split_content_to_paragraph(content: dict | list):
    if isinstance(content, list):
        return list(content)
    paragraphs = list(content.get("paragraphs", []))
    for section in content.get("sections", []):
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


def paragraph_to_text(content):
    parts = []
    for sentence in paragraph_sentences(content):
        if not isinstance(sentence, dict) or not sentence.get("text"):
            continue
        text = str(sentence.get("text") or "").strip()
        environment_type = sentence.get("environment_type", "text")
        if environment_type == "paragraph_name":
            parts.append(r"\paragraph{" + text + "}")
        elif environment_type == "text":
            parts.append(text)
        else:
            parts.append(f"\\begin{{{environment_type}}} {text} \\end{{{environment_type}}}")
    return " ".join(parts).strip()


def safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return paragraph_to_text(value)
    if isinstance(value, dict):
        if "text" in value:
            return str(value.get("text") or "").strip()
        return section_to_text(value)
    return ""


def paragraphs_to_text(paragraphs: Iterable[list[dict]]) -> str:
    return "\n\n".join(filter(None, (paragraph_to_text(p) for p in paragraphs)))


def section_text(section: dict[str, Any], include_children: bool = True) -> str:
    blocks = [paragraphs_to_text(section.get("paragraphs", []))]
    if include_children:
        for child in section.get("sections", []) or []:
            child_text = section_text(child, include_children=True)
            if child_text:
                blocks.append(child_text)
    return "\n\n".join(block for block in blocks if block)


def section_to_text(section: dict) -> str:
    blocks = [paragraph_to_text(paragraph) for paragraph in section.get("paragraphs", [])]
    for child in section.get("sections", []):
        child_text = section_to_text(child)
        if child_text:
            blocks.append(child_text)
    return "\n\n".join(filter(None, blocks))


def iter_sections(content: dict) -> Iterator[dict]:
    for section in content.get("sections", []):
        yield section
        yield from iter_sections(section)


def get_section_titles(content: dict) -> List[str]:
    return [section.get("title", "") for section in iter_sections(content) if section.get("title")]


def get_first_section(content: dict) -> Dict[str, Any] | None:
    sections = content.get("sections", []) if isinstance(content, dict) else []
    return sections[0] if sections else None


def cosine_similarity_matrix(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def normalize_heading(title: str) -> str:
    value = (title or "").strip()
    value = re.sub(r"^\d+(?:\.\d+)*[.)-]?\s+", "", value)
    value = re.sub(r"^(?:[ivxlcdm]+)[.)-]\s+", "", value, flags=re.IGNORECASE)
    value = value.replace("&", " and ").replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip().lower()



def _paper_ids(paper: dict[str, Any]) -> set[str]:
    ids = set()
    for key in ("id", "paperId", "corpusId"):
        if paper.get(key):
            ids.add(str(paper[key]).replace("https://openalex.org/", ""))
    raw_ids = paper.get("ids")
    if isinstance(raw_ids, dict):
        ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids.values() if value)
    elif isinstance(raw_ids, (list, tuple, set)):
        ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids if value)
    return {item for item in ids if item}


def _paper_title(paper: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()


def _is_same_paper(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ids = _paper_ids(left)
    right_ids = _paper_ids(right)
    if left_ids and right_ids and left_ids & right_ids:
        return True
    left_title = _paper_title(left)
    return bool(left_title and left_title == _paper_title(right))


def _literature_pool_papers(literature_pool: Any) -> list[dict[str, Any]]:
    if isinstance(literature_pool, dict):
        literature_pool = literature_pool.get("literature_pool", literature_pool)
    values = literature_pool.values() if isinstance(literature_pool, dict) else literature_pool or []
    papers = []
    for item in values:
        if isinstance(item, dict) and isinstance(item.get("paper"), dict):
            papers.append(item["paper"])
        elif isinstance(item, dict) and item.get("title"):
            papers.append(item)
    return papers


def _abstract_sentences(abstract: str) -> list[str]:
    text = re.sub(r"\s+", " ", abstract or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _clean_proposed_entity(name: str) -> str:
    name = re.sub(r"^[\s:;,()\[\]{}]+|[\s:;,()\[\]{}.]+$", "", name or "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _candidate_names_from_phrase(phrase: str) -> list[str]:
    phrase = re.split(
        r"\b(?:for|to|that|which|by|using|based on|with|via|in order to|capable of)\b",
        phrase,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    phrase = re.split(r"[,;:]", phrase, maxsplit=1)[0]
    names = []
    paren = re.search(r"(.+?)\s*\(([A-Za-z][A-Za-z0-9-]{1,20})\)", phrase)
    if paren:
        names.extend([paren.group(1), paren.group(2)])
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", phrase)
    names.extend(quoted)
    acronym = re.search(r"\b([A-Z][A-Za-z0-9-]*(?:-[A-Z0-9][A-Za-z0-9]*)*)\b", phrase)
    if acronym:
        names.append(acronym.group(1))
    title_like = re.search(r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,7})\b", phrase)
    if title_like:
        names.append(title_like.group(1))
    return list(dict.fromkeys(_clean_proposed_entity(name) for name in names if _clean_proposed_entity(name)))


def extract_literature_pool_proposed_entities(
    literature_pool: Any,
    cited_papers: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Programmatically extract precise named contributions from non-cited pool papers."""
    cited_papers = cited_papers or []
    proposed: dict[str, dict[str, Any]] = {}
    trigger = re.compile(
        r"\b(?:we|this paper|this work|our work|our paper)\s+"
        r"(?:propose|proposes|proposed|introduce|introduces|introduced|present|presents|presented)\s+"
        r"(?:a|an|the|our|novel|new)?\s*(?P<name>[^.]{2,180})",
        flags=re.IGNORECASE,
    )
    called = re.compile(
        r"\b(?:we|this paper|this work|our work|our paper)\s+"
        r"(?:propose|introduce|present)\s+[^.]{0,80}?\b(?:called|named|termed)\s+(?P<name>[^,.;]{2,120})",
        flags=re.IGNORECASE,
    )
    for paper in _literature_pool_papers(literature_pool):
        if any(_is_same_paper(paper, cited) for cited in cited_papers):
            continue
        abstract = paper.get("abstract") or ""
        for sentence in _abstract_sentences(abstract):
            names = []
            for pattern in (called, trigger):
                match = pattern.search(sentence)
                if match:
                    names.extend(_candidate_names_from_phrase(match.group("name")))
            for name in names:
                proposed.setdefault(name, paper)
    return proposed
