import json
import re
import numpy as np
from typing import Any, Iterable, Iterator, List, Dict


def extract_json(text: str) -> dict:
    """从文本中提取 JSON 对象"""
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
    

def split_content_to_paragraph(content: dict | list):
    if isinstance(content, list):
        return list(content)
    paragraphs = list(content.get("paragraphs", []))
    for section in content.get("sections", []):
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


def paragraph_to_text(content: list[dict]):
    return " ".join(s.get("text", "") for s in content if s.get("text")).strip()


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


def get_top_level_section_titles(content: dict) -> List[str]:
    return [section.get("title", "") for section in content.get("sections", []) if section.get("title")]


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
    left = left / left_norm
    right = right / right_norm
    return left @ right.T
