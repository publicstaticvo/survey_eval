import json
import random
import re
import time
import Levenshtein
import numpy as np
import requests
import unidecode
from typing import Any, Iterable, Iterator, List, Dict


def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text or "")
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()


def request_template(method: str, url: str, headers: dict, parameters: dict, timeout: int = 60, sleep: bool = True) -> dict:
    if sleep:
        time.sleep(2 * random.random())
    request_headers = dict(headers or {})
    if method == "post":
        request_headers["Content-type"] = "application/json"
        response = requests.post(url, headers=request_headers, json=parameters, timeout=timeout)
    else:
        response = requests.get(url, headers=request_headers, params=parameters, timeout=timeout)
    response.raise_for_status()
    return response.json()


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
    

def extract_list(text: str) -> list:
    """从文本中提取 JSON 列表"""
    if not text:
        return []
    
    try:
        return json.loads(text)
    except Exception:
        pass
    
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    
    try:
        candidate = text[start : end + 1].replace("'", '"')
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)
    except Exception:
        return []
    

def clean_token(text: str) -> str:
    text = (text or "").replace("\\", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    punct = r"\":.,!?&'()[]{}"
    head_tail_pat = re.compile(f"^[{re.escape(punct)}]+|[{re.escape(punct)}]+$")
    inner_pat = re.compile(f"[{re.escape(punct)}]")
    cleaned_tokens = []
    for token in text.split():
        token = head_tail_pat.sub("", token)
        if inner_pat.search(token):
            continue
        if token:
            cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def index_to_abstract(indexes: dict | None):
    if not indexes:
        return None
    abstract_length = max(v[-1] for v in indexes.values())
    abstract = ["<mask>" for _ in range(abstract_length + 1)]
    for token, positions in indexes.items():
        for i in positions:
            abstract[i] = token
    return " ".join(abstract)


def valid_check(query: str, target: str, ratio: float = 0.1) -> bool:
    if not target:
        return False
    query = normalize_text(query)
    target = normalize_text(target)
    if not query or not target:
        return False
    if query in target or target in query:
        return True
    distance = Levenshtein.distance(query, target)
    return distance <= max(1, int(ratio * len(query)))


def split_content_to_paragraph(content: dict | list):
    if isinstance(content, list):
        return list(content)
    paragraphs = list(content.get("paragraphs", []))
    for section in content.get("sections", []):
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


def paragraph_to_text(content: list[dict]):
    return " ".join(s.get("text", "") for s in content if s.get("text")).strip()


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


def paragraphs_to_text(paragraphs: Iterable[list[dict]]) -> str:
    return "\n\n".join(filter(None, (paragraph_to_text(p) for p in paragraphs)))


def get_section_titles(content: dict) -> List[str]:
    return [section.get("title", "") for section in iter_sections(content) if section.get("title")]


def get_top_level_section_titles(content: dict) -> List[str]:
    return [section.get("title", "") for section in content.get("sections", []) if section.get("title")]


def flatten_sections(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    flattened = []

    def _walk(section: Dict[str, Any], parent_titles: List[str]):
        title = section.get("title", "").strip()
        title_path = [*parent_titles, title] if title else list(parent_titles)
        flattened.append(
            {
                "section_id": section.get("section_id"),
                "title": title,
                "title_path": " > ".join(x for x in title_path if x),
                "depth": len(title_path),
            }
        )
        for child in section.get("sections", []): _walk(child, title_path)

    for section in paper.get("sections", []): _walk(section, [])
    return flattened


def get_first_section(content: dict) -> dict | None:
    sections = content.get("sections", [])
    return sections[0] if sections else None


def get_last_sections(content: dict, n: int) -> List[dict]:
    sections = content.get("sections", [])
    return sections[-n:] if sections else []


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return paragraphs_to_text(value)
    if isinstance(value, dict):
        if "text" in value:
            return safe_text(value["text"])
        return section_to_text(value)
    return str(value).strip()


def make_excerpt(text: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


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


def cosine_similarity_pair(left, right) -> float:
    matrix = cosine_similarity_matrix(np.asarray([left]), np.asarray([right]))
    return float(matrix[0, 0])
