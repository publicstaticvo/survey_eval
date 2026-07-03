import json
import re
import Levenshtein
import numpy as np
import unidecode
from typing import Dict, List, Any


def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text or "")
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()


def extract_json(text: str) -> Dict:
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


def paragraph_sentences(paragraph):
    if isinstance(paragraph, dict):
        return paragraph.get("sentences", []) or []
    return paragraph


def split_content_to_paragraph(content: Dict | List):
    if isinstance(content, list): return list(content)
    paragraphs = [
        paragraph_sentences(paragraph)
        for paragraph in content.get("paragraphs", [])
    ]
    for section in content.get("sections", []):
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


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


def get_top_level_section_titles(content: dict) -> List[str]:
    return [section.get("title", "") for section in content.get("sections", []) if section.get("title")]


def paragraph_to_text(content, include_environments: bool):
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
        elif include_environments:
            parts.append(f"\\begin{{{environment_type}}} {text} \\end{{{environment_type}}}")
    return " ".join(parts).strip()


def paragraphs_to_text(paragraphs, ie = False) -> str:
    return "\n\n".join(filter(None, (paragraph_to_text(p, ie) for p in paragraphs)))


def section_to_text(section: dict, ie: bool = False) -> str:
    blocks = [paragraph_to_text(paragraph, ie) for paragraph in section.get("paragraphs", [])]
    for child in section.get("sections", []):
        child_text = section_to_text(child)
        if child_text:
            blocks.append(child_text)
    return "\n\n".join(filter(None, blocks))

