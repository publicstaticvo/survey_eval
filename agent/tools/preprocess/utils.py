import json
import re
import Levenshtein
import numpy as np
import unidecode

def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text or "")
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()


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
