import json
import re
from typing import Any, Dict

import Levenshtein
import numpy as np
import unidecode


def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text or "")
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()



ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "toward", "towards", "using", "via", "with",
}

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


def extract_json(text: str) -> Dict:
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


def cosine_similarity_matrix(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def cosine_similarity_pair(left, right) -> float:
    matrix = cosine_similarity_matrix(np.asarray([left]), np.asarray([right]))
    return float(matrix[0, 0])


def normalize_heading(title: str) -> str:
    value = (title or "").strip()
    value = re.sub(r"^\d+(?:\.\d+)*[.)-]?\s+", "", value)
    value = re.sub(r"^(?:[ivxlcdm]+)[.)-]\s+", "", value, flags=re.IGNORECASE)
    value = value.replace("&", " and ").replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def paper_ids(paper: dict[str, Any]) -> set[str]:
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


def paper_title(paper: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()


def is_same_paper(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ids = paper_ids(left)
    right_ids = paper_ids(right)
    if left_ids and right_ids and left_ids & right_ids:
        return True
    left_title = paper_title(left)
    return bool(left_title and left_title == paper_title(right))


def literature_pool_papers(literature_pool: Any) -> list[dict[str, Any]]:
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


def abstract_sentences(abstract: str) -> list[str]:
    text = re.sub(r"\s+", " ", abstract or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_proposed_entity(name: str) -> str:
    name = re.sub(r"^[\s:;,()\[\]{}]+|[\s:;,()\[\]{}.]+$", "", name or "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def candidate_names_from_phrase(phrase: str) -> list[str]:
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
    names.extend(re.findall(r"['\"]([^'\"]{2,80})['\"]", phrase))
    acronym = re.search(r"\b([A-Z][A-Za-z0-9-]*(?:-[A-Z0-9][A-Za-z0-9]*)*)\b", phrase)
    if acronym:
        names.append(acronym.group(1))
    title_like = re.search(r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){1,7})\b", phrase)
    if title_like:
        names.append(title_like.group(1))
    return list(dict.fromkeys(clean_proposed_entity(name) for name in names if clean_proposed_entity(name)))


def extract_literature_pool_proposed_entities(
    literature_pool: Any,
    cited_papers: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
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
    for paper in literature_pool_papers(literature_pool):
        if any(is_same_paper(paper, cited) for cited in cited_papers):
            continue
        for sentence in abstract_sentences(paper.get("abstract") or ""):
            names = []
            for pattern in (called, trigger):
                match = pattern.search(sentence)
                if match:
                    names.extend(candidate_names_from_phrase(match.group("name")))
            for name in names:
                proposed.setdefault(name, paper)
    return proposed