import re
import json
import time
import random
import requests
import unidecode
import Levenshtein


def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text)
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()


def request_template(method: str, url: str, headers: dict, parameters: dict, timeout: int = 60, sleep: bool = True) -> dict:
    if sleep: time.sleep(2 * random.random())
    if method == "post": 
        headers['Content-type'] = "application/json"
        request_method = requests.post
    else:
        request_method = requests.get
    response = request_method(url, headers=headers, params=parameters, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data


def extract_json(text: str) -> dict:
    """从文本中提取 JSON 对象"""
    if not text:
        return {}
    
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return json.loads(re.findall(r"```(?:json)?\s*(\{.*?\})\s```", text, re.DOTALL)[-1])
    except Exception:
        pass
    
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    
    candidate = text[start:end+1].replace("'", '"')
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)
    

def extract_list(text: str) -> dict:
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
        candidate = text[start:end+1].replace("'", '"')
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)
    except Exception:
        return []
    

def clean_token(text: str) -> str:
    text = text.replace("\\", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    punct = r"\":.,!?&'()[]{}"
    head_tail_pat = re.compile(f'^[{re.escape(punct)}]+|[{re.escape(punct)}]+$')
    inner_pat = re.compile(f'[{re.escape(punct)}]')
    cleaned_tokens = []
    for t in text.split():
        t = head_tail_pat.sub('', t)
        if inner_pat.search(t): continue
        if t: cleaned_tokens.append(t)
    return ' '.join(cleaned_tokens)


def index_to_abstract(indexes: dict | None):
    if not indexes: return None
    abstract_length = max(v[-1] for v in indexes.values())
    abstract = ["<mask>" for _ in range(abstract_length + 1)]
    for k, v in indexes.items():
        for i in v:
            abstract[i] = k
    return " ".join(abstract)


def valid_check(query: str, target: str, ratio: float = 0.1) -> bool:
    if not target: return False
    query = normalize_text(query)
    target = normalize_text(target)
    if query in target: return True
    distance = Levenshtein.distance(query, target)
    return distance <= ratio * len(query)


def split_content_to_paragraph(content: dict | list):
    if isinstance(content, list): return content
    paragraphs = content['paragraphs']
    for section in content['sections']:
        paragraphs.extend(split_content_to_paragraph(section))
    return paragraphs


def paragraph_to_text(content: list[dict]):
    return " ".join([s['text'] for s in content])
