import re
import unidecode
import Levenshtein


def normalize_text(text: str) -> str:
    text = unidecode.unidecode(text or "")
    text = re.sub(r"[^0-9a-zA-Z]", "", text)
    return text.lower()


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
