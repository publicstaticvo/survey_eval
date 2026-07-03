from __future__ import annotations

from typing import Any


def citation_items(citations: Any) -> list[dict[str, Any]]:
    """Return normalized citation marker/key pairs from old and new sentence formats."""
    items: list[dict[str, Any]] = []
    if isinstance(citations, dict):
        iterable = citations.items()
        for marker, key in iterable:
            if key:
                try:
                    marker_value: int | str = int(marker)
                except (TypeError, ValueError):
                    marker_value = str(marker)
                items.append({"marker": marker_value, "key": str(key)})
        return items

    for citation in citations or []:
        if isinstance(citation, dict) and not any(key in citation for key in ("key", "ref_text", "marker", "number")):
            items.extend(citation_items(citation))
            continue
        marker = None
        key = citation
        if isinstance(citation, dict):
            marker = citation.get("marker") or citation.get("number")
            key = citation.get("key") or citation.get("ref_text")
        if key:
            item = {"key": str(key)}
            if marker is not None:
                try:
                    item["marker"] = int(marker)
                except (TypeError, ValueError):
                    item["marker"] = str(marker)
            items.append(item)
    return items


def citation_keys(citations: Any) -> list[str]:
    return list(dict.fromkeys(item["key"] for item in citation_items(citations) if item.get("key")))


def has_citations(citations: Any) -> bool:
    return bool(citation_keys(citations))


def citation_marker_key_map(citations: Any) -> dict[Any, str]:
    return {item.get("marker", item["key"]): item["key"] for item in citation_items(citations) if item.get("key")}
