from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "downloaded"


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:420]


def citation_count(citations) -> int:
    if isinstance(citations, dict):
        return len(citations)
    if isinstance(citations, list):
        return len(citations)
    return 0


def iter_sentences(section, path):
    title = section.get("title", "")
    sid = section.get("section_id", "")
    here = f"{path} > {sid} {title}".strip()
    for paragraph in section.get("paragraphs", []) or []:
        for sent in paragraph:
            if isinstance(sent, dict):
                yield here, sent
    for child in section.get("sections", []) or []:
        yield from iter_sentences(child, here)


def iter_subsections(section, main_index, total_main):
    for child in section.get("sections", []) or []:
        title = child.get("title", "")
        sid = child.get("section_id", "")
        text_len = sum(len(sent.get("text", "")) for _, sent in iter_sentences(child, ""))
        if 2 < main_index <= total_main - 2 and text_len > 300:
            yield {"section_id": sid, "title": title, "text_len": text_len}
        yield from iter_subsections(child, main_index, total_main)


def sentence_score(item):
    path, sent = item
    text = sent.get("text", "")
    return citation_count(sent.get("citations")) * 1000 + min(len(text), 500)


def main() -> None:
    manifest = json.loads((ROOT / "selected_manifest.json").read_text(encoding="utf-8-sig"))
    for paper_item in manifest:
        path = Path(paper_item["json"])
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        full = data["full_text"]
        sections = full.get("sections", []) or []
        total = len(sections)
        print("\n" + "=" * 100)
        print(f"TITLE: {data['title']}")
        print(f"DATE: {data['publication_date']} | QUERY: {data['query']}")
        print("SECTIONS:", " | ".join(f"{s.get('section_id')} {s.get('title')}" for s in sections))

        intro = sections[0] if sections else {}
        intro_candidates = []
        for loc, sent in iter_sentences(intro, ""):
            text = sent.get("text", "")
            if re.search(r"\b(contribution|paper|survey|section|we|our|organize|review|summari[sz]e|introduce)\b", text, re.I):
                intro_candidates.append(clean(text))
        print("INTRO_CANDIDATES:")
        for text in intro_candidates[:8]:
            print(f"- {text}")

        cit_candidates = []
        for idx, section in enumerate(sections, start=1):
            if idx <= 2 or idx > total - 2:
                continue
            for loc, sent in iter_sentences(section, ""):
                if citation_count(sent.get("citations")) > 0:
                    cit_candidates.append((loc, sent))
        cit_candidates.sort(key=sentence_score, reverse=True)
        print("CITED_MIDDLE_SENTENCES:")
        for loc, sent in cit_candidates[:8]:
            print(f"- LOC: {loc}")
            print(f"  CIT: {sent.get('citations')}")
            print(f"  TXT: {clean(sent.get('text', ''))}")

        sub_candidates = []
        for idx, section in enumerate(sections, start=1):
            sub_candidates.extend(iter_subsections(section, idx, total))
        sub_candidates.sort(key=lambda x: x["text_len"], reverse=True)
        print("SUBSECTION_DELETE_CANDIDATES:")
        for sub in sub_candidates[:8]:
            print(f"- {sub['section_id']} {sub['title']} ({sub['text_len']} chars)")


if __name__ == "__main__":
    main()
