from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "downloaded"


def clean(text: str, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def citation_count(citations) -> int:
    if isinstance(citations, dict):
        return len(citations)
    if isinstance(citations, list):
        return len(citations)
    return 0


def iter_sentences(section, loc=""):
    label = f"{loc} > {section.get('section_id')} {section.get('title')}".strip()
    for paragraph in section.get("paragraphs", []) or []:
        for sent in paragraph:
            if isinstance(sent, dict):
                yield label, sent
    for child in section.get("sections", []) or []:
        yield from iter_sentences(child, label)


def iter_subsections(section, main_index, total_main):
    for child in section.get("sections", []) or []:
        text_len = sum(len(sent.get("text", "")) for _, sent in iter_sentences(child))
        if 2 < main_index <= total_main - 2 and text_len > 300:
            yield text_len, child.get("section_id"), child.get("title")
        yield from iter_subsections(child, main_index, total_main)


def main() -> None:
    manifest = json.loads((ROOT / "selected_manifest.json").read_text(encoding="utf-8-sig"))
    for i, paper_item in enumerate(manifest, start=1):
        data = json.loads(Path(paper_item["json"]).read_text(encoding="utf-8-sig"))
        sections = data["full_text"].get("sections", []) or []
        total = len(sections)
        print(f"\n{i}. {data['title']} ({data['publication_date']})")
        print("query:", data["query"])
        print("main:", " | ".join(f"{s.get('section_id')} {s.get('title')}" for s in sections))
        intro_hits = []
        for loc, sent in iter_sentences(sections[0] if sections else {}):
            text = sent.get("text", "")
            if re.search(r"\b(contribution|scope|survey|paper|Section|we|our|review|summari[sz]e|introduce|focus)\b", text, re.I):
                intro_hits.append(clean(text))
        print("intro:")
        for hit in intro_hits[:3]:
            print("  -", hit)
        cited = []
        for idx, sec in enumerate(sections, start=1):
            if idx <= 2 or idx > total - 2:
                continue
            for loc, sent in iter_sentences(sec):
                cc = citation_count(sent.get("citations"))
                text = sent.get("text", "")
                if cc and "\\begin{table" not in text and "\\begin{figure" not in text and len(text) > 50:
                    cited.append((cc, len(text), loc, sent))
        cited.sort(key=lambda x: (x[0], x[1]), reverse=True)
        print("cited:")
        for cc, _, loc, sent in cited[:3]:
            print("  -", loc)
            print("    citations:", sent.get("citations"))
            print("    text:", clean(sent.get("text", "")))
        subs = []
        for idx, sec in enumerate(sections, start=1):
            subs.extend(iter_subsections(sec, idx, total))
        subs.sort(reverse=True)
        print("delete:")
        for text_len, sid, title in subs[:3]:
            print(f"  - {sid} {title} ({text_len})")


if __name__ == "__main__":
    main()
