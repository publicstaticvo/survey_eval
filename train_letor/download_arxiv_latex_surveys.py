from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from survey_eval.agent.tools.utility.latex_parser import LatexPaperParser


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_SRC = "https://arxiv.org/e-print/{arxiv_id}"
ATOM = "{http://www.w3.org/2005/Atom}"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_title(title: str) -> str:
    title = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", title or "")
    title = re.sub(r"[^\w\s]", " ", title.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", title).strip()


def slugify(title: str) -> str:
    slug = re.sub(r"\s+", "_", title.strip().lower())
    slug = re.sub(r"[^a-z0-9_\-]+", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:150] or "untitled"


def title_similarity(left: str, right: str) -> float:
    a = normalize_title(left)
    b = normalize_title(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def arxiv_query(title: str, pause: float) -> dict | None:
    params = urllib.parse.urlencode(
        {
            "search_query": f'ti:"{title}"',
            "start": 0,
            "max_results": 5,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "survey-eval-latex-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    time.sleep(pause)

    root = ET.fromstring(data)
    best = None
    best_score = 0.0
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        score = title_similarity(title, arxiv_title)
        if score > best_score:
            best_score = score
            arxiv_id_url = entry.findtext(f"{ATOM}id") or ""
            best = {
                "arxiv_title": arxiv_title,
                "arxiv_id": arxiv_id_url.rstrip("/").split("/")[-1],
                "published": entry.findtext(f"{ATOM}published") or "",
                "score": score,
            }
    if best and best["score"] >= 0.88:
        return best
    return None


def safe_extract_tar(buffer_path: Path, target_dir: Path) -> bool:
    try:
        with tarfile.open(buffer_path, mode="r:*") as archive:
            target_root = target_dir.resolve()
            for member in archive.getmembers():
                member_path = (target_root / member.name).resolve()
                if target_root != member_path and target_root not in member_path.parents:
                    raise RuntimeError(f"Unsafe tar member: {member.name}")
            archive.extractall(target_root)
        return True
    except tarfile.TarError:
        return False


def unpack_source(buffer_path: Path, target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    if safe_extract_tar(buffer_path, target_dir):
        return True

    raw = buffer_path.read_bytes()
    if raw.startswith(b"%PDF-"):
        return False
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return False
        if raw.startswith(b"%PDF-"):
            return False
    (target_dir / "source.tex").write_bytes(raw)
    return True


def download_source(arxiv_id: str, target_dir: Path) -> bool:
    url = ARXIV_SRC.format(arxiv_id=arxiv_id)
    buffer_path = target_dir.with_suffix(".download")
    req = urllib.request.Request(url, headers={"User-Agent": "survey-eval-latex-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        buffer_path.write_bytes(resp.read())
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return unpack_source(buffer_path, target_dir)
    finally:
        buffer_path.unlink(missing_ok=True)


def parse_to_json(source_dir: Path, output_path: Path, item: dict) -> None:
    parser = LatexPaperParser()
    main_tex = parser._find_main_tex(source_dir)
    if not main_tex:
        raise RuntimeError("No .tex file found")
    paper = LatexPaperParser().parse(main_tex)
    if paper is None:
        raise RuntimeError("LatexPaperParser returned no paper")
    data = {
        "title": item["title"],
        "publication_date": item["publication_date"],
        "query": item["query"],
        "arxiv_id": item["arxiv_id"],
        "arxiv_title": item["arxiv_title"],
        "latex_source_dir": str(source_dir),
        "full_text": paper.get_skeleton(),
    }
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_candidates(base_dir: Path) -> list[dict]:
    surge_rows = load_jsonl(base_dir / "surge.jsonl")
    query_by_title = {row["title"]: row.get("query", "") for row in load_jsonl(base_dir / "surveys_with_query.jsonl")}
    candidates = []
    for row in surge_rows:
        title = row["title"]
        if title not in query_by_title:
            continue
        candidates.append(
            {
                "title": title,
                "publication_date": row.get("publication_date", ""),
                "query": query_by_title[title],
            }
        )
    candidates.sort(key=lambda x: x["publication_date"], reverse=True)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pause", type=float, default=3.1)
    parser.add_argument("--max-check", type=int, default=120)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_root = base_dir / "downloaded"
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(base_dir)

    successes = []
    checked = 0
    for item in candidates:
        if len(successes) >= args.limit or checked >= args.max_check:
            break
        checked += 1
        title = item["title"]
        slug = slugify(title)
        source_dir = output_root / slug
        json_path = output_root / f"{slug}.json"
        if json_path.exists() and source_dir.exists():
            successes.append(title)
            print(f"[keep] {item['publication_date']} {title}")
            continue
        try:
            match = arxiv_query(title, pause=args.pause)
            if not match:
                print(f"[skip:no-arxiv-title] {item['publication_date']} {title}")
                continue
            item.update(match)
            if not download_source(item["arxiv_id"], source_dir):
                print(f"[skip:no-latex-src] {item['publication_date']} {title} -> {item['arxiv_id']}")
                shutil.rmtree(source_dir, ignore_errors=True)
                continue
            parse_to_json(source_dir, json_path, item)
            successes.append(title)
            print(f"[ok:{len(successes)}] {item['publication_date']} {title} -> {item['arxiv_id']}")
        except Exception as exc:
            print(f"[skip:error] {item['publication_date']} {title}: {type(exc).__name__}: {exc}")
            shutil.rmtree(source_dir, ignore_errors=True)
            json_path.unlink(missing_ok=True)

    manifest_path = output_root / "selected_manifest.json"
    selected = []
    for title in successes:
        slug = slugify(title)
        data = json.loads((output_root / f"{slug}.json").read_text(encoding="utf-8"))
        selected.append(
            {
                "title": data["title"],
                "publication_date": data["publication_date"],
                "query": data["query"],
                "arxiv_id": data["arxiv_id"],
                "source_dir": data["latex_source_dir"],
                "json": str(output_root / f"{slug}.json"),
            }
        )
    manifest_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} papers. Manifest: {manifest_path}")
    return 0 if len(selected) >= args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())

