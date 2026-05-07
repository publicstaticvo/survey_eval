import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from collect_openreview_survey_reviews import (
    API,
    AUTHOR_REPLY_RE,
    NON_LITERATURE_SURVEY_RE,
    REVIEW_INV_RE,
    STRONG_SURVEY_RE,
    TITLE_RE,
    as_text,
    console_safe,
    extract_review,
    get_submission,
    is_literature_survey,
    normalized_title,
    pdf_url,
    record_score,
    safe_name,
    title_from_search_note,
    value,
)


OUT_DIR = Path("golden/data")
EXTRA_INDEX = OUT_DIR / "_index_extra.json"
TARGET = 50

SEARCH_TERMS = [
    "a survey",
    "survey",
    "literature review",
    "systematic review",
    "systematized literature review",
    "scoping review",
    "comprehensive review",
    "comprehensive survey",
    "survey and benchmark",
    "survey and evaluation",
]

INVITATIONS = [
    "TMLR/-/Submission",
    "ICLR.cc/2024/Conference/-/Submission",
    "ICLR.cc/2025/Conference/-/Submission",
    "ICLR.cc/2026/Conference/-/Submission",
    "NeurIPS.cc/2023/Conference/-/Submission",
    "NeurIPS.cc/2023/Track/Datasets_and_Benchmarks/-/Submission",
    "NeurIPS.cc/2024/Conference/-/Submission",
    "NeurIPS.cc/2024/Datasets_and_Benchmarks_Track/-/Submission",
    "NeurIPS.cc/2025/Conference/-/Submission",
    "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track/-/Submission",
    "EMNLP/2023/Conference/-/Submission",
    "ACM.org/TheWebConf/2024/Conference/-/Submission",
    "ACM.org/TheWebConf/2025/Conference/-/Submission",
    "colmweb.org/COLM/2025/Conference/-/Submission",
    "aclweb.org/ACL/ARR/2024/April/-/Submission",
    "aclweb.org/ACL/ARR/2024/August/-/Submission",
    "aclweb.org/ACL/ARR/2026/January/-/Submission",
]


def request_json(path, params=None, retries=5):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "survey-eval-openreview-extra-stream/0.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def load_existing():
    forum_ids = set()
    titles = set()
    max_idx = 0
    for path in OUT_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        m = re.match(r"(\d+)_", path.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        forum = data.get("openreview_forum_id")
        if forum:
            forum_ids.add(forum)
        title = data.get("paper_title")
        if title:
            titles.add(normalized_title(title))
    return forum_ids, titles, max_idx


def is_review_note(note):
    invitations = " ".join(note.get("invitations") or [])
    if not REVIEW_INV_RE.search(invitations):
        return False
    if AUTHOR_REPLY_RE.search(invitations) and "Official_Review" not in invitations:
        return False
    content_text = "\n".join(as_text(v) for v in (note.get("content") or {}).values())
    return len(content_text.strip()) >= 250


def build_record(candidate):
    data = request_json("/notes", {"forum": candidate["forum"], "limit": 350})
    notes = data.get("notes") or []
    if not notes:
        return None
    sub = get_submission(notes, candidate)
    content = sub.get("content") or {}
    title = value(content, "title") or candidate["title"]
    if not TITLE_RE.search(title):
        return None
    if NON_LITERATURE_SURVEY_RE.search(title):
        return None
    if re.search(r"\btutorial\b", title, re.I) and not re.search(r"\bsurvey\b", title, re.I):
        return None
    reviews = [extract_review(n) for n in notes if is_review_note(n)]
    reviews.sort(key=lambda r: (r.get("cdate") or 0, r.get("review_id") or ""))
    total_chars = sum(len(r["text"]) for r in reviews)
    long_reviews = sum(1 for r in reviews if len(r["text"]) >= 600)
    if len(reviews) < 2 or total_chars < 2500 or long_reviews < 2:
        return None
    forum = sub.get("forum") or candidate["forum"]
    return {
        "source": "OpenReview",
        "review_page_url": f"https://openreview.net/forum?id={forum}",
        "paper_title": title,
        "paper_url": pdf_url(value(content, "pdf")),
        "openreview_forum_id": forum,
        "venue": value(content, "venue"),
        "venue_id": value(content, "venueid"),
        "domain": sub.get("domain") or candidate.get("domain", ""),
        "abstract": value(content, "abstract"),
        "review_count": len(reviews),
        "review_total_chars": total_chars,
        "reviews": reviews,
    }


def candidate_stream(existing_forums):
    seen = set()
    for term in SEARCH_TERMS:
        for offset in range(0, 12000, 1000):
            try:
                data = request_json(
                    "/notes/search",
                    {
                        "term": term,
                        "type": "terms",
                        "content": "title",
                        "limit": 1000,
                        "offset": offset,
                    },
                )
            except Exception:
                break
            notes = data.get("notes") or []
            if not notes:
                break
            for note in notes:
                if note.get("domain") == "DBLP.org":
                    continue
                forum = note.get("forum") or note.get("id")
                if not forum or forum in existing_forums or forum in seen:
                    continue
                title = title_from_search_note(note)
                if not title or not TITLE_RE.search(title):
                    continue
                seen.add(forum)
                yield {
                    "forum": forum,
                    "domain": note.get("domain", ""),
                    "title": title,
                }
            if len(notes) < 1000:
                break
            time.sleep(0.2)
        time.sleep(0.4)
    for invitation in INVITATIONS:
        for offset in range(0, 14000, 1000):
            try:
                data = request_json(
                    "/notes",
                    {"invitation": invitation, "limit": 1000, "offset": offset},
                )
            except Exception:
                break
            notes = data.get("notes") or []
            if not notes:
                break
            for note in notes:
                forum = note.get("forum") or note.get("id")
                if not forum or forum in existing_forums or forum in seen:
                    continue
                title = value(note.get("content") or {}, "title")
                if not title or not TITLE_RE.search(title):
                    continue
                seen.add(forum)
                yield {
                    "forum": forum,
                    "domain": note.get("domain", ""),
                    "title": title,
                }
            if len(notes) < 1000:
                break
            time.sleep(0.15)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_forums, existing_titles, max_idx = load_existing()
    print(f"existing={len(existing_forums)}")

    accepted = []
    seen_titles = set(existing_titles)
    for i, candidate in enumerate(candidate_stream(existing_forums), 1):
        try:
            record = build_record(candidate)
        except Exception as exc:
            print(f"skip_error {candidate['forum']} {exc}")
            continue
        if not record:
            continue
        norm = normalized_title(record["paper_title"])
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        accepted.append(record)
        print(
            f"accepted={len(accepted)} reviews={record['review_count']} "
            f"chars={record['review_total_chars']} {console_safe(record['paper_title'][:90])}"
        )
        if len(accepted) >= 90:
            break
        if i % 10 == 0:
            time.sleep(0.3)

    accepted.sort(key=record_score, reverse=True)
    survey_like = [r for r in accepted if is_literature_survey(r)]
    fallback = [r for r in accepted if r not in survey_like]
    selected = (survey_like + fallback)[:TARGET]

    index = []
    for offset, record in enumerate(selected, 1):
        idx = max_idx + offset
        filename = safe_name(idx, record["paper_title"], record["openreview_forum_id"])
        (OUT_DIR / filename).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index.append(
            {
                "file": filename,
                "title": record["paper_title"],
                "review_page_url": record["review_page_url"],
                "review_count": record["review_count"],
                "review_total_chars": record["review_total_chars"],
                "domain": record["domain"],
                "venue": record["venue"],
                "venue_id": record["venue_id"],
            }
        )

    EXTRA_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_extra={len(index)} out_dir={OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
