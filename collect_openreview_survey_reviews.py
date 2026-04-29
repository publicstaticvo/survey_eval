import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://api2.openreview.net"
OUT_DIR = Path("golden")

SEARCH_TERMS = [
    "survey",
    "literature review",
    "systematic review",
    "systematized literature review",
    "scoping review",
    "comprehensive review",
    "taxonomy",
    "tutorial",
]

CANDIDATE_INVITATIONS = [
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

TITLE_RE = re.compile(
    r"\b(survey|taxonomy|tutorial)\b|literature review|systematic review|"
    r"systematized literature review|scoping review|comprehensive review",
    re.I,
)

STRONG_SURVEY_RE = re.compile(
    r"\b(a|an|the)?\s*(comprehensive\s+)?survey\b|"
    r"\bsurvey\s+(on|of|for|about|in)\b|"
    r"\bliterature review\b|\bsystematic review\b|"
    r"\bsystematized literature review\b|\bscoping review\b|"
    r"\bcomprehensive survey\b",
    re.I,
)

NON_LITERATURE_SURVEY_RE = re.compile(
    r"survey simulations?|survey design|survey responses?|survey questionnaire|"
    r"questionnaire|polling|census|survey equipment|gnss-low cost survey|"
    r"academic survey tasks?|literature review generation|literature survey automation|"
    r"catalogue generation for literature review|llms for literature review|"
    r"match the (observations|conclusions) of systematic reviews",
    re.I,
)

REVIEW_INV_RE = re.compile(
    r"Official_Review|Meta_Review|Public_Review|Review$|/-/Review|Review_Form",
    re.I,
)

AUTHOR_REPLY_RE = re.compile(
    r"Rebuttal|Author|Response|Comment|Revision|Camera_Ready|Decision",
    re.I,
)


def request_json(path, params=None, retries=4):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "survey-eval-openreview-collector/0.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def value(obj, key, default=""):
    if not isinstance(obj, dict):
        return default
    v = obj.get(key, default)
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def as_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return "\n".join(as_text(i) for i in x if as_text(i))
    if isinstance(x, dict):
        if "value" in x:
            return as_text(x["value"])
        return "\n".join(as_text(v) for v in x.values() if as_text(v))
    return str(x)


def title_from_search_note(note):
    forum_content = note.get("forumContent") or {}
    title = value(forum_content, "title")
    if title:
        return title
    invitations = " ".join(note.get("invitations") or [])
    if "/-/Submission" in invitations:
        return value(note.get("content") or {}, "title")
    return ""


def collect_candidates():
    candidates = {}
    for term in SEARCH_TERMS:
        for offset in range(0, 14000, 1000):
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
            notes = data.get("notes") or []
            if not notes:
                break
            for note in notes:
                if note.get("domain") == "DBLP.org":
                    continue
                title = title_from_search_note(note)
                if not title or not TITLE_RE.search(title):
                    continue
                forum = note.get("forum") or note.get("id")
                if forum:
                    candidates[forum] = {
                        "forum": forum,
                        "domain": note.get("domain", ""),
                        "title": title,
                    }
            if len(notes) < 1000:
                break
            time.sleep(0.25)
        time.sleep(0.5)
    for invitation in CANDIDATE_INVITATIONS:
        for offset in range(0, 5000, 1000):
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
                title = value(note.get("content") or {}, "title")
                if not title or not TITLE_RE.search(title):
                    continue
                forum = note.get("forum") or note.get("id")
                if forum:
                    candidates[forum] = {
                        "forum": forum,
                        "domain": note.get("domain", ""),
                        "title": title,
                    }
            if len(notes) < 1000:
                break
            time.sleep(0.2)
    return candidates


def get_submission(notes, fallback):
    submissions = [
        n for n in notes if "/-/Submission" in " ".join(n.get("invitations") or [])
    ]
    if submissions:
        submissions.sort(key=lambda n: n.get("cdate", 0))
        return submissions[0]
    for n in notes:
        title = value(n.get("content") or {}, "title")
        if title and (n.get("forum") == n.get("id")):
            return n
    return {"content": {"title": {"value": fallback.get("title", "")}}, "forum": fallback["forum"]}


def is_review_note(note):
    invitations = " ".join(note.get("invitations") or [])
    if not REVIEW_INV_RE.search(invitations):
        return False
    if AUTHOR_REPLY_RE.search(invitations) and "Official_Review" not in invitations:
        return False
    content_text = "\n".join(as_text(v) for v in (note.get("content") or {}).values())
    return len(content_text.strip()) >= 250


def extract_review(note):
    content = note.get("content") or {}
    fields = {}
    for k, v in content.items():
        text = as_text(v).strip()
        if text:
            fields[k] = text
    main_text_parts = []
    preferred = [
        "review",
        "summary",
        "strengths",
        "weaknesses",
        "questions",
        "limitations",
        "soundness",
        "presentation",
        "contribution",
        "comment",
        "main_review",
    ]
    for key in preferred:
        if key in fields:
            main_text_parts.append(f"{key}: {fields[key]}")
    if not main_text_parts:
        main_text_parts = [f"{k}: {v}" for k, v in fields.items()]
    invitations = note.get("invitations") or []
    return {
        "review_id": note.get("id", ""),
        "review_url": f"https://openreview.net/forum?id={note.get('forum')}&noteId={note.get('id')}",
        "invitations": invitations,
        "signature": " ".join(note.get("signatures") or []),
        "cdate": note.get("cdate"),
        "mdate": note.get("mdate"),
        "content": fields,
        "text": "\n\n".join(main_text_parts),
    }


def pdf_url(pdf):
    if not pdf:
        return ""
    if isinstance(pdf, list):
        pdf = pdf[0] if pdf else ""
    if isinstance(pdf, dict):
        pdf = pdf.get("value", "")
    if not pdf:
        return ""
    if str(pdf).startswith("http"):
        return str(pdf)
    return "https://openreview.net" + str(pdf)


def safe_name(index, title, forum):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    slug = slug[:90] or forum
    return f"{index:03d}_{slug}_{forum}.json"


def console_safe(text):
    return str(text).encode("gbk", "replace").decode("gbk")


def normalized_title(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def is_literature_survey(record):
    title = record["paper_title"]
    abstract = record.get("abstract") or ""
    if NON_LITERATURE_SURVEY_RE.search(title):
        return False
    if re.search(r"\btutorial\b", title, re.I) and not re.search(r"\bsurvey\b", title, re.I):
        return False
    if STRONG_SURVEY_RE.search(title):
        return True
    if "taxonomy" in title.lower():
        return bool(re.search(r"\b(survey|review|literature|systematic|comprehensive)\b", abstract, re.I))
    return False


def record_score(record):
    title = record["paper_title"]
    strong = 1 if STRONG_SURVEY_RE.search(title) else 0
    lit = 1 if is_literature_survey(record) else 0
    taxonomy_only = 1 if ("taxonomy" in title.lower() and not strong) else 0
    return (
        lit,
        strong,
        -taxonomy_only,
        min(record["review_count"], 6),
        record["review_total_chars"],
    )


def build_record(candidate):
    data = request_json("/notes", {"forum": candidate["forum"], "limit": 300})
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
    long_reviews = sum(1 for r in reviews if len(r["text"]) >= 700)
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates()
    print(f"candidates={len(candidates)}")
    records = []
    for i, cand in enumerate(candidates.values(), 1):
        try:
            record = build_record(cand)
        except Exception as exc:
            print(f"skip_error {cand['forum']} {exc}")
            continue
        if record:
            records.append(record)
            print(
                f"accepted={len(records)} reviews={record['review_count']} "
                f"chars={record['review_total_chars']} "
                f"{console_safe(record['paper_title'][:90])}"
            )
        if len(records) >= 90:
            break
        if i % 10 == 0:
            time.sleep(0.8)
    deduped = []
    seen_titles = set()
    for record in sorted(records, key=record_score, reverse=True):
        norm = normalized_title(record["paper_title"])
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        deduped.append(record)
    survey_like = [r for r in deduped if is_literature_survey(r)]
    fallback = [r for r in deduped if r not in survey_like]
    selected = (survey_like + fallback)[:50]
    for old in OUT_DIR.glob("*.json"):
        old.unlink()
    for idx, record in enumerate(selected, 1):
        path = OUT_DIR / safe_name(idx, record["paper_title"], record["openreview_forum_id"])
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path = OUT_DIR / "_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "file": safe_name(i, r["paper_title"], r["openreview_forum_id"]),
                    "title": r["paper_title"],
                    "review_page_url": r["review_page_url"],
                    "review_count": r["review_count"],
                    "review_total_chars": r["review_total_chars"],
                }
                for i, r in enumerate(selected, 1)
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={len(selected)} out_dir={OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
