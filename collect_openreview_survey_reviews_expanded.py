from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collect_openreview_survey_reviews import (
    API,
    NON_LITERATURE_SURVEY_RE,
    STRONG_SURVEY_RE,
    TITLE_RE,
    console_safe,
    extract_review,
    get_submission,
    normalized_title,
    pdf_url,
    title_from_search_note,
    value,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "golden" / "data"
STATE_DIR = ROOT / "golden" / ".openreview_state"
STATE_PATH = STATE_DIR / "expanded_crawler_state.json"
INDEX_PATH = OUT_DIR / "_index.json"
EXTRA_INDEX_PATH = OUT_DIR / "_index_extra.json"
SUMMARY_PATH = OUT_DIR / "_expanded_crawl_summary.json"
REQUEST_RETRIES = 3
REQUEST_TIMEOUT = 20
REQUEST_INTERVAL = 1.0
LAST_REQUEST_TIME = 0.0

SEARCH_TERMS = [
    "a survey",
    "survey",
    "surveys",
    "comprehensive survey",
    "systematic review",
    "systematized literature review",
    "scoping review",
    "literature review",
    "comprehensive review",
    "review",
    "taxonomy",
    "taxonomies",
    "tutorial",
    "overview",
    "primer",
    "roadmap",
    "state of the art",
]

ARR_MONTHS = ["January", "February", "April", "June", "August", "October", "December"]

VENUE_DOMAIN_TEMPLATES = [
    ("ICLR.cc/{year}/Conference", range(2018, 2027)),
    ("NeurIPS.cc/{year}/Conference", range(2018, 2027)),
    ("NeurIPS.cc/{year}/Track/Datasets_and_Benchmarks", range(2021, 2027)),
    ("NeurIPS.cc/{year}/Datasets_and_Benchmarks_Track", range(2021, 2027)),
    ("ICML.cc/{year}/Conference", range(2018, 2027)),
    ("AISTATS.org/{year}/Conference", range(2018, 2027)),
    ("UAI.org/{year}/Conference", range(2018, 2027)),
    ("AAAI.org/{year}/Conference", range(2020, 2027)),
    ("IJCAI.org/{year}/Conference", range(2020, 2027)),
    ("EMNLP/{year}/Conference", range(2020, 2027)),
    ("aclweb.org/ACL/{year}/Conference", range(2020, 2027)),
    ("aclweb.org/NAACL/{year}/Conference", range(2021, 2027)),
    ("aclweb.org/EACL/{year}/Conference", range(2021, 2027)),
    ("COLING/{year}/Conference", range(2020, 2027)),
    ("colmweb.org/COLM/{year}/Conference", range(2024, 2027)),
    ("ACM.org/TheWebConf/{year}/Conference", range(2020, 2027)),
    ("KDD.org/{year}/Conference", range(2020, 2027)),
    ("SIGIR.org/SIGIR/{year}/Conference", range(2020, 2027)),
    ("WSDM.com/{year}/Conference", range(2020, 2027)),
    ("CIKM/{year}/Conference", range(2020, 2027)),
    ("MIDL.io/{year}/Conference", range(2020, 2027)),
    ("CVPR.thecvf.com/{year}/Conference", range(2020, 2027)),
    ("ICCV.thecvf.com/{year}/Conference", range(2021, 2027, 2)),
    ("ECCV.ecva.net/{year}/Conference", range(2020, 2027, 2)),
    ("ACMMM.org/{year}/Conference", range(2020, 2027)),
    ("ICSE/{year}/Conference", range(2020, 2027)),
    ("FSE/{year}/Conference", range(2020, 2027)),
    ("CHI/{year}/Conference", range(2020, 2027)),
]

SOFT_SURVEY_TITLE_RE = re.compile(
    r"\b(overview|primer|roadmap|taxonomy|taxonomies|tutorial|perspective|state[- ]of[- ]the[- ]art)\b",
    re.I,
)
ABSTRACT_SURVEY_RE = re.compile(
    r"\b(this|we|paper|work|study|article)\s+(presents?|provides?|offers?|conducts?|surveys?|reviews?|summari[sz]es?|"
    r"systemati[sz]es?|categor[iy]zes?|taxonomi[sz]es?)\b.{0,220}\b(survey|review|overview|taxonomy|taxonomies|literature|field|area|methods?|approaches?|models?|benchmarks?)\b|"
    r"\b(comprehensive|systematic|scoping|critical|structured|extensive)\s+(survey|review|overview|taxonomy)\b|"
    r"\b(survey|review|overview|taxonomy)\s+of\s+(the\s+)?(literature|existing|current|recent|prior|field|area|methods?|approaches?|models?|benchmarks?)\b|"
    r"\bwe\s+(survey|review|summari[sz]e|systemati[sz]e|categor[iy]ze)\s+(existing|current|recent|prior|the\s+literature)\b",
    re.I | re.S,
)
ABSTRACT_NOISE_RE = re.compile(
    r"\b(peer review|reviewer|review process|review score|review comments?|rebuttal|response|appendix|"
    r"survey participants?|survey respondents?|survey data|survey question(?:naire)?s?|user survey|human survey)\b",
    re.I,
)


def request_json(path: str, params: dict[str, Any] | None = None, retries: int | None = None) -> dict[str, Any]:
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    if retries is None:
        retries = REQUEST_RETRIES
    last_error: Exception | None = None
    global LAST_REQUEST_TIME
    for attempt in range(retries):
        wait = REQUEST_INTERVAL - (time.monotonic() - LAST_REQUEST_TIME)
        if wait > 0:
            time.sleep(wait)
        LAST_REQUEST_TIME = time.monotonic()
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "survey-eval-openreview-expanded-crawler/0.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 404}:
                return {}
            last_error = exc
            if exc.code == 429:
                time.sleep(max(10.0, REQUEST_INTERVAL * 5))
            else:
                time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def search_domains(include_tmlr: bool) -> list[str]:
    domains = {
        template.format(year=year)
        for template, years in VENUE_DOMAIN_TEMPLATES
        for year in years
    }
    domains.update(
        f"aclweb.org/ACL/ARR/{year}/{month}"
        for year in range(2021, 2027)
        for month in ARR_MONTHS
    )
    if include_tmlr:
        domains.add("TMLR")
    return sorted(domains)


def is_review_note(note: dict[str, Any]) -> bool:
    invitations = " ".join(note.get("invitations") or [])
    if not re.search(r"Official_Review|Public_Review|Review$|/-/Review|Review_Form", invitations, re.I):
        return False
    if re.search(r"Rebuttal|Author|Response|Comment|Revision|Camera_Ready|Decision", invitations, re.I):
        return "Official_Review" in invitations
    content_text = "\n".join(str(value).strip() for value in (note.get("content") or {}).values())
    return len(content_text) >= 250


def survey_filter(title: str, abstract: str = "", *, source_field: str = "title") -> tuple[bool, str]:
    title = title or ""
    abstract = abstract or ""
    joined = f"{title}\n{abstract}"
    if not title or NON_LITERATURE_SURVEY_RE.search(title) or ABSTRACT_NOISE_RE.search(title):
        return False, "noise-title"
    if STRONG_SURVEY_RE.search(title) or TITLE_RE.search(title):
        return True, "strong-title"
    if SOFT_SURVEY_TITLE_RE.search(title) and ABSTRACT_SURVEY_RE.search(abstract):
        return True, "soft-title-with-survey-abstract"
    if source_field == "abstract" and ABSTRACT_SURVEY_RE.search(abstract) and not ABSTRACT_NOISE_RE.search(joined):
        return True, "survey-abstract"
    return False, "not-survey-like"


def is_survey_like(title: str, abstract: str = "", *, source_field: str = "title") -> bool:
    return survey_filter(title, abstract, source_field=source_field)[0]


def candidate_from_note(note: dict[str, Any], source_task: str, source_field: str) -> dict[str, Any] | None:
    invitations = " ".join(note.get("invitations") or [])
    forum_content = note.get("forumContent") or {}
    is_submission = "/-/Submission" in invitations or note.get("forum") == note.get("id")
    if not forum_content and not is_submission:
        return None
    title = value(forum_content, "title") if forum_content else value(note.get("content") or {}, "title")
    if not title:
        title = title_from_search_note(note)
    abstract = value(forum_content or note.get("content") or {}, "abstract")
    keep, reason = survey_filter(title, abstract, source_field=source_field)
    if not keep:
        return None
    forum = note.get("forum") or note.get("id")
    if not forum:
        return None
    return {
        "forum": forum,
        "domain": note.get("domain", ""),
        "title": title,
        "abstract": abstract,
        "source_tasks": [source_task],
        "survey_filter_reason": reason,
    }


def merge_candidate(candidates: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> None:
    existing = candidates.get(candidate["forum"])
    if existing is None:
        candidates[candidate["forum"]] = candidate
        return
    existing.setdefault("source_tasks", []).extend(candidate.get("source_tasks", []))
    if not existing.get("domain") and candidate.get("domain"):
        existing["domain"] = candidate["domain"]
    if not existing.get("abstract") and candidate.get("abstract"):
        existing["abstract"] = candidate["abstract"]


def search_task(term: str, domain: str, field: str, page_limit: int | None) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    limit = 1000
    page = 0
    while page_limit is None or page < page_limit:
        data = request_json(
            "/notes/search",
            {
                "term": term,
                "type": "terms",
                "content": field,
                "group": domain,
                "limit": limit,
                "offset": page * limit,
            },
        )
        notes = data.get("notes") or []
        if not notes:
            break
        source_task = f"search::{field}::{term}::{domain}"
        for note in notes:
            if note.get("domain") == "DBLP.org":
                continue
            candidate = candidate_from_note(note, source_task, field)
            if candidate:
                merge_candidate(candidates, candidate)
        if len(notes) < limit:
            break
        page += 1
        time.sleep(0.1)
    return candidates


def collect_candidates(
    include_tmlr: bool,
    page_limit: int | None,
    target_candidates: int | None,
    task_limit: int | None,
    domains_override: list[str] | None,
    terms_override: list[str] | None,
    fields_override: list[str] | None,
) -> dict[str, dict[str, Any]]:
    state = read_json(STATE_PATH, {"completed_tasks": [], "candidates": {}, "errors": {}})
    completed = set(state.get("completed_tasks") or [])
    candidates = dict(state.get("candidates") or {})
    domains = sorted(domains_override) if domains_override else search_domains(include_tmlr)
    terms = terms_override or SEARCH_TERMS
    fields = fields_override or ["title", "abstract"]
    attempted_tasks = 0
    for domain in domains:
        for field in fields:
            for term in terms:
                task_id = f"search::{field}::{term}::{domain}"
                if task_id in completed:
                    continue
                if task_limit is not None and attempted_tasks >= task_limit:
                    return candidates
                attempted_tasks += 1
                try:
                    found = search_task(term, domain, field, page_limit)
                    for candidate in found.values():
                        merge_candidate(candidates, candidate)
                    completed.add(task_id)
                    state["completed_tasks"] = sorted(completed)
                    state["candidates"] = candidates
                    write_json(STATE_PATH, state)
                    print(f"task_done found={len(found)} total_candidates={len(candidates)} {task_id}", flush=True)
                except Exception as exc:
                    state.setdefault("errors", {})[task_id] = str(exc)
                    write_json(STATE_PATH, state)
                    print(f"task_error {task_id} {exc}", flush=True)
                if target_candidates and len(candidates) >= target_candidates:
                    return candidates
                time.sleep(0.1)
    return candidates


def load_existing() -> tuple[set[str], set[str], int]:
    forum_ids: set[str] = set()
    titles: set[str] = set()
    max_index = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        match = re.match(r"(\d+)_", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("openreview_forum_id"):
            forum_ids.add(record["openreview_forum_id"])
        if record.get("paper_title"):
            titles.add(normalized_title(record["paper_title"]))
    return forum_ids, titles, max_index


def safe_name(index: int, title: str, forum: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    slug = slug[:90] or "untitled"
    return f"{index:03d}_{slug}_{forum}.json"


def build_record(
    candidate: dict[str, Any],
    min_review_count: int,
    min_review_chars: int,
    min_long_review_count: int,
    min_long_review_chars: int,
) -> dict[str, Any] | None:
    data = request_json("/notes", {"forum": candidate["forum"], "limit": 500})
    notes = data.get("notes") or []
    if not notes:
        return None
    sub = get_submission(notes, candidate)
    content = sub.get("content") or {}
    title = value(content, "title") or candidate.get("title", "")
    abstract = value(content, "abstract") or candidate.get("abstract", "")
    source_tasks = " ".join(candidate.get("source_tasks") or [])
    source_field = "abstract" if "search::abstract::" in source_tasks else "title"
    keep, reason = survey_filter(title, abstract, source_field=source_field)
    if not keep:
        return None
    reviews = [extract_review(note) for note in notes if is_review_note(note)]
    reviews.sort(key=lambda r: (r.get("cdate") or 0, r.get("review_id") or ""))
    total_chars = sum(len(review["text"]) for review in reviews)
    long_reviews = sum(1 for review in reviews if len(review["text"]) >= min_long_review_chars)
    if len(reviews) < min_review_count or total_chars < min_review_chars or long_reviews < min_long_review_count:
        return None
    forum = sub.get("forum") or candidate["forum"]
    cdate = sub.get("cdate")
    return {
        "source": "OpenReview",
        "review_page_url": f"https://openreview.net/forum?id={forum}",
        "paper_title": title,
        "paper_url": pdf_url(value(content, "pdf")),
        "openreview_forum_id": forum,
        "submission_note_id": sub.get("id", ""),
        "submission_invitation": " ".join(sub.get("invitations") or []),
        "submission_cdate": cdate,
        "submission_mdate": sub.get("mdate"),
        "submission_date": datetime.fromtimestamp(cdate / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if cdate else "",
        "venue": value(content, "venue"),
        "venue_id": value(content, "venueid"),
        "domain": sub.get("domain") or candidate.get("domain", ""),
        "abstract": abstract,
        "survey_filter_reason": reason,
        "review_count": len(reviews),
        "review_total_chars": total_chars,
        "reviews": reviews,
    }


def rebuild_indexes(extra_files: list[dict[str, Any]]) -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "file": path.name,
                "title": record.get("paper_title", ""),
                "review_page_url": record.get("review_page_url", ""),
                "review_count": record.get("review_count", len(record.get("reviews") or [])),
                "review_total_chars": record.get(
                    "review_total_chars",
                    sum(len(review.get("text", "")) for review in record.get("reviews") or []),
                ),
                "domain": record.get("domain", ""),
                "venue": record.get("venue", ""),
                "venue_id": record.get("venue_id", ""),
            }
        )
    write_json(INDEX_PATH, rows)
    write_json(EXTRA_INDEX_PATH, extra_files)


def append_records(
    records: list[dict[str, Any]],
    existing_forums: set[str],
    existing_titles: set[str],
    max_index: int,
) -> list[dict[str, Any]]:
    written = []
    next_index = max_index + 1
    for record in sorted(records, key=lambda r: (r.get("domain", ""), normalized_title(r.get("paper_title", "")))):
        forum = record.get("openreview_forum_id", "")
        title_key = normalized_title(record.get("paper_title", ""))
        if forum in existing_forums or title_key in existing_titles:
            continue
        filename = safe_name(next_index, record.get("paper_title", ""), forum)
        path = OUT_DIR / filename
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(
            {
                "file": filename,
                "title": record.get("paper_title", ""),
                "review_page_url": record.get("review_page_url", ""),
                "review_count": record.get("review_count", 0),
                "review_total_chars": record.get("review_total_chars", 0),
                "domain": record.get("domain", ""),
                "venue": record.get("venue", ""),
                "venue_id": record.get("venue_id", ""),
            }
        )
        existing_forums.add(forum)
        existing_titles.add(title_key)
        next_index += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded incremental OpenReview crawler for survey papers with public reviews.")
    parser.add_argument("--target-total", type=int, default=500, help="Stop after this many JSON records exist in golden/data; use -1 for no target.")
    parser.add_argument("--target-candidates", type=int, default=0, help="Stop candidate discovery after this many candidates; 0 means no candidate cap.")
    parser.add_argument("--task-limit", type=int, default=0, help="Stop after this many unfinished search tasks; useful for smoke tests.")
    parser.add_argument("--domain", action="append", default=[], help="Restrict discovery to one OpenReview group/domain; repeatable.")
    parser.add_argument("--term", action="append", default=[], help="Restrict discovery to one search term; repeatable.")
    parser.add_argument("--field", action="append", choices=["title", "abstract"], default=[], help="Restrict discovery to title or abstract; repeatable.")
    parser.add_argument("--include-tmlr", action="store_true", help="Also search TMLR. Default prioritizes non-TMLR venues.")
    parser.add_argument("--page-limit", type=int, default=None, help="Search pages per term/domain; useful for smoke tests.")
    parser.add_argument("--forum-offset", type=int, default=0, help="Skip this many sorted new candidate forums before fetching.")
    parser.add_argument("--forum-limit", type=int, default=0, help="Maximum new candidate forums to fetch after discovery; 0 means no cap.")
    parser.add_argument("--min-review-count", type=int, default=2)
    parser.add_argument("--min-review-chars", type=int, default=1800)
    parser.add_argument("--min-long-review-count", type=int, default=1)
    parser.add_argument("--min-long-review-chars", type=int, default=600)
    parser.add_argument("--request-retries", type=int, default=3, help="Retries per OpenReview request; lower for exploratory batches.")
    parser.add_argument("--request-timeout", type=int, default=20, help="Timeout in seconds for each OpenReview request.")
    parser.add_argument("--request-interval", type=float, default=1.0, help="Minimum seconds between OpenReview requests.")
    parser.add_argument("--reset-state", action="store_true", help="Forget completed search tasks and rediscover candidates.")
    parser.add_argument("--discover-only", action="store_true", help="Only discover candidate forums and update state; do not fetch reviews or write data files.")
    parser.add_argument("--skip-discovery", action="store_true", help="Use candidates already stored in expanded_crawler_state.json and only fetch forum details.")
    args = parser.parse_args()

    global REQUEST_RETRIES, REQUEST_TIMEOUT, REQUEST_INTERVAL
    REQUEST_RETRIES = max(1, args.request_retries)
    REQUEST_TIMEOUT = max(3, args.request_timeout)
    REQUEST_INTERVAL = max(0.0, args.request_interval)

    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()

    existing_forums, existing_titles, max_index = load_existing()
    existing_total = len(existing_forums)
    if args.skip_discovery:
        candidates = read_json(STATE_PATH, {"candidates": {}}).get("candidates") or {}
    else:
        candidates = collect_candidates(
            include_tmlr=args.include_tmlr,
            page_limit=args.page_limit,
            target_candidates=args.target_candidates or None,
            task_limit=args.task_limit or None,
            domains_override=args.domain or None,
            terms_override=args.term or None,
            fields_override=args.field or None,
        )
    if args.discover_only:
        summary = {
            "existing_records_before": existing_total,
            "candidate_count": len(candidates),
            "new_candidate_forums": sum(1 for forum in candidates if forum not in existing_forums),
            "total_records_after": existing_total,
            "mode": "discover_only",
        }
        write_json(SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    def candidate_priority(item: tuple[str, dict[str, Any]]) -> tuple[int, str, str]:
        forum, candidate = item
        reason = candidate.get("survey_filter_reason", "")
        rank = {"strong-title": 0, "soft-title-with-survey-abstract": 1, "survey-abstract": 2}.get(reason, 3)
        return rank, candidate.get("domain", ""), normalized_title(candidate.get("title", ""))

    ordered_candidates = [
        candidate for forum, candidate in sorted(candidates.items(), key=candidate_priority) if forum not in existing_forums
    ]
    if args.forum_offset > 0:
        ordered_candidates = ordered_candidates[args.forum_offset:]
    if args.forum_limit > 0:
        ordered_candidates = ordered_candidates[: args.forum_limit]
    accepted = []
    rejected = 0
    errors: dict[str, str] = {}
    for idx, candidate in enumerate(ordered_candidates, 1):
        if args.target_total > 0 and existing_total + len(accepted) >= args.target_total:
            break
        if normalized_title(candidate.get("title", "")) in existing_titles:
            continue
        print(
            f"checking_forum={idx}/{len(ordered_candidates)} {candidate['forum']} "
            f"{console_safe(candidate.get('title', '')[:90])}",
            flush=True,
        )
        try:
            record = build_record(
                candidate,
                args.min_review_count,
                args.min_review_chars,
                args.min_long_review_count,
                args.min_long_review_chars,
            )
        except Exception as exc:
            errors[candidate["forum"]] = str(exc)
            print(f"forum_error {candidate['forum']} {exc}", flush=True)
            continue
        if record is None:
            rejected += 1
            continue
        accepted.append(record)
        print(
            f"accepted_new={len(accepted)} reviews={record['review_count']} "
            f"chars={record['review_total_chars']} {console_safe(record['paper_title'][:90])}",
            flush=True,
        )
        if idx % 10 == 0:
            time.sleep(0.4)

    written = append_records(accepted, existing_forums, existing_titles, max_index)
    rebuild_indexes(written)
    venue_counts = Counter(row["domain"] or row["venue_id"] or row["venue"] for row in written)
    summary = {
        "existing_records_before": existing_total,
        "candidate_count": len(candidates),
        "processed_new_candidates": len(ordered_candidates),
        "accepted_new": len(accepted),
        "written_new": len(written),
        "rejected_new": rejected,
        "errors": errors,
        "total_records_after": existing_total + len(written),
        "new_records_by_domain": dict(sorted(venue_counts.items())),
        "index_file": str(INDEX_PATH),
        "extra_index_file": str(EXTRA_INDEX_PATH),
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

















