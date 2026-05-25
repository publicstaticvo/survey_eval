import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Incremental OpenReview crawler for CS survey papers with public reviews.
#
# Recommended full run:
#   python -B collect_openreview_survey_reviews_extra.py --target 200 --request-workers 3 --forum-workers 3
#
# Resume behavior:
#   Progress is stored in golden/.openreview_state/crawler_state.json.
#   Re-run the same command to continue after interruption. Use --reset-state only
#   when you intentionally want to re-discover candidates from scratch.
#
# Smoke test:
#   python -B collect_openreview_survey_reviews_extra.py --reset-state --target 1 \
#     --candidate-task-limit 2 --invitation-offset-limit 1
#
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
STATE_DIR = Path("golden/.openreview_state")
STATE_PATH = STATE_DIR / "crawler_state.json"
EXTRA_INDEX = OUT_DIR / "_index_extra.json"

DEFAULT_TARGET = -1
DEFAULT_REQUEST_WORKERS = 3
DEFAULT_FORUM_WORKERS = 3

SEARCH_TERMS = [
    "a survey",
    "survey",
    "comprehensive survey",
    "systematic review",
    "systematized literature review",
    "scoping review",
    "literature review",
    "comprehensive review"
]

ARR_MONTHS = ["February", "April", "June", "August", "October", "December"]

INVITATIONS = [
    "TMLR/-/Submission",
    "ICLR.cc/2023/Conference/-/Submission",
    "ICLR.cc/2024/Conference/-/Submission",
    "ICLR.cc/2025/Conference/-/Submission",
    "ICLR.cc/2026/Conference/-/Submission",
    "NeurIPS.cc/2023/Conference/-/Submission",
    "NeurIPS.cc/2023/Track/Datasets_and_Benchmarks/-/Submission",
    "NeurIPS.cc/2024/Conference/-/Submission",
    "NeurIPS.cc/2024/Datasets_and_Benchmarks_Track/-/Submission",
    "NeurIPS.cc/2025/Conference/-/Submission",
    "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track/-/Submission",
    "NeurIPS.cc/2026/Conference/-/Submission",
    "ICML.cc/2023/Conference/-/Submission",
    "ICML.cc/2024/Conference/-/Submission",
    "ICML.cc/2025/Conference/-/Submission",
    "ICML.cc/2026/Conference/-/Submission",
    "UAI.org/2023/Conference/-/Submission",
    "UAI.org/2024/Conference/-/Submission",
    "UAI.org/2025/Conference/-/Submission",
    "UAI.org/2026/Conference/-/Submission",
    "AISTATS.org/2024/Conference/-/Submission",
    "AISTATS.org/2025/Conference/-/Submission",
    "AISTATS.org/2026/Conference/-/Submission",
    "AAAI.org/2024/Conference/-/Submission",
    "AAAI.org/2025/Conference/-/Submission",
    "AAAI.org/2026/Conference/-/Submission",
    "IJCAI.org/2024/Conference/-/Submission",
    "IJCAI.org/2025/Conference/-/Submission",
    "IJCAI.org/2026/Conference/-/Submission",
    "EMNLP/2023/Conference/-/Submission",
    "EMNLP/2024/Conference/-/Submission",
    "EMNLP/2025/Conference/-/Submission",
    "aclweb.org/ACL/2024/Conference/-/Submission",
    "aclweb.org/ACL/2025/Conference/-/Submission",
    "aclweb.org/ACL/2026/Conference/-/Submission",
    "aclweb.org/NAACL/2024/Conference/-/Submission",
    "aclweb.org/NAACL/2025/Conference/-/Submission",
    "aclweb.org/EACL/2024/Conference/-/Submission",
    "COLING/2025/Conference/-/Submission",
    "colmweb.org/COLM/2024/Conference/-/Submission",
    "colmweb.org/COLM/2025/Conference/-/Submission",
    "ACM.org/TheWebConf/2024/Conference/-/Submission",
    "ACM.org/TheWebConf/2025/Conference/-/Submission",
    "ACM.org/TheWebConf/2026/Conference/-/Submission",
    "KDD.org/2024/Conference/-/Submission",
    "KDD.org/2025/Conference/-/Submission",
    "KDD.org/2026/Conference/-/Submission",
    "SIGIR.org/SIGIR/2024/Conference/-/Submission",
    "SIGIR.org/SIGIR/2025/Conference/-/Submission",
    "WSDM.com/2024/Conference/-/Submission",
    "WSDM.com/2025/Conference/-/Submission",
    "CIKM/2024/Conference/-/Submission",
    "CIKM/2025/Conference/-/Submission",
    "MIDL.io/2024/Conference/-/Submission",
    "MIDL.io/2025/Conference/-/Submission",
    "MIDL.io/2026/Conference/-/Submission",
    "CVPR.thecvf.com/2024/Conference/-/Submission",
    "CVPR.thecvf.com/2025/Conference/-/Submission",
    "ICCV.thecvf.com/2025/Conference/-/Submission",
    "ECCV.ecva.net/2024/Conference/-/Submission",
    "ACMMM.org/2024/Conference/-/Submission",
    "ACMMM.org/2025/Conference/-/Submission",
    "ICSE/2024/Conference/-/Submission",
    "ICSE/2025/Conference/-/Submission",
    "FSE/2024/Conference/-/Submission",
    "FSE/2025/Conference/-/Submission",
    "CHI/2024/Conference/-/Submission",
    "CHI/2025/Conference/-/Submission",
] + [
    f"aclweb.org/ACL/ARR/{year}/{month}/-/Submission"
    for year in [2023, 2024, 2025, 2026]
    for month in ARR_MONTHS
]

SEARCH_DOMAINS = sorted(
    {
        invitation.rsplit("/-/Submission", 1)[0]
        for invitation in INVITATIONS
        if "/-/Submission" in invitation
    }
)

_request_lock = threading.Lock()
_last_request_time = 0.0
_file_write_lock = threading.Lock()


def request_json(path, params=None, retries=100, min_interval=0.25):
    global _last_request_time
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(retries):
        with _request_lock:
            wait = min_interval - (time.monotonic() - _last_request_time)
            if wait > 0:
                time.sleep(wait)
            _last_request_time = time.monotonic()
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "survey-eval-openreview-parallel-crawler/0.2",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last_error}")


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, data):
    with _file_write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def load_state():
    state = read_json(STATE_PATH, {})
    state.setdefault("completed_search_tasks", [])
    state.setdefault("completed_invitation_tasks", [])
    state.setdefault("candidates", {})
    state.setdefault("processed_forums", [])
    state.setdefault("accepted_forums", [])
    state.setdefault("rejected_forums", [])
    state.setdefault("errors", {})
    return state


def save_state(state):
    write_json(STATE_PATH, state)


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


def valid_survey_title(title):
    if not title or not TITLE_RE.search(title):
        return False
    if NON_LITERATURE_SURVEY_RE.search(title):
        return False
    if re.search(r"\btutorial\b", title, re.I) and not re.search(r"\bsurvey\b", title, re.I):
        return False
    return True


def build_record(candidate):
    data = request_json("/notes", {"forum": candidate["forum"], "limit": 350})
    notes = data.get("notes") or []
    if not notes:
        return None
    sub = get_submission(notes, candidate)
    content = sub.get("content") or {}
    title = value(content, "title") or candidate["title"]
    if not valid_survey_title(title):
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


def paged_offsets(page_size, page_limit=None):
    page = 0
    while page_limit is None or page < page_limit:
        yield page * page_size
        page += 1


def search_task(term, domain, page_limit=None):
    candidates = {}
    limit = 1000
    for offset in paged_offsets(limit, page_limit):
        data = request_json(
            "/notes/search",
            {
                "term": term,
                "type": "terms",
                "content": "title",
                "group": domain,
                "limit": limit,
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
            if not valid_survey_title(title):
                continue
            forum = note.get("forum") or note.get("id")
            if forum:
                candidates[forum] = {
                    "forum": forum,
                    "domain": note.get("domain", ""),
                    "title": title,
                    "source_task": f"search::{term}::{domain}",
                }
        if len(notes) < limit:
            break
    return candidates


def invitation_task(invitation, page_limit=None):
    candidates = {}
    limit = 1000
    for offset in paged_offsets(limit, page_limit):
        data = request_json(
            "/notes",
            {"invitation": invitation, "limit": limit, "offset": offset},
        )
        notes = data.get("notes") or []
        if not notes:
            break
        for note in notes:
            title = value(note.get("content") or {}, "title")
            if not valid_survey_title(title):
                continue
            forum = note.get("forum") or note.get("id")
            if forum:
                candidates[forum] = {
                    "forum": forum,
                    "domain": note.get("domain", ""),
                    "title": title,
                    "source_task": f"invitation::{invitation}",
                }
        if len(notes) < limit:
            break
    return candidates


def merge_candidates(state, candidates, existing_forums):
    added = 0
    store = state["candidates"]
    for forum, candidate in candidates.items():
        if forum in existing_forums:
            continue
        if forum not in store:
            store[forum] = candidate
            added += 1
    return added


def run_candidate_stage(
    state,
    existing_forums,
    request_workers,
    candidate_task_limit=None,
    search_page_limit=None,
    invitation_offset_limit=None,
):
    completed_search = set(state["completed_search_tasks"])
    completed_invitation = set(state["completed_invitation_tasks"])

    search_tasks = [
        (term, domain)
        for term in SEARCH_TERMS
        for domain in SEARCH_DOMAINS
        if f"{term}::{domain}" not in completed_search
    ]
    invitation_tasks = [
        invitation
        for invitation in INVITATIONS
        if invitation not in completed_invitation
    ]
    if candidate_task_limit is not None:
        search_tasks = search_tasks[:candidate_task_limit]
        invitation_tasks = invitation_tasks[:candidate_task_limit]

    print(
        f"candidate_stage search_tasks={len(search_tasks)} "
        f"invitation_tasks={len(invitation_tasks)}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=request_workers) as pool:
        future_map = {
            pool.submit(search_task, term, domain, search_page_limit): (
                "search",
                term,
                domain,
            )
            for term, domain in search_tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            kind, term, domain = future_map[future]
            key = f"{term}::{domain}"
            try:
                candidates = future.result()
                added = merge_candidates(state, candidates, existing_forums)
                state["completed_search_tasks"].append(key)
                print(f"search_done added={added} term={console_safe(term)} domain={domain}")
            except Exception as exc:
                state["errors"][f"search::{key}"] = str(exc)
                print(f"search_error term={console_safe(term)} domain={domain} error={exc}")
            save_state(state)

        future_map = {
            pool.submit(invitation_task, invitation, invitation_offset_limit): (
                "invitation",
                invitation,
            )
            for invitation in invitation_tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            _, invitation = future_map[future]
            try:
                candidates = future.result()
                added = merge_candidates(state, candidates, existing_forums)
                state["completed_invitation_tasks"].append(invitation)
                print(f"invitation_done added={added} invitation={invitation}")
            except Exception as exc:
                state["errors"][f"invitation::{invitation}"] = str(exc)
                print(f"invitation_error invitation={invitation} error={exc}")
            save_state(state)


def next_output_index():
    max_idx = 0
    for path in OUT_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        match = re.match(r"(\d+)_", path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def save_record(record):
    idx = next_output_index()
    filename = safe_name(idx, record["paper_title"], record["openreview_forum_id"])
    path = OUT_DIR / filename
    while path.exists():
        idx += 1
        filename = safe_name(idx, record["paper_title"], record["openreview_forum_id"])
        path = OUT_DIR / filename
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return filename


def run_forum_stage(state, existing_forums, existing_titles, target, forum_workers):
    processed = set(state["processed_forums"])
    accepted = set(state["accepted_forums"])
    rejected = set(state["rejected_forums"])
    seen_titles = set(existing_titles)

    for path in OUT_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            title = data.get("paper_title")
            if title:
                seen_titles.add(normalized_title(title))
        except Exception:
            continue

    pending = [
        candidate
        for forum, candidate in state["candidates"].items()
        if forum not in existing_forums
        and forum not in processed
        and forum not in accepted
        and forum not in rejected
    ]
    print(f"forum_stage pending={len(pending)} already_accepted={len(accepted)} target={target}")

    new_index = read_json(EXTRA_INDEX, [])
    saved_count = 0
    state_lock = threading.Lock()

    def handle(candidate):
        forum = candidate["forum"]
        try:
            record = build_record(candidate)
        except Exception as exc:
            return ("error", forum, str(exc), None, None)
        if not record:
            return ("reject", forum, "", None, None)
        norm = normalized_title(record["paper_title"])
        return ("accept", forum, norm, record, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=forum_workers) as pool:
        futures = [pool.submit(handle, candidate) for candidate in pending]
        for future in concurrent.futures.as_completed(futures):
            status, forum, info, record, _ = future.result()
            with state_lock:
                state["processed_forums"].append(forum)
                if status == "accept" and info not in seen_titles:
                    seen_titles.add(info)
                    filename = save_record(record)
                    index_item = {
                        "file": filename,
                        "title": record["paper_title"],
                        "review_page_url": record["review_page_url"],
                        "review_count": record["review_count"],
                        "review_total_chars": record["review_total_chars"],
                        "domain": record["domain"],
                        "venue": record["venue"],
                        "venue_id": record["venue_id"],
                    }
                    new_index.append(index_item)
                    state["accepted_forums"].append(forum)
                    saved_count += 1
                    print(
                        f"saved={saved_count} reviews={record['review_count']} "
                        f"chars={record['review_total_chars']} "
                        f"{console_safe(record['paper_title'][:90])}"
                    )
                elif status == "error":
                    state["errors"][f"forum::{forum}"] = info
                    print(f"forum_error forum={forum} error={info}")
                else:
                    state["rejected_forums"].append(forum)
                write_json(EXTRA_INDEX, new_index)
                save_state(state)
                if saved_count == target: break

    return saved_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--request-workers", type=int, default=DEFAULT_REQUEST_WORKERS)
    parser.add_argument("--forum-workers", type=int, default=DEFAULT_FORUM_WORKERS)
    parser.add_argument("--skip-candidate-stage", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument(
        "--candidate-task-limit",
        type=int,
        default=None,
        help="Debug/smoke-test limit for search and invitation tasks.",
    )
    parser.add_argument(
        "--invitation-offset-limit",
        type=int,
        default=None,
        help="Debug/smoke-test limit for paginated invitation pages.",
    )
    parser.add_argument(
        "--search-page-limit",
        type=int,
        default=None,
        help="Optional cap for search pages per term/domain. Default: stop only on empty/short page.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset_state and STATE_PATH.exists():
        STATE_PATH.unlink()

    existing_forums, existing_titles, _ = load_existing()
    state = load_state()
    print(
        f"existing_files_forums={len(existing_forums)} "
        f"state_candidates={len(state['candidates'])}"
    )

    if not args.skip_candidate_stage:
        run_candidate_stage(
            state,
            existing_forums,
            args.request_workers,
            args.candidate_task_limit,
            args.search_page_limit,
            args.invitation_offset_limit,
        )

    saved = run_forum_stage(
        state,
        existing_forums,
        existing_titles,
        args.target,
        args.forum_workers,
    )
    print(
        f"done saved_new={saved} candidates={len(state['candidates'])} "
        f"state={STATE_PATH.resolve()} index={EXTRA_INDEX.resolve()}"
    )


if __name__ == "__main__":
    main()
