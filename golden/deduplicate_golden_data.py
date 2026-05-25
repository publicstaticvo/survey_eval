import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_PATH = DATA_DIR / "_index.json"
EXTRA_INDEX_PATH = DATA_DIR / "_index_extra.json"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title or "").strip("_").lower()
    return slug[:90] or "untitled"


def canonical_record(record: dict) -> dict:
    reviews = []
    for review in record.get("reviews") or []:
        reviews.append(
            {
                "review_id": review.get("review_id", ""),
                "review_url": review.get("review_url", ""),
                "invitations": review.get("invitations") or [],
                "signature": review.get("signature", ""),
                "content": review.get("content") or {},
                "text": review.get("text", ""),
            }
        )
    reviews.sort(key=lambda r: (r["review_id"], r["review_url"], r["text"]))
    return {
        "source": record.get("source", ""),
        "review_page_url": record.get("review_page_url", ""),
        "paper_title": record.get("paper_title", ""),
        "paper_url": record.get("paper_url", ""),
        "openreview_forum_id": record.get("openreview_forum_id", ""),
        "venue": record.get("venue", ""),
        "venue_id": record.get("venue_id", ""),
        "domain": record.get("domain", ""),
        "abstract": record.get("abstract", ""),
        "review_count": record.get("review_count", len(reviews)),
        "review_total_chars": record.get(
            "review_total_chars", sum(len(r["text"]) for r in reviews)
        ),
        "reviews": reviews,
    }


def content_hash(record: dict) -> str:
    payload = json.dumps(canonical_record(record), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def same_forum_hash(record: dict) -> str:
    payload = json.dumps(
        {
            "openreview_forum_id": record.get("openreview_forum_id", ""),
            "paper_title": normalize_title(record.get("paper_title", "")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records():
    records = []
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"failed to parse {path}: {exc}") from exc
        title = record.get("paper_title", "")
        forum = record.get("openreview_forum_id", "")
        records.append(
            {
                "path": path,
                "record": record,
                "title_key": normalize_title(title),
                "forum": forum,
                "content_hash": content_hash(record),
                "same_forum_hash": same_forum_hash(record),
            }
        )
    return records


def choose_keep(items):
    def score(item):
        path = item["path"]
        record = item["record"]
        match = re.match(r"(\d+)_", path.name)
        number = int(match.group(1)) if match else 10**9
        return (
            -int(record.get("review_count") or 0),
            -int(record.get("review_total_chars") or 0),
            number,
            path.name,
        )

    return sorted(items, key=score)[0]


def plan_dedup(records):
    duplicate_paths = set()
    report = {
        "total_files_before": len(records),
        "exact_content_duplicate_groups": [],
        "same_forum_duplicate_groups": [],
        "same_title_different_content_groups": [],
        "kept": [],
        "removed": [],
    }

    by_content = defaultdict(list)
    for item in records:
        by_content[item["content_hash"]].append(item)
    for group in by_content.values():
        if len(group) <= 1:
            continue
        keep = choose_keep(group)
        removed = [item for item in group if item is not keep]
        duplicate_paths.update(item["path"] for item in removed)
        report["exact_content_duplicate_groups"].append(
            {
                "kept": keep["path"].name,
                "removed": [item["path"].name for item in removed],
                "title": keep["record"].get("paper_title", ""),
                "forum": keep["forum"],
            }
        )

    remaining = [item for item in records if item["path"] not in duplicate_paths]
    by_forum = defaultdict(list)
    for item in remaining:
        by_forum[item["same_forum_hash"]].append(item)
    for group in by_forum.values():
        if len(group) <= 1:
            continue
        keep = choose_keep(group)
        removed = [item for item in group if item is not keep]
        duplicate_paths.update(item["path"] for item in removed)
        report["same_forum_duplicate_groups"].append(
            {
                "kept": keep["path"].name,
                "removed": [item["path"].name for item in removed],
                "title": keep["record"].get("paper_title", ""),
                "forum": keep["forum"],
                "note": "same OpenReview forum/title; content may differ due to recrawl metadata or review updates",
            }
        )

    remaining = [item for item in records if item["path"] not in duplicate_paths]
    by_title = defaultdict(list)
    for item in remaining:
        by_title[item["title_key"]].append(item)
    for group in by_title.values():
        if len(group) <= 1:
            continue
        hashes = {item["content_hash"] for item in group}
        forums = {item["forum"] for item in group}
        if len(hashes) > 1 or len(forums) > 1:
            report["same_title_different_content_groups"].append(
                {
                    "title": group[0]["record"].get("paper_title", ""),
                    "files": [item["path"].name for item in group],
                    "forums": sorted(forums),
                    "content_hashes": sorted(hashes),
                    "action": "kept_all",
                }
            )

    keep_items = [item for item in records if item["path"] not in duplicate_paths]
    report["removed"] = sorted(path.name for path in duplicate_paths)
    report["kept"] = sorted(item["path"].name for item in keep_items)
    report["total_files_after"] = len(keep_items)
    return keep_items, duplicate_paths, report


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_and_remove(duplicate_paths, report):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_backups = sorted(DATA_DIR.glob("_duplicates_backup_*"))
    backup_dir = existing_backups[-1] if existing_backups else DATA_DIR / f"_duplicates_backup_{timestamp}"
    failed = []
    if duplicate_paths:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(duplicate_paths):
            target = backup_dir / path.name
            try:
                if not target.exists():
                    shutil.copy2(str(path), str(target))
                elif file_digest(target) != file_digest(path):
                    target = backup_dir / f"{path.stem}_{timestamp}{path.suffix}"
                    shutil.copy2(str(path), str(target))
                path.unlink()
            except PermissionError as exc:
                failed.append({"file": path.name, "error": str(exc)})
        report["duplicates_backup_dir"] = str(backup_dir)
    else:
        report["duplicates_backup_dir"] = ""
    report["failed_to_remove"] = failed
    if failed:
        raise RuntimeError(
            "Some duplicate files could not be removed. Close any editor handles and rerun. "
            + json.dumps(failed, ensure_ascii=False)
        )
    return backup_dir if duplicate_paths else None


def rewrite_numbered_files(keep_items):
    temp_dir = DATA_DIR / f"_renumber_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    def original_number(item):
        match = re.match(r"(\d+)_", item["path"].name)
        return int(match.group(1)) if match else 10**9

    ordered = sorted(
        keep_items,
        key=lambda item: (
            original_number(item),
            normalize_title(item["record"].get("paper_title", "")),
            item["forum"],
        ),
    )

    index = []
    renames = []
    for idx, item in enumerate(ordered, 1):
        record = item["record"]
        filename = f"{idx:03d}_{slugify(record.get('paper_title', ''))}_{record.get('openreview_forum_id', '')}.json"
        target = temp_dir / filename
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append(
            {
                "file": filename,
                "title": record.get("paper_title", ""),
                "review_page_url": record.get("review_page_url", ""),
                "review_count": record.get("review_count", len(record.get("reviews") or [])),
                "review_total_chars": record.get(
                    "review_total_chars",
                    sum(len(r.get("text", "")) for r in record.get("reviews") or []),
                ),
                "domain": record.get("domain", ""),
                "venue": record.get("venue", ""),
                "venue_id": record.get("venue_id", ""),
            }
        )
        renames.append({"old": item["path"].name, "new": filename})

    for item in keep_items:
        item["path"].unlink()
    for path in temp_dir.glob("*.json"):
        shutil.move(str(path), str(DATA_DIR / path.name))
    temp_dir.rmdir()

    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    EXTRA_INDEX_PATH.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding="utf-8")
    return renames, index


def main():
    records = load_records()
    keep_items, duplicate_paths, report = plan_dedup(records)
    backup_and_remove(duplicate_paths, report)
    renames, index = rewrite_numbered_files(keep_items)
    report["renumbered"] = renames
    report["index_file"] = str(INDEX_PATH)
    report["extra_index_reset"] = str(EXTRA_INDEX_PATH)
    report_path = DATA_DIR / "_deduplicate_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"before={report['total_files_before']}")
    print(f"removed={len(report['removed'])}")
    print(f"after={report['total_files_after']}")
    print(f"same_title_different_content={len(report['same_title_different_content_groups'])}")
    print(f"index_entries={len(index)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
