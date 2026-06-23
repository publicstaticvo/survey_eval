from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API = "https://api2.openreview.net"
GOLDEN_DIR = Path(r"P:\AI4S\survey_eval\golden")
PDF_CONTENT_DIR = GOLDEN_DIR / "pdf_content"


def request_json(path: str, params: dict | None = None, retries: int = 4) -> dict:
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "survey-eval-openreview-backfill/0.1",
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


def get_submission(notes: list[dict], fallback: dict) -> dict:
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
    return {"forum": fallback["forum"]}


def submission_fields(sub: dict) -> dict:
    submission_cdate = sub.get("cdate")
    return {
        "submission_note_id": sub.get("id", ""),
        "submission_invitation": " ".join(sub.get("invitations") or []),
        "submission_cdate": submission_cdate,
        "submission_mdate": sub.get("mdate"),
        "submission_date": datetime.fromtimestamp(submission_cdate / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if submission_cdate else "",
    }


def update_json_file(path: Path, fields: dict) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for key, value in fields.items():
        if data.get(key) != value:
            data[key] = value
            changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def iter_content_files() -> list[Path]:
    return sorted(PDF_CONTENT_DIR.glob("*.json"))


def main() -> None:
    updated = 0
    skipped = 0
    for content_path in iter_content_files():
        data = json.loads(content_path.read_text(encoding="utf-8"))
        forum = data.get("openreview_forum_id")
        if not forum:
            skipped += 1
            continue

        payload = request_json("/notes", {"forum": forum, "limit": 300})
        notes = payload.get("notes") or []
        if not notes:
            skipped += 1
            continue

        sub = get_submission(notes, {"forum": forum})
        fields = submission_fields(sub)

        changed = update_json_file(content_path, fields)
        if changed:
            updated += 1
            print(f"updated {content_path.name}")
        else:
            skipped += 1

    print(f"done updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
