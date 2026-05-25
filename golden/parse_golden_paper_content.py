import asyncio
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from survey_eval.agent.tools.utility.paper_download import (  # noqa: E402
    PaperDownload,
    extract_arxiv_id_from_url,
)
from survey_eval.agent.tools.utility.request_utils import SessionManager  # noqa: E402
from survey_eval.agent.tools.utility.tool_config import ToolConfig  # noqa: E402


DATA_DIR = Path("golden/data")
OUT_DIR = Path("golden/paper_content")
ERROR_PATH = OUT_DIR / "_errors.json"
ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_input_files() -> list[Path]:
    return sorted(path for path in DATA_DIR.glob("*.json") if not path.name.startswith("_"))


def find_arxiv_ids(record: dict) -> list[str]:
    ids = []
    arxiv_id = extract_arxiv_id_from_url(record.get("paper_url", ""))
    if arxiv_id:
        ids.append(arxiv_id)

    chunks = [
        record.get("paper_title", ""),
        record.get("paper_url", ""),
        record.get("abstract", ""),
    ]
    for review in record.get("reviews", []):
        chunks.append(review.get("text", ""))
        chunks.extend(str(value) for value in review.get("content", {}).values())
    for text in chunks:
        ids.extend(ARXIV_RE.findall(text or ""))
    return list(dict.fromkeys(ids))


def extract_skeleton(result: dict) -> dict:
    if not result: return {}
    paper = result.get("result", result)
    if not isinstance(paper, dict): return {}
    skeleton = paper.get("full_content") or paper.get("paper") or {}
    return skeleton if isinstance(skeleton, dict) else {}


async def parse_from_arxiv(downloader: PaperDownload, arxiv_id: str) -> dict:
    result = await asyncio.wait_for(downloader._try_arxiv_source(arxiv_id), timeout=260)
    skeleton = extract_skeleton(result)
    if skeleton.get("sections"): return skeleton
    return {}


async def parse_from_pdf(downloader: PaperDownload, paper_url: str) -> dict:
    if not paper_url: return {}
    result = await asyncio.wait_for(downloader._try_one_url(paper_url), timeout=260)
    skeleton = extract_skeleton(result)
    if skeleton.get("sections"): return skeleton
    return {}


def openreview_pdf_url(record: dict) -> str:
    forum_id = record.get("openreview_forum_id", "")
    if not forum_id: return ""
    return f"https://openreview.net/pdf?id={forum_id}"


async def process_one(path: Path, downloader: PaperDownload) -> tuple[str, str, str]:
    output_path = OUT_DIR / path.name
    if output_path.exists():
        return path.name, "skip", "already downloaded"

    record = read_json(path)
    for arxiv_id in find_arxiv_ids(record):
        try:
            paper = await parse_from_arxiv(downloader, arxiv_id)
        except Exception as exc:
            print(f"{path.name} arxiv {arxiv_id} failed: {exc}")
            continue
        if paper:
            write_json(output_path, {**record, "source": "arxiv", "paper": paper})
            return path.name, "arxiv", arxiv_id

    paper_url = record.get("paper_url", "")
    # try:
    #     paper = await parse_from_pdf(downloader, paper_url)
    # except Exception as exc:
    #     print(f"{path.name} pdf failed: {exc}")
    #     paper = {}
    # if paper:
    #     write_json(output_path, {**record, "source": "pdf", "paper": paper})
    #     return path.name, "pdf", paper_url

    fallback_url = openreview_pdf_url(record)
    if fallback_url and fallback_url != paper_url:
        try:
            paper = await parse_from_pdf(downloader, fallback_url)
        except Exception as exc:
            print(f"{path.name} openreview pdf failed: {exc}")
            paper = {}
        if paper:
            write_json(output_path, {**record, "source": "openreview_pdf", "paper": paper})
            return path.name, "openreview_pdf", fallback_url

    return path.name, "failed", fallback_url or paper_url


async def main_async():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = read_json(ERROR_PATH) if ERROR_PATH.exists() else {}
    downloader = PaperDownload(ToolConfig())

    await SessionManager.init()
    try:
        tasks = [asyncio.create_task(process_one(path, downloader)) for path in iter_input_files()]
        for task in asyncio.as_completed(tasks):
            name, source, detail = await task
            if source != "skip": print(f"{source}: {name} {detail}")
            if source == "failed":
                errors[name] = detail
                write_json(ERROR_PATH, errors)
    finally:
        await SessionManager.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
