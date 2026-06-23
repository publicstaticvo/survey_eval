import argparse
import asyncio
import contextlib
import io
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import aiohttp
from tqdm import tqdm

from survey_eval.agent.tools.utility.openalex import get_openalex_client
from survey_eval.agent.tools.utility.paper_download import (
    PaperDownload,
    download_bytes_to_memory,
    extract_arxiv_id_from_url,
)
from survey_eval.agent.tools.utility.request_utils import RateLimit, SessionManager
from survey_eval.agent.tools.utility.s2 import get_semantic_scholar_client
from survey_eval.agent.tools.utility.tool_config import ToolConfig


DATA_DIR = Path("data")
OUT_DIR = Path("pdf_content")
SOURCE_DIR = Path("papers")
LOCAL_DIR = Path(".")
PDF_TIMEOUT = 1200
OPENALEX_ARXIV_SELECT = (
    "id,title,doi,ids,best_oa_location,locations,publication_date,created_date,"
    "cited_by_count,counts_by_year,abstract_inverted_index,authorships"
)
MAX_FOLDER_NAME_LEN = 160
OPENREVIEW_ID = "dailyyulun@163.com"
OPENREVIEW_PASSWORD = "Indwyx_0904"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_input_files() -> list[Path]:
    return sorted(path for path in DATA_DIR.glob("*.json") if not path.name.startswith("_"))


def safe_folder_name(title: str, fallback: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", title or "")
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = fallback
    return name[:MAX_FOLDER_NAME_LEN].rstrip(" .")


def extract_skeleton(result: dict) -> dict:
    if not result: return {}
    paper = result.get("result", result)
    if not isinstance(paper, dict): return {}
    skeleton = paper.get("full_content") or paper.get("paper") or {}
    return skeleton if isinstance(skeleton, dict) else {}


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def title_matches_record(paper: dict, record: dict) -> bool:
    paper_title = normalize_title(str(paper.get("title", "")))
    record_title = normalize_title(str(record.get("paper_title", "")))
    return bool(paper_title and record_title and paper_title == record_title)


def add_text_environment_type(paper: dict) -> dict:
    def visit_section(section: dict) -> None:
        for paragraph in section.get("paragraphs") or []:
            if not isinstance(paragraph, list):
                continue
            for sentence in paragraph:
                if isinstance(sentence, dict):
                    sentence["environment_type"] = "text"
        for child in section.get("sections") or []:
            if isinstance(child, dict):
                visit_section(child)

    if not isinstance(paper, dict):
        return paper
    abstract = paper.get("abstract")
    if isinstance(abstract, dict):
        visit_section(abstract)
    for paragraph in paper.get("paragraphs") or []:
        if not isinstance(paragraph, list):
            continue
        for sentence in paragraph:
            if isinstance(sentence, dict):
                sentence["environment_type"] = "text"
    for section in paper.get("sections") or []:
        if isinstance(section, dict):
            visit_section(section)
    return paper


def pdf_candidate_urls(record: dict) -> list[str]:
    urls = []
    # paper_url = str(record.get("paper_url") or "").strip()
    # if paper_url:
    #     urls.append(paper_url)
    review_page_url = str(record.get("review_page_url") or "").strip()
    if review_page_url:
        urls.append(review_page_url.replace("/forum?", "/pdf?"))
    return list(dict.fromkeys(urls))


def normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value


def extract_doi(paper: dict | None) -> str:
    paper = paper or {}
    doi = paper.get("doi") or ""
    if not doi:
        external_ids = paper.get("external_ids") or paper.get("externalIds") or {}
        doi = external_ids.get("DOI") or external_ids.get("doi") or ""
    return normalize_doi(str(doi))


def extract_arxiv_ids_from_locations(work: dict | None) -> list[str]:
    work = work or {}
    ids = []
    locations = []
    if work.get("best_oa_location"):
        locations.append(work["best_oa_location"])
    locations.extend(work.get("locations") or [])
    for location in locations:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            arxiv_id = extract_arxiv_id_from_url(str(location.get(key) or ""))
            if arxiv_id:
                ids.append(arxiv_id)
    return list(dict.fromkeys(ids))


async def find_arxiv_ids(record: dict, downloader: PaperDownload) -> list[str]:
    title = str(record.get("paper_title", "")).strip()
    if not title:
        return []

    openalex_work = None
    try:
        openalex = get_openalex_client(downloader.openalex.config if downloader.openalex else ToolConfig())
        openalex_work = await openalex.find_work_by_title(title, select=OPENALEX_ARXIV_SELECT)
    except Exception as exc:
        print(f"{title} OpenAlex lookup failed: {exc}")
    arxiv_ids = extract_arxiv_ids_from_locations(openalex_work)
    if arxiv_ids: return arxiv_ids

    openalex_doi = extract_doi(openalex_work)
    try:
        s2 = get_semantic_scholar_client(downloader.openalex.config if downloader.openalex else ToolConfig())
        s2_work = await s2.find_work_by_title(title)
    except Exception as exc:
        print(f"{title} Semantic Scholar lookup failed: {exc}")
        return []
    s2_doi = extract_doi(s2_work)
    if openalex_doi and s2_doi and openalex_doi != s2_doi:
        print(f"{title} Semantic Scholar DOI mismatch: {s2_doi} != {openalex_doi}")
        return []
    if openalex_doi and not s2_doi:
        print(f"{title} Semantic Scholar result has no DOI to match OpenAlex DOI {openalex_doi}")
        return []

    external_ids = (s2_work or {}).get("external_ids") or (s2_work or {}).get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv") or external_ids.get("ARXIV")
    return [str(arxiv_id).strip()] if arxiv_id else []


async def parse_from_arxiv(downloader: PaperDownload, arxiv_id: str) -> dict:
    result = await asyncio.wait_for(downloader._try_arxiv_source(arxiv_id), timeout=260)
    skeleton = extract_skeleton(result)
    if skeleton.get("sections"): return skeleton
    return {}


async def login_openreview() -> None:
    if not OPENREVIEW_ID or not OPENREVIEW_PASSWORD:
        print("OpenReview credentials are empty; PDF downloads may fail for login-required papers")
        return
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"id": OPENREVIEW_ID, "password": OPENREVIEW_PASSWORD}
    async with SessionManager.get().post(
        "https://api2.openreview.net/login",
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"OpenReview login failed: {resp.status} {text[:200]}")
        try:
            token = (json.loads(text) or {}).get("token", "")
        except json.JSONDecodeError:
            token = ""
        if token:
            session = SessionManager.get()
            session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"})


async def download_pdf_to_memory(url: str, timeout: int = PDF_TIMEOUT, proxy: str | None = None):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with RateLimit.DOWNLOAD_SEMAPHORE:
        async with SessionManager.get().get(
            url,
            headers=headers,
            allow_redirects=True,
            proxy=proxy,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            content = await resp.read()
            if not content:
                raise RuntimeError("empty response")
            return content


async def parse_pdf_buffer(downloader: PaperDownload, buffer: bytes) -> dict:
    if buffer[:5] != b"%PDF-":
        raise RuntimeError("downloaded content is not a PDF")
    await asyncio.sleep(0)
    pdf_buffer = io.BytesIO(buffer)
    pdf_buffer.seek(0)
    data = aiohttp.FormData()
    data.add_field("input", pdf_buffer.read(), filename="paper.pdf", content_type="application/pdf")
    async with RateLimit.PARSE_SEMAPHORE:
        async with SessionManager.get().post(
            f"{downloader.grobid}/api/processFulltextDocument",
            data=data,
            timeout=aiohttp.ClientTimeout(total=PDF_TIMEOUT),
        ) as resp:
            resp.raise_for_status()
            xml_content = await resp.text()
    if not xml_content:
        raise RuntimeError("GROBID returned no XML content")
    with contextlib.redirect_stdout(io.StringIO()):
        parsed = downloader._post_hook(xml_content)
    skeleton = extract_skeleton(parsed)
    if not skeleton.get("sections"):
        raise RuntimeError("GROBID parsed no sections")
    return add_text_environment_type(skeleton)


async def parse_pdf_url(downloader: PaperDownload, url: str) -> dict:
    parsed = urlparse(url or "")
    proxy = downloader.arxiv_proxy_url 
    # if parsed.netloc.lower() in {"arxiv.org", "www.arxiv.org"} else None
    buffer = await download_pdf_to_memory(url, timeout=PDF_TIMEOUT, proxy=proxy)
    return await parse_pdf_buffer(downloader, buffer)


async def parse_local_pdf(downloader: PaperDownload, input_path: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    return await parse_pdf_buffer(downloader, input_path.read_bytes())


async def download_arxiv_source(downloader: PaperDownload, arxiv_id: str, target_dir: Path) -> bool:
    src_url = f"https://arxiv.org/src/{arxiv_id}"
    buffer = await download_bytes_to_memory(src_url, proxy=downloader.arxiv_proxy_url)
    if not buffer:
        print(f"{src_url} No source buffer")
        return False
    if buffer[:5] == b"%PDF-":
        print(f"{src_url} returned PDF instead of source")
        return False

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source_", dir=str(target_dir.parent), ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        try:
            downloader._safe_extract_tar(buffer, tmp_dir)
        except tarfile.TarError:
            tex_path = tmp_dir / "source.tex"
            tex_path.write_bytes(buffer)
        if target_dir.exists():
            return True
        tmp_dir.rename(target_dir)
    return True


async def process_one_source(path: Path, downloader: PaperDownload) -> tuple[str, str, str]:
    record = read_json(path)
    title = record.get("paper_title", "")
    target_dir = SOURCE_DIR / safe_folder_name(title, path.stem)
    if target_dir.exists():
        return path.name, "skip", str(target_dir)

    arxiv_ids = await find_arxiv_ids(record, downloader)
    if not arxiv_ids:
        return path.name, "failed", "no arxiv id"

    for arxiv_id in arxiv_ids:
        try:
            if await asyncio.wait_for(download_arxiv_source(downloader, arxiv_id, target_dir), timeout=180):
                return path.name, "source", str(target_dir)
        except Exception as exc:
            print(f"{path.name} arxiv source {arxiv_id} failed: {exc}")
    return path.name, "failed", ", ".join(arxiv_ids)


async def process_one(path: Path, downloader: PaperDownload) -> tuple[str, str, str]:
    output_path = OUT_DIR / path.name
    if output_path.exists():
        return path.name, "skip", "already downloaded"

    record = read_json(path)
    arxiv_ids = await find_arxiv_ids(record, downloader)
    if not arxiv_ids:
        return path.name, "failed", "no arxiv id"

    for arxiv_id in arxiv_ids:
        try:
            paper = await parse_from_arxiv(downloader, arxiv_id)
        except Exception as exc:
            print(f"{path.name} arxiv {arxiv_id} failed: {exc}")
            continue
        if paper:
            add_text_environment_type(paper)
            if not title_matches_record(paper, record):
                print(
                    f"{path.name} arxiv {arxiv_id} title mismatch: "
                    f"{paper.get('title', '')} != {record.get('paper_title', '')}"
                )
                continue
            write_json(output_path, {**record, "source": "arxiv", "paper": paper})
            return path.name, "arxiv", arxiv_id

    return path.name, "failed", ", ".join(arxiv_ids)


def local_pdf_path(path: Path) -> Path:
    stem = path.name[:3]
    for suffix in (".pdf", ".PDF", ".json", ".JSON"):
        candidate = LOCAL_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return LOCAL_DIR / f"{stem}.pdf"


async def process_one_pdf(path: Path, downloader: PaperDownload, local: bool = False) -> tuple[str, str, str]:
    output_path = OUT_DIR / path.name
    if output_path.exists():
        return path.name, "skip", "already downloaded"

    record = read_json(path)
    if local:
        pdf_path = local_pdf_path(path)
        try:
            paper = await asyncio.wait_for(parse_local_pdf(downloader, pdf_path), timeout=PDF_TIMEOUT)
        except Exception as exc:
            print(f"{path.name} local PDF parse failed: {pdf_path} ({type(exc)} {exc})")
            return path.name, "failed", str(pdf_path)
        write_json(output_path, {**record, "source": "local_pdf", "paper": paper})
        return path.name, "local_pdf", str(pdf_path)

    urls = pdf_candidate_urls(record)
    if not urls:
        return path.name, "failed", "no pdf url"

    for url in urls:
        try:
            paper = await asyncio.wait_for(parse_pdf_url(downloader, url), timeout=PDF_TIMEOUT)
        except Exception as exc:
            print(f"{path.name} PDF download/parse failed: {url} ({type(exc)} {exc})")
            continue
        write_json(output_path, {**record, "source": "pdf", "paper": paper})
        return path.name, "pdf", url

    return path.name, "failed", ", ".join(urls)


async def main_async(source_only: bool = False, pdf_only: bool = False, local: bool = False):
    output_dir = SOURCE_DIR if source_only else OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    downloader = PaperDownload(ToolConfig())

    await SessionManager.init()
    try:
        if not source_only and not local:
            await login_openreview()
        if source_only:
            process = process_one_source
        elif pdf_only:
            process = lambda path, downloader: process_one_pdf(path, downloader, local=local)
        else:
            process = process_one
        tasks = [asyncio.create_task(process(path, downloader)) for path in iter_input_files()]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parsing papers"):
            name, source, detail = await task
            if source not in {"skip", "pdf"}: print(f"{source}: {name} {detail}")
    finally:
        await SessionManager.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="download and extract arXiv source files without parsing papers",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="download papers from paper_url/review_page_url PDFs and parse them with GROBID",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="in --pdf-only mode, parse LOCAL_DIR PDFs named by the first three characters of each input filename",
    )
    args = parser.parse_args()
    if args.source_only and args.pdf_only:
        parser.error("--source-only and --pdf-only cannot be used together")
    if args.local and not args.pdf_only:
        parser.error("--local requires --pdf-only")
    asyncio.run(main_async(source_only=args.source_only, pdf_only=args.pdf_only, local=args.local))


if __name__ == "__main__":
    main()
