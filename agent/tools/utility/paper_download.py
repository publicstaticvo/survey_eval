import asyncio
import io
import random
import re
import traceback
import tarfile
import tempfile
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse
import sys

import aiohttp
from tenacity import retry, retry_if_exception, retry_if_result, stop_after_attempt, wait_exponential

if __package__:
    from .grobidpdf import PaperParser
    from .latex_parser import LatexPaperParser
    from .openalex import OPENALEX_SELECT, get_openalex_client
    from .request_utils import RateLimit, SessionManager
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from survey_eval.agent.tools.utility.grobidpdf import PaperParser
    from survey_eval.agent.tools.utility.latex_parser import LatexPaperParser
    from survey_eval.agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
    from survey_eval.agent.tools.utility.request_utils import RateLimit, SessionManager

parser = PaperParser()


def grobid_should_retry(exception: Exception) -> bool:
    if isinstance(exception, asyncio.TimeoutError):
        return True
    if isinstance(exception, aiohttp.ClientError):
        return True
    if isinstance(exception, aiohttp.ServerDisconnectedError):
        return True
    if isinstance(exception, aiohttp.ClientResponseError) and exception.status in [429, 503]:
        return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(grobid_should_retry),
    reraise=True,
)
async def parse_with_grobid(grobid_url, pdf_buffer: io.BytesIO) -> Optional[str]:
    """Use GROBID to parse a PDF buffer into XML."""
    url = f"{grobid_url}/api/processFulltextDocument"
    try:
        await asyncio.sleep(2 * random.random())
        pdf_buffer.seek(0)

        data = aiohttp.FormData()
        data.add_field("input", pdf_buffer.read(), filename="paper.pdf", content_type="application/pdf")

        async with RateLimit.PARSE_SEMAPHORE:
            async with SessionManager.get().post(url, data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                return await resp.text()

    except KeyboardInterrupt:
        raise
    except asyncio.TimeoutError:
        print("GROBID timeout, will retry")
        raise
    except aiohttp.ClientError as e:
        print(f"GROBID client error: {e}, will retry")
        raise
    except Exception as e:
        print(f"GROBID unexpected error: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_result(lambda x: x is None),
)
async def download_paper_to_memory(url: str, timeout: int = 600):
    """Download a PDF into memory."""
    try:
        async with RateLimit.DOWNLOAD_SEMAPHORE:
            async with SessionManager.get().get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                content = await resp.read()
                return io.BytesIO(content)
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_result(lambda x: x is None),
)
async def download_bytes_to_memory(url: str, timeout: int = 180) -> Optional[bytes]:
    try:
        async with RateLimit.DOWNLOAD_SEMAPHORE:
            async with SessionManager.get().get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                return await resp.read()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{url} No source buffer: {exc}")
        return None


def yield_location(x):
    urls = set()
    best = x.get("best_oa_location")
    if best and best.get("pdf_url"):
        urls.add(best["pdf_url"])
        yield best["pdf_url"]
    for item in x.get("locations", []):
        if item.get("pdf_url") and item["pdf_url"] not in urls:
            urls.add(item["pdf_url"])
            yield item["pdf_url"]


def extract_arxiv_id_from_url(url: str) -> str:
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    if host not in {"arxiv.org", "www.arxiv.org"}:
        return ""
    match = re.search(r"/(?:abs|pdf|html|src)/([^/?#]+)", parsed.path)
    if not match:
        return ""
    arxiv_id = match.group(1).removesuffix(".pdf")
    return arxiv_id.strip()


def extract_arxiv_ids(paper_meta: dict) -> list[str]:
    ids = []
    for url in yield_location(paper_meta):
        arxiv_id = extract_arxiv_id_from_url(url)
        if arxiv_id:
            ids.append(arxiv_id)
    for key in ("doi", "arxiv_id"):
        value = paper_meta.get(key)
        if isinstance(value, str) and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", value.strip()):
            ids.append(value.strip())
    external_ids = paper_meta.get("external_ids") or paper_meta.get("externalIds") or {}
    for key, value in external_ids.items():
        if str(key).lower() == "arxiv" and value:
            ids.append(str(value).strip())
    return list(dict.fromkeys(ids))


class PaperDownload:
    def __init__(self, grobid_url):
        self.paper_parser = PaperParser()
        if hasattr(grobid_url, "grobid_url"):
            self.grobid = grobid_url.grobid_url
            self.openalex = get_openalex_client(grobid_url)
        else:
            self.grobid = grobid_url
            self.openalex = None

    def _post_hook(self, xml_content: str) -> dict:
        try:
            paper = self.paper_parser.parse(xml_content)
            if not paper:
                print("No paper.")
                return {}
            print(f"Parsed papers. It has {len(paper.children)} sections.")
        except Exception as e:
            print(f"No paper: {e}")
            raise
        abstract = "\n\n".join(" ".join(s.text for s in p.sentences) for p in paper.abstract.paragraphs) if paper.abstract else None
        return {"full_content": paper.get_skeleton(), "abstract": abstract}

    def _latex_post_hook(self, paper) -> dict:
        if not paper:
            return {}
        abstract = None
        if paper.abstract:
            abstract = "\n\n".join(
                " ".join(getattr(sentence, "text", "") for sentence in paragraph.sentences)
                for paragraph in paper.abstract.children
                if hasattr(paragraph, "sentences")
            )
        skeleton = paper.get_skeleton()
        print(f"Parsed TeX source. It has {len(skeleton.get('sections', []))} sections.")
        return {"full_content": skeleton, "abstract": abstract}

    def _safe_extract_tar(self, buffer: bytes, target_dir: Path) -> None:
        with tarfile.open(fileobj=io.BytesIO(buffer), mode="r:gz") as archive:
            target_root = target_dir.resolve()
            for member in archive.getmembers():
                member_path = (target_root / member.name).resolve()
                if target_root not in member_path.parents and member_path != target_root:
                    raise RuntimeError(f"Unsafe tar member path: {member.name}")
            archive.extractall(target_root)

    def _read_text_file(self, path: Path) -> str:
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    def _find_main_tex(self, source_dir: Path) -> Path | None:
        tex_files = [path for path in source_dir.rglob("*.tex") if path.is_file()]
        if not tex_files:
            return None

        scored = []
        preferred_names = {"main.tex", "ms.tex", "paper.tex", "article.tex", "root.tex", "uq_survey.tex"}
        for path in tex_files:
            try:
                content = self._read_text_file(path)
            except Exception:
                continue
            has_document = "\\begin{document}" in content
            score = 0
            if has_document:
                score += 100
            if path.name.lower() in preferred_names:
                score += 30
            if "\\documentclass" in content:
                score += 20
            if "\\section" in content:
                score += min(content.count("\\section") * 5, 40)
            score += min(len(content) // 2000, 20)
            scored.append((score, path))
        if not scored:
            return tex_files[0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    async def _try_arxiv_source(self, arxiv_id: str) -> dict:
        src_url = f"https://arxiv.org/src/{arxiv_id}"
        try:
            buffer = await download_bytes_to_memory(src_url)
            if not buffer:
                return {"result": None, "download_error": True, "parse_error": False}
            print(f"Downloaded TeX source from {src_url}")
            tmp_parent = Path("C:/tmp") if Path("C:/tmp").exists() else None
            with tempfile.TemporaryDirectory(
                prefix=f"arxiv_{re.sub(r'[^0-9A-Za-z]+', '_', arxiv_id)}_",
                dir=str(tmp_parent) if tmp_parent else None,
                ignore_cleanup_errors=True,
            ) as tmp:
                source_dir = Path(tmp)
                try:
                    self._safe_extract_tar(buffer, source_dir)
                except tarfile.TarError:
                    if buffer[:5] == b"%PDF-":
                        print(f"{src_url} returned PDF instead of TeX source")
                        return {"result": None, "download_error": False, "parse_error": True}
                    tex_path = source_dir / "source.tex"
                    tex_path.write_bytes(buffer)
                tex_count = len([path for path in source_dir.rglob("*.tex") if path.is_file()])
                main_tex = self._find_main_tex(source_dir)
                if not main_tex:
                    print(f"{src_url} No TeX file")
                    return {"result": None, "download_error": False, "parse_error": True}
                print(f"Parsing TeX source main file {main_tex.name} from {tex_count} TeX files")
                latex_content = self._read_text_file(main_tex)
                parser = LatexPaperParser(latex_content, base_path=str(main_tex.parent))
                paper = parser.parse()
                result = self._latex_post_hook(paper)
                if result:
                    return {"result": result, "download_error": False, "parse_error": False}
                return {"result": None, "download_error": False, "parse_error": True}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"arXiv source failed for {arxiv_id}: {exc}")
            trace_lines = traceback.format_exc().strip().splitlines()
            print("\n".join(trace_lines[-8:]))
            return {"result": None, "download_error": False, "parse_error": True}

    async def _try_one_url(self, url: str) -> dict:
        """Try downloading and parsing a paper from one URL."""
        try:
            pdf_buffer = await download_paper_to_memory(url)
            if not pdf_buffer:
                print(f"{url} No pdf buffer")
                return {"result": None, "download_error": True, "parse_error": False}
            print(f"Downloaded PDF from {url}")

            xml_content = await parse_with_grobid(self.grobid, pdf_buffer)
            if not xml_content:
                print(f"{url} No xml content")
                return {"result": None, "download_error": False, "parse_error": True}
            print(f"Parsed from {url}")
            return {"result": self._post_hook(xml_content), "download_error": False, "parse_error": False}

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"URL failed downloading from {url}: {e}")
            return {"result": None, "download_error": True, "parse_error": False}

    async def download_single_paper(
        self,
        paper_meta: dict | None = None,
        openalex_id: str = "",
        title: str = "",
    ) -> Optional[dict]:
        """Try all candidate URLs and return the first successfully parsed paper."""
        paper_meta = dict(paper_meta or {})
        if not paper_meta:
            if not title:
                raise ValueError("download_single_paper requires at least one of paper_meta or title")
            if self.openalex is None:
                return None
            paper_meta = await self.openalex.find_work_by_title(
                title,
                select=f"{OPENALEX_SELECT},best_oa_location,locations",
            ) or {}
            if not paper_meta:
                return None

        arxiv_ids = extract_arxiv_ids(paper_meta)
        for arxiv_id in arxiv_ids:
            result = await self._try_arxiv_source(arxiv_id)
            if result.get("result"):
                return result["result"]

        tasks = [asyncio.create_task(self._try_one_url(url)) for url in list(yield_location(paper_meta))]
        saw_download_error = False
        saw_parse_error = False

        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if result.get("download_error"):
                        saw_download_error = True
                    if result.get("parse_error"):
                        saw_parse_error = True
                    if result.get("result"):
                        for other_task in tasks:
                            if not other_task.done():
                                other_task.cancel()
                        return result["result"]
                except asyncio.CancelledError:
                    continue
                except Exception as e:
                    print(f"URL failed at download_single_paper: {e}")
                    continue
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        work_id = openalex_id or paper_meta.get("id", "")
        if self.openalex is not None and work_id and saw_download_error and not saw_parse_error:
            try:
                xml_content = await self.openalex.download_work_content(work_id, download_type="grobid_xml")
                return self._post_hook(xml_content)
            except Exception as e:
                print(f"OpenAlex content fallback failed for {work_id}: {e}")

        return None


async def _debug_arxiv_source(arxiv_id: str = "2512.15567"):
    await SessionManager.init()
    try:
        downloader = PaperDownload("")
        result = await asyncio.wait_for(downloader._try_arxiv_source(arxiv_id), timeout=240)
        parsed = bool(result.get("result"))
        print(f"arxiv_id={arxiv_id} parsed={parsed}")
        if parsed:
            skeleton = result["result"]["full_content"]
            print(f"title={skeleton.get('title')}")
            print(f"sections={len(skeleton.get('sections', []))}")
            if not skeleton.get("sections"):
                raise RuntimeError("Parsed TeX source but found no sections")
        else:
            raise RuntimeError(f"Failed to download or parse arXiv source {arxiv_id}")
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    work_ids = ["2308.04268", "2302.13425", '2306.04459']
    asyncio.run(_debug_arxiv_source(work_ids[1]))
