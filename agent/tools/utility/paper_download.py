import asyncio
import io
import json
import random
import re, os
import traceback
import tarfile
import tempfile
import time
from typing import Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse
import sys

import aiohttp
from tenacity import retry, retry_if_exception, retry_if_result, stop_after_attempt, wait_exponential
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

if __package__:
    from .grobidpdf import PaperParser
    from .latex_parser import LatexPaperParser
    from .openalex import OPENALEX_SELECT, get_openalex_client
    from .request_utils import RateLimit, SessionManager
    from .tool_config import ToolConfig
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from survey_eval.agent.tools.utility.grobidpdf import PaperParser
    from survey_eval.agent.tools.utility.latex_parser import LatexPaperParser
    from survey_eval.agent.tools.utility.openalex import OPENALEX_SELECT, get_openalex_client
    from survey_eval.agent.tools.utility.request_utils import RateLimit, SessionManager
    from survey_eval.agent.tools.utility.tool_config import ToolConfig

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_result(lambda x: x is None),
)
def _browser_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }


async def download_paper_to_memory(url: str, timeout: int = 120, proxy: str | None = None):
    """Download a PDF into memory."""
    try:
        async with RateLimit.DOWNLOAD_SEMAPHORE:
            async with SessionManager.get().get(
                url,
                headers=_browser_headers(),
                allow_redirects=True,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                content = await resp.read()
                return io.BytesIO(content)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"download paper to memory from {url}: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_result(lambda x: x is None),
)
async def download_bytes_to_memory(url: str, timeout: int = 90, proxy: str | None = None) -> Optional[bytes]:
    try:
        async with RateLimit.LATEX_DOWNLOAD_SEMAPHORE:
            async with SessionManager.get().get(
                url,
                headers=_browser_headers(),
                allow_redirects=True,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                return await resp.read()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{url} No source buffer: {exc} {type(exc)}")


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


def is_direct_pdf_url(url: str) -> bool:
    parsed = urlparse(url or "")
    path = parsed.path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


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
            self.arxiv_proxy_url = grobid_url.arxiv_proxy_url
        else:
            self.grobid = grobid_url
            self.openalex = None
            self.arxiv_proxy_url = "http://localhost:7890"

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

    def _latex_post_hook(self, paper, latex_content: str = "") -> dict:
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
        print(f"Parsed TeX source. It has {len(skeleton['sections'])} sections.")
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
        if not tex_files: return          
        scored = []
        preferred_names = {"main.tex", "ms.tex", "paper.tex", "article.tex", "root.tex", "uq_survey.tex"}
        for path in tex_files:
            try:
                content = self._read_text_file(path)
            except Exception:
                continue
            has_document = "\\begin{document}" in content
            score = 0
            if has_document: score += 100
            if path.name.lower() in preferred_names: score += 30
            if "\\documentclass" in content: score += 20
            if "\\section" in content: score += min(content.count("\\section") * 5, 40)
            score += min(len(content) // 2000, 20)
            scored.append((score, path))
        if not scored:
            return tex_files[0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    async def _try_arxiv_source(self, arxiv_id: str) -> dict:
        src_url = f"https://arxiv.org/src/{arxiv_id}"
        try:
            buffer = await download_bytes_to_memory(src_url, proxy=self.arxiv_proxy_url)
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
                result = self._latex_post_hook(paper, latex_content)
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
            parsed = urlparse(url or "")
            proxy = self.arxiv_proxy_url if parsed.netloc.lower() in {"arxiv.org", "www.arxiv.org"} else None
            pdf_buffer = await download_paper_to_memory(url, proxy=proxy)
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
            if self.openalex is None:
                print("download_single_paper needs paper_meta when no OpenAlex client is configured")
                return
            if openalex_id:
                try:
                    paper_meta = await self.openalex.get_entity(
                        openalex_id,
                        select=f"{OPENALEX_SELECT},best_oa_location,locations",
                    ) or {}
                except Exception as exc:
                    print(f"download_single_paper OpenAlex id lookup failed for {openalex_id}: {exc}")
            if not paper_meta and title:
                try:
                    paper_meta = await self.openalex.find_work_by_title(
                        title,
                        select=f"{OPENALEX_SELECT},best_oa_location,locations",
                    ) or {}
                except Exception as exc:
                    print(f"download_single_paper OpenAlex title lookup failed for {title}: {exc}")
            if not paper_meta:
                return

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


class SemanticScholarPaperDownload(PaperDownload):
    """Download papers from Semantic Scholar style metadata."""

    def _semantic_scholar_urls(self, paper_meta: dict, excluded_urls: set[str] | None = None) -> list[str]:
        excluded_urls = excluded_urls or set()
        urls = []
        open_access_pdf = paper_meta.get("openAccessPdf") or paper_meta.get("open_access_pdf") or {}
        if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
            urls.append(open_access_pdf["url"])
        for key in ("url", "pdf_url", "pdfUrl"):
            if paper_meta.get(key):
                urls.append(paper_meta[key])
        external_ids = paper_meta.get("external_ids") or paper_meta.get("externalIds") or {}
        for key, value in external_ids.items():
            if not value:
                continue
            key_lower = str(key).lower()
            if key_lower == "arxiv":
                urls.append(f"https://arxiv.org/abs/{value}")
            elif key_lower == "doi":
                doi = str(value)
                urls.append(doi if doi.startswith("http") else f"https://doi.org/{doi}")
        return [
            url
            for url in dict.fromkeys(str(url).strip() for url in urls if str(url).strip())
            if url not in excluded_urls
        ]

    async def _find_pdf_urls_from_page(self, url: str) -> list[str]:
        try:
            async with RateLimit.DOWNLOAD_SEMAPHORE:
                async with SessionManager.get().get(
                    url,
                    allow_redirects=True,
                    timeout=aiohttp.ClientTimeout(total=45),
                ) as resp:
                    resp.raise_for_status()
                    final_url = str(resp.url)
                    content_type = resp.headers.get("content-type", "").lower()
                    if "application/pdf" in content_type:
                        return [final_url]
                    text = await resp.text(errors="ignore")
        except Exception as exc:
            print(f"{url} page pdf discovery failed: {exc}")
            return []

        if BeautifulSoup is None:
            return []
        soup = BeautifulSoup(text, "html.parser")
        pdf_urls = []
        patterns = [
            {"class": "obj_galley_link pdf"},
            {"class": "document-access-icon-pdf"},
        ]
        for attrs in patterns:
            for element in soup.find_all("a", attrs=attrs):
                href = element.get("href")
                if href:
                    pdf_urls.append(urljoin(final_url, href))
        for element in soup.find_all("a", string=re.compile(r"PDF", re.I)):
            href = element.get("href")
            if href:
                pdf_urls.append(urljoin(final_url, href))
        for element in soup.find_all("a", href=re.compile(r"\.pdf(?:$|[?#])", re.I)):
            href = element.get("href")
            if href:
                pdf_urls.append(urljoin(final_url, href))
        return list(dict.fromkeys(pdf_urls))

    def _find_pdf_urls_with_selenium_sync(self, url: str) -> list[str]:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(120)
            driver.set_script_timeout(120)
            driver.get(url)
            anchors = driver.find_elements(By.TAG_NAME, "a")
            urls = []
            for anchor in anchors:
                href = anchor.get_attribute("href") or ""
                text = anchor.text or ""
                if re.search(r"\.pdf(?:$|[?#])", href, re.I) or re.search(r"\bPDF\b", text, re.I):
                    urls.append(urljoin(driver.current_url, href))
            return list(dict.fromkeys(urls))
        except Exception as exc:
            print(f"{url} selenium pdf discovery failed: {exc}")
            return []
        finally:
            if driver is not None:
                driver.quit()

    async def _find_pdf_urls_with_selenium(self, url: str) -> list[str]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._find_pdf_urls_with_selenium_sync, url),
                timeout=150,
            )
        except asyncio.TimeoutError:
            print(f"{url} selenium pdf discovery timeout")
            return []

    def _download_pdf_with_selenium_sync(self, url: str) -> bytes | None:
        tmp_parent = Path("C:/tmp") if Path("C:/tmp").exists() else None
        with tempfile.TemporaryDirectory(
            prefix="selenium_pdf_",
            dir=str(tmp_parent) if tmp_parent else None,
            ignore_cleanup_errors=True,
        ) as tmp:
            download_dir = Path(tmp)
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option(
                "prefs",
                {
                    "download.default_directory": str(download_dir),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "plugins.always_open_pdf_externally": True,
                },
            )
            driver = None
            try:
                driver = webdriver.Chrome(options=options)
                driver.set_page_load_timeout(120)
                driver.set_script_timeout(120)
                driver.execute_cdp_cmd(
                    "Page.setDownloadBehavior",
                    {"behavior": "allow", "downloadPath": str(download_dir)},
                )
                driver.get(url)
                deadline = time.monotonic() + 120
                last_size = -1
                stable_count = 0
                while time.monotonic() < deadline:
                    files = [
                        path
                        for path in download_dir.iterdir()
                        if path.is_file() and not path.name.endswith(".crdownload")
                    ]
                    pdf_files = [path for path in files if path.suffix.lower() == ".pdf"]
                    candidates = pdf_files or files
                    if candidates:
                        target = max(candidates, key=lambda path: path.stat().st_mtime)
                        size = target.stat().st_size
                        if size > 0 and size == last_size:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_size = size
                        if stable_count >= 2:
                            content = target.read_bytes()
                            if content[:5] == b"%PDF-":
                                return content
                    time.sleep(1)
                return None
            except Exception as exc:
                print(f"{url} selenium pdf download failed: {exc}")
                return None
            finally:
                if driver is not None:
                    driver.quit()

    async def _download_pdf_with_selenium(self, url: str) -> io.BytesIO | None:
        try:
            content = await asyncio.wait_for(
                asyncio.to_thread(self._download_pdf_with_selenium_sync, url),
                timeout=150,
            )
        except asyncio.TimeoutError:
            print(f"{url} selenium pdf download timeout")
            return None
        return io.BytesIO(content) if content else None

    async def _try_semantic_url(self, url: str) -> dict:
        arxiv_id = extract_arxiv_id_from_url(url)
        if arxiv_id:
            source_result = await self._try_arxiv_source(arxiv_id)
            if source_result.get("result"):
                return source_result
            return await self._try_one_url(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        if is_direct_pdf_url(url):
            result = await self._try_one_url(url)
            if result.get("result"):
                return result
            pdf_buffer = await self._download_pdf_with_selenium(url)
            if not pdf_buffer:
                return result
            xml_content = await parse_with_grobid(self.grobid, pdf_buffer)
            if not xml_content:
                return {"result": None, "download_error": False, "parse_error": True}
            return {"result": self._post_hook(xml_content), "download_error": False, "parse_error": False}

        pdf_urls = await self._find_pdf_urls_from_page(url)
        if not pdf_urls:
            pdf_urls = await self._find_pdf_urls_with_selenium(url)
        tasks = [asyncio.create_task(self._try_one_url(pdf_url)) for pdf_url in pdf_urls]
        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result.get("result"):
                    for other_task in tasks:
                        if not other_task.done():
                            other_task.cancel()
                    return result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {"result": None, "download_error": True, "parse_error": False}

    async def download_single_paper(
        self,
        paper_meta: dict | None = None,
        openalex_id: str = "",
        title: str = "",
        excluded_urls: set[str] | None = None,
    ) -> Optional[dict]:
        """Try Semantic Scholar links and return the first successfully parsed paper."""
        paper_meta = dict(paper_meta or {})
        if not paper_meta:
            return await super().download_single_paper(
                paper_meta=paper_meta,
                openalex_id=openalex_id,
                title=title,
            )
        urls = self._semantic_scholar_urls(paper_meta, excluded_urls=excluded_urls)
        if not urls: return
        if len(urls) == 1:
            result = await self._try_semantic_url(urls[0])
            return result.get("result")

        tasks = [asyncio.create_task(self._try_semantic_url(url)) for url in urls]
        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result.get("result"):
                    for other_task in tasks:
                        if not other_task.done():
                            other_task.cancel()
                    return result["result"]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


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


async def test_arxiv_latex_source(arxiv_id: str, output_target: str | None = None) -> dict:
    """Test the arXiv TeX parsing path and optionally save the parsed skeleton."""
    await SessionManager.init()
    try:
        downloader = PaperDownload("")
        result = await asyncio.wait_for(downloader._try_arxiv_source(arxiv_id), timeout=240)
        parsed = bool(result.get("result"))
        print(f"arxiv_id={arxiv_id} parsed={parsed}")
        if not parsed:
            raise RuntimeError(f"Failed to download or parse arXiv source {arxiv_id}")

        skeleton = result["result"].get("full_content", {})
        print(f"title={skeleton.get('title')}")
        print(f"sections={len(skeleton.get('sections', []))}")
        if output_target:
            output_path = Path(output_target)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved skeleton to {output_path}")
        return result["result"]
    finally:
        await SessionManager.close()


async def test_pdf_url_download(
    url: str,
    output_target: str | None = None,
    config: ToolConfig | str | None = None,
) -> dict:
    """Test downloading a PDF from a specific URL and parsing it into a paper skeleton."""
    await SessionManager.init()
    try:
        downloader = SemanticScholarPaperDownload(config or ToolConfig())
        result = await downloader._try_semantic_url(url)
        parsed = bool(result.get("result"))
        print(f"url={url} parsed={parsed}")
        if not parsed:
            raise RuntimeError(f"Failed to download or parse PDF URL {url}")

        skeleton = result["result"].get("full_content", {})
        print(f"title={skeleton.get('title')}")
        print(f"sections={len(skeleton.get('sections', []))}")
        if output_target:
            output_path = Path(output_target)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved skeleton to {output_path}")
        return result["result"]
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    work_ids = ["2407.01878", "2205.11916", '2306.04459']
    asyncio.run(_debug_arxiv_source(work_ids[1]))
