import asyncio
import io
import random
from typing import Optional

import aiohttp
from tenacity import retry, retry_if_exception, retry_if_result, stop_after_attempt, wait_exponential

from .grobidpdf.paper_parser import PaperParser
from .openalex import get_openalex_client
from .request_utils import RateLimit, SessionManager

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

    async def download_single_paper(self, paper_meta: dict, openalex_id: str = "") -> Optional[dict]:
        """Try all candidate URLs and return the first successfully parsed paper."""
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
