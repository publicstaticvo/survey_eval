import asyncio, aiohttp
import random
import io
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
    retry_if_result,              # 返回None的时候也要重试
)
from .paper_parser import PaperParser
from .request_utils import RateLimit, SessionManager

GROBID_URL = "https://localhost:8070"
parser = PaperParser()


def grobid_should_retry(exception: Exception) -> bool:
    if isinstance(exception, asyncio.TimeoutError): return True
    if isinstance(exception, aiohttp.ClientError): return True
    if isinstance(exception, aiohttp.ServerDisconnectedError): return True
    if isinstance(exception, aiohttp.ClientResponseError) and exception.status in [429, 503]: return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(grobid_should_retry),
    reraise=True
)
async def parse_with_grobid(pdf_buffer: io.BytesIO) -> Optional[str]:
    """通过 GROBID 解析 PDF（带重试）"""
    url = f"{GROBID_URL}/api/processFulltextDocument"
    try:
        # 添加随机延迟避免过载
        await asyncio.sleep(2 * random.random())
        
        # 重置 buffer 位置
        pdf_buffer.seek(0)
        
        # 构造 multipart/form-data
        data = aiohttp.FormData()
        data.add_field('input', pdf_buffer.read(), filename='paper.pdf', content_type='application/pdf')
        
        async with RateLimit.PARSE_SEMAPHORE:
            async with SessionManager.get().post(url, data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                return await resp.text()
            
    except KeyboardInterrupt:
        raise
    except asyncio.TimeoutError:
        # print("GROBID timeout, will retry")
        raise  # 让 tenacity 处理重试
    except aiohttp.ClientError as e:
        # print(f"GROBID client error: {e}, will retry")
        raise
    except Exception as e:
        # 其他错误不重试，直接返回 None
        print(f"GROBID unexpected error: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_result(lambda x: x is None)
)
async def download_paper_to_memory(url: str, timeout: int = 600):
    """下载 PDF 文件"""
    try:
        async with RateLimit.HTTP_SEMAPHORE:
            async with SessionManager.get().get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                content = await resp.read()
                return io.BytesIO(content)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # print(f"Download failed {url}: {e}")
        return None
    

def yield_location(x):
    urls = set()
    y = x["best_oa_location"]
    if y and y['pdf_url']: 
        urls.add(y['pdf_url'])
        yield y['pdf_url']
    for y in x['locations']:
        if y['pdf_url'] and y['pdf_url'] not in urls: 
            urls.add(y['pdf_url'])
            yield y['pdf_url']


class PaperDownload:    

    def __init__(self):
        self.paper_parser = PaperParser()

    def _post_hook(self, xml_content: str) -> dict:
        try:
            paper = self.paper_parser.parse(xml_content)
            if not paper:
                return {}
        except Exception as e:
            return {}
        abstract = "\n\n".join(" ".join(s.text for s in p.sentences) for p in paper.abstract.paragraphs) if paper.abstract else None
        return {"full_content": paper.get_skeleton(), "abstract": abstract}

    async def _try_one_url(self, url: str) -> Optional[str]:
        """尝试从单个 URL 下载并解析论文"""
        try:
            # 步骤1: 下载 PDF
            pdf_buffer = await download_paper_to_memory(url)
            if not pdf_buffer:
                return None
            
            # 步骤2: 通过 GROBID 解析
            xml_content = await parse_with_grobid(pdf_buffer)
            if not xml_content:
                return None
            
            return self._post_hook(xml_content)
        
        except KeyboardInterrupt:
            raise
            
        except Exception as e:
            return None
    

    async def download_single_paper(self, paper_meta: dict) -> Optional[str]:
        """处理单篇论文：尝试所有 URL，返回第一个成功的"""
        
        # 为该论文的所有 URL 创建任务
        tasks = [asyncio.create_task(self._try_one_url(url)) for url in list(yield_location(paper_meta))]
        
        # 使用 as_completed 获取第一个成功的结果
        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    if result:
                        # 取消其他任务
                        for other_task in tasks:
                            if not other_task.done():
                                other_task.cancel()                                
                        return result
                except asyncio.CancelledError:
                    # 某些 task 被 cancel 时不会视为错误，继续尝试其他 task
                    continue
                except Exception as e:
                    print(f"URL failed: {e}")
                    continue
        finally:
            # 确保所有任务都被清理
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
        return None  # 所有 URL 都失败
