import io
import json
import random
import asyncio, aiohttp, aiofiles
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
    retry_if_result,              # 返回None的时候也要重试
)
from typing import Optional
from abc import ABC, abstractmethod

from .tool_config import LLMServerInfo
from .paper_parser import PaperParser

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}
email_pool = [
    "dailyyulun@gmail.com",
    "fqpcvtjj@hotmail.com",
    "ts.yu@siat.ac.cn",
    "yutianshu.yts@alibaba-inc.com",
    "yts17@mails.tsinghua.edu.cn"
    "yutianshu2025@ia.ac.cn",
    "yutianshu25@ucas.ac.cn",
    "dailyyulun@163.com",
    "lundufiles@163.com",
    "lundufiles123@163.com"
]
parser = PaperParser()
RETRY_EXCEPTION_TYPES = [
    aiohttp.ClientError, 
    asyncio.TimeoutError, 
    aiohttp.ServerDisconnectedError, 
    json.JSONDecodeError,
    AssertionError,
    KeyError,
]
URL_DOMAIN = "https://openalex.org/"
GROBID_URL = "https://localhost:8070"


# =============== Global Semaphore ===============
class RateLimit:
    OPENALEX_SEMAPHORE = asyncio.Semaphore(20)              # 搜索 API
    AGENT_SEMAPHORE = asyncio.Semaphore(100)                # LLM
    SBERT_SEMAPHORE = asyncio.Semaphore(50)                 # LLM
    PARSE_SEMAPHORE = asyncio.Semaphore(50)                 # GROBID docker镜像本地解析


class SessionManager:
    _global_session: Optional[aiohttp.ClientSession] = None
    
    @classmethod
    async def init(cls):
        """进入上下文时调用"""
        if cls._global_session is None:
            connector = aiohttp.TCPConnector(limit=250, limit_per_host=100, ttl_dns_cache=300)
            cls._global_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60)
            )
    
    @classmethod
    async def close(cls):
        """退出上下文时调用"""
        if cls._global_session and not cls._global_session.closed:
            await cls._global_session.close()
            cls._global_session = None
    
    @classmethod
    def get(cls) -> aiohttp.ClientSession:
        """获取全局 session"""
        if cls._global_session is None:
            raise RuntimeError("SessionManager not initialized")
        return cls._global_session


# =============== Openalex ===============
def openalex_should_retry(exception: BaseException) -> bool:
    if any(isinstance(exception, x) for x in RETRY_EXCEPTION_TYPES): return True
    if isinstance(exception, aiohttp.ClientResponseError) and exception.status not in [400, 401, 403, 404]: return True
    return False


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, exp_base=2, min=1, max=30),
    retry=retry_if_exception(openalex_should_retry),
    reraise=True
)
async def async_request_template(
    method: str,
    url: str,
    headers: dict = None,
    parameters: dict = None
) -> dict:
    """使用全局 session"""
    session = SessionManager.get()
    
    headers = headers or {}
    parameters = parameters or {}
    
    if method.lower() == "post":
        headers.setdefault("Content-Type", "application/json")
        async with session.post(url, headers=headers, json=parameters) as resp:
            resp.raise_for_status()
            return await resp.json()
    else:
        async with session.get(url, headers=headers, params=parameters) as resp:
            resp.raise_for_status()
            return await resp.json()


async def openalex_search_paper(
        endpoint: str,
        filter: dict = None,
        do_sample: bool = False,
        per_page: int = 1,
        add_email: bool | str = True,
        **request_kwargs
    ) -> dict:
    """使用 async_request_template，间接使用全局 session"""
    assert per_page <= 200, "Per page is at most 200"
    # 整理参数
    url = f"https://api.openalex.org/{endpoint}"
    if filter:
        # filter
        filter_string = ",".join([f"{k}:{v}" for k, v in filter.items()])
        request_kwargs["filter"] = filter_string
    if do_sample:
        # use per_page as num_samples
        request_kwargs['sample'] = per_page
        request_kwargs['seed'] = random.randint(0, 32767)        
    if add_email:
        request_kwargs['mailto'] = add_email if isinstance(add_email, str) else random.choice(email_pool)
    if per_page > 25: 
        request_kwargs['per-page'] = per_page
    # Go!
    return await async_request_template("get", url, headers, request_kwargs)


# =============== LLM client ===============
def llm_should_retry(exception: BaseException) -> bool:
    if isinstance(exception, KeyboardInterrupt): return False
    if isinstance(exception, NotImplementedError): return False
    return True


class AsyncLLMClient(ABC):

    def __init__(
            self, 
            llm: LLMServerInfo, 
            sampling_params: dict = {}, 
        ):
        self.llm = llm
        self.sampling_params = sampling_params
        
    @abstractmethod
    def _availability(self, response):
        raise NotImplementedError

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception(llm_should_retry) | retry_if_result(bool)
    )
    async def call(self, endpoint: str = "chat/completions", **kwargs) -> dict | None:
        self._context = kwargs.get("context", None)
        payload = {"model": self.llm.model}

        if endpoint == "chat/completions":
            payload.update(self.sampling_params)
            if "messages" not in kwargs:
                raise ValueError("Must have messages for chat/completions")
            messages = kwargs['messages']
            if isinstance(messages, str):
                messages = [{'role': 'user', "content": messages}]
            payload['messages'] = messages

        payload.update({k: v for k, v in kwargs.items() if k not in ["context", 'messages']})  

        url = f"{self.llm.base_url.rstrip('/')}/v1/{endpoint}"
        data = await async_request_template("post", url, headers, payload)
        # content = data["choices"][0]["message"]["content"]
        return self._availability(data)
    

# =============== Grobid ===============
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
async def parse_with_grobid(session: aiohttp.ClientSession, pdf_buffer: io.BytesIO) -> Optional[str]:
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
        
        async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
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
async def download_paper_to_memory(session: aiohttp.ClientSession, url: str, timeout: int = 600):
    """下载 PDF 文件"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            content = await resp.read()
            return io.BytesIO(content)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # print(f"Download failed {url}: {e}")
        return None
    

async def try_one_url(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    """尝试从单个 URL 下载并解析论文"""
    async with RateLimit.PARSE_SEMAPHORE:  # 限流
        try:
            # 步骤1: 下载 PDF
            pdf_buffer = await download_paper_to_memory(session, url)
            if not pdf_buffer:
                return None
            
            # 步骤2: 通过 GROBID 解析
            xml_text = await parse_with_grobid(session, pdf_buffer)
            if not xml_text:
                return None
            
            # 步骤3: 用 XMLPaperParser 解析 XML
            paper = await asyncio.to_thread(parser.parse, xml_text)

            # abstract
            abstract = " ".join(p.text for p in paper.abstract.paragraphs) if paper.abstract else "None"
            
            return {"title": paper.title, "abstract": abstract, "url": url, "structure": paper.get_skeleton()}
        
        except KeyboardInterrupt:
            raise
            
        except Exception as e:
            return None



