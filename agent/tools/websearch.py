import asyncio, aiohttp
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
)
from trafilatura import extract

from .tool_config import ToolConfig
from .utils import extract_json
from .llmclient import AsyncChat
from .evidence_check import EvidenceCheck
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .grobidpdf.paper_parser import PaperParser
from .paper_download import parse_with_grobid, download_paper_to_memory
from .request_utils import RateLimit, async_request_template, HEADERS, SessionManager


def has_url(item: Dict[str, Any]) -> bool:
    pass


def llm_should_retry(exception: BaseException) -> bool:
    if isinstance(exception, KeyboardInterrupt): return False
    if isinstance(exception, NotImplementedError): return False
    return True


class FactualLLMClient(AsyncChat):

    PROMPT: str = FACTUAL_CORRECTNESS_PROMPT

    def _availability(self, response, context):
        response = extract_json(response)
        return response['judgment'], response['evidence']


class WebSearchFallback:
    
    def __init__(self, config: ToolConfig):
        self.llm = FactualLLMClient(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)
        self.url = config.websearch_url
        self.key = config.websearch_apikey
        self.grobid = config.grobid_url
        self.paper_parser = PaperParser()

    async def _websearch(self, query):
        headers = {"Content-type": "application/json", "X-API-KEY": self.key} | HEADERS
        payload = {"q": query}
        async with RateLimit.WEBSEARCH_SEMAPHORE:
            response = await async_request_template('post', self.url, headers, payload)

    async def _visit_url(self, url: str, item: Dict[str, Any]):
        try:
            # 第一步：发出response
            async with RateLimit.WEBSEARCH_SEMAPHORE:
                session = SessionManager.get()
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    # 第二步：检查response文件类型，分为PDF、文本、错误信息和其他
                    headers = resp.headers
                    match headers['Content-type']:
                        case 'application/pdf':
                            # 2.1 PDF：下载并解析内容，转换成skeleton
                            pdf_buffer = await download_paper_to_memory(url)
                            if not pdf_buffer:
                                return {"item": item, "exist": True, "issue": "fail to download PDF"}
                            # 步骤2: 通过 GROBID 解析
                            xml_content = await parse_with_grobid(self.grobid, pdf_buffer)
                            if not xml_content:
                                return {"item": item, "exist": True, "issue": "fail to convert PDF into XML"}
                            paper = self.paper_parser.parse(xml_content)
                            if not xml_content:
                                return {"item": item, "exist": True, "issue": "fail to parse XML"}
                            return {"item": item, "exist": True, "full_content": paper.get_skeleton()}
                        case 'text/html':
                            # 2.2 文本：用trafilatura获取全文，字符串形式输出。
                            content = await resp.text()
                            extracted = extract(content)
                            return {"item": item, "exist": True, "full_content": extracted}
                        case _:
                            # 2.3 其他：只验证链接可访问即可。
                            return {"item": item, "exist": True, "issue": "Not parseable content"}
        except Exception as e:
            pass

    async def _websearch(self, item: Dict[str, Any]):
        pass

    async def _search_single_item(self, item: Dict[str, Any]):
        if url := has_url(item):
            await self._visit_url(url, item)
        else:
            await self._websearch(item)
    
    async def __call__(self, zero_refs: Dict[str, Any]) -> Dict[str, List]:
        """
        输入：status == 0 的所有样本。
        """
        tasks = [asyncio.create_task(self._search_single_item(x)) for x in zero_refs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
