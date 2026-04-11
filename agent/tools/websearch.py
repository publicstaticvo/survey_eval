import asyncio, aiohttp
from typing import List, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,           # 最大重试次数
    wait_exponential,             # 指数退避
    retry_if_exception,           # 遇到什么异常才重试
)
from trafilatura import extract
from sentence_transformers.util import cos_sim

from .tool_config import ToolConfig
from .utils import extract_json
from .llmclient import AsyncChat
from .evidence_check import EvidenceCheck
from .prompts import FACTUAL_CORRECTNESS_PROMPT
from .grobidpdf.paper_parser import PaperParser
from .sbert_client import SentenceTransformerClient
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
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)
        self.llm = FactualLLMClient(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)
        self.url = config.websearch_url
        self.key = config.websearch_apikey
        self.grobid = config.grobid_url
        self.paper_parser = PaperParser()

    def _title_similarity(self, query: str, result: str):
        embed = self.sentence_transformer.embed([query, result])
        return cos_sim(embed[0], embed[1])
    
    async def _extract_content_from_url(self, url: str, item: Any):
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
                            return {"issue": "fail to download PDF"}
                        # 步骤2: 通过 GROBID 解析
                        xml_content = await parse_with_grobid(self.grobid, pdf_buffer)
                        if not xml_content:
                            return {"issue": "fail to convert PDF into XML"}
                        paper = self.paper_parser.parse(xml_content)
                        if not xml_content:
                            return {"issue": "fail to parse XML"}
                        return {"full_content": paper.get_skeleton()}
                    case 'text/html':
                        # 2.2 文本：用trafilatura获取全文，字符串形式输出。
                        content = await resp.text()
                        extracted = extract(content)
                        return {"full_content": extracted}
                    case _:
                        # 2.3 其他：只验证链接可访问即可。
                        return {"issue": "Not parseable content"}

    async def _websearch(self, item: Dict[str, Any]):
        headers = {"Content-type": "application/json", "X-API-KEY": self.key} | HEADERS
        payload = {"q": item['title']}
        async with RateLimit.WEBSEARCH_SEMAPHORE:
            results = await async_request_template('post', self.url, headers, payload)
        candidates = []
        # 先验证results['knowledgeGraph']，再验证results['organic']
        if self._title_similarity(results["knowledgeGraph"]['title'], item['title']) > 0.8:
            candidates.append({
                "title": results["knowledgeGraph"]['title'],
                'link': results["knowledgeGraph"]['descriptionLink'],
                'snippet': results["knowledgeGraph"]['description'],
                'date': results["knowledgeGraph"]['attributes'].get('Initial release date', None),
                'position': 0
            })
        for result in results['organic']:
            # 标题字符串相似度（简单edit distance或token overlap）
            if self._title_similarity(result["title"], item['title']) > 0.8:
                candidates.append(result)
        if not candidates: return {"item": item, "exist": False}
        # TODO: 实现“让LLM过滤所有的candidates，判断每一项匹配/不匹配，返回所有匹配对应的完整项目列表”。
        # TODO: 需要像其他模块的代码实现一样继承AsyncChat，并自己将提示词写在prompts.py中。
        filtered_candidates = ...
        if not filtered_candidates: return {"item": item, "exist": False}
        content = self._extract_content_from_url(filtered_candidates[0]['link'], item)
        return {"item": item, "exist": True, **content}

    async def _visit_url(self, url: str, item: Dict[str, Any]):
        try:
            content = self._extract_content_from_url(url, item)
            return {"item": item, "exist": True, **content}
        except Exception as e:
            return {"item": item, "exist": False}

    async def _search_single_item(self, item: Dict[str, Any]):
        if url := has_url(item):
            return await self._visit_url(url, item)
        else:
            return await self._websearch(item)
    
    async def __call__(self, refs_to_search: Dict[str, Any]) -> Dict[str, List]:
        """
        输入：status == 3 的所有样本。
        """
        tasks = [asyncio.create_task(self._search_single_item(x)) for x in refs_to_search]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
