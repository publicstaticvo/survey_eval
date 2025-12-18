import re
import asyncio, aiohttp
from typing import Dict, Optional, Any

from .tool_config import ToolConfig
from .paper_parser import PaperParser
from .utils import valid_check, index_to_abstract
from .request_utils import openalex_search_paper, URL_DOMAIN, RateLimit, try_one_url, SessionManager


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
        

async def process_single_paper(session: aiohttp.ClientSession, paper_meta: dict) -> Optional[dict]:
    """处理单篇论文：尝试所有 URL，返回第一个成功的"""
    
    # 为该论文的所有 URL 创建任务
    tasks = [asyncio.create_task(try_one_url(session, url)) for url in list(yield_location(paper_meta))]
    
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


class CitationParser:
    """
    A class to download and parse all cited papers of the survey paper using GROBID service.
    Returns a Paper object with hierarchical structure.
    """
    
    def __init__(self, config: ToolConfig):
        self.n_workers = config.grobid_num_workers
        self.paper_parser = PaperParser(config.grobid_url, False)
    
    async def _search_paper_from_api(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        paper = citation_info，可能要改成特定的类。
        """
        paper_title = re.sub(r"[:,.!?&]", "", paper['title'])
        # 获取所有正确结果      
        info = {
            "metadata": None, 
            "title": paper['title'],
            "abstract": "", 
            "full_content": {}
        }
        
        async with RateLimit.OPENALEX_SEMAPHORE:
            results = await openalex_search_paper("works", {"default.search": paper_title})

        session = SessionManager.get()        
        for paper_info in results.get("results", []):
            if valid_check(paper_title, paper_info['display_name']): 
                paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
                # download full content
                if not info['full_content'] and (content := await process_single_paper(session, info)): 
                    info['full_content'] = content
                # abstract
                if not info['abstract'] and (abstract := index_to_abstract(paper_info['abstract_inverted_index'])):
                    info['abstract'] = abstract
                # metadata
                paper_info = self._get_metadata(paper_info)
                if info['metadata'] is None: 
                    info['metadata'] = paper_info
                    info['metadata']['id'] = [info['metadata']['id']]
                else:
                    info['metadata']['id'].append(paper_info['id'])
                    info['metadata']['locations'].extend(paper_info['locations'])
                    if paper_info['cited_by_count'] > info['metadata']['cited_by_count']:
                        info['metadata']['cited_by_count'] = paper_info['cited_by_count']
                        info['metadata']['counts_by_year'] = paper_info['counts_by_year']
                    if paper_info['publication_date'] < info['metadata']['publication_date']:
                        info['metadata']['publication_date'] = paper_info['publication_date']

        # status == 0 -> OK
        # status == 1 -> fail to download paper
        # status == 2 -> fail to download paper and fetch abstract
        # status == 3 -> fail to get information of the citation. TODO: Fallback to WebSearch.
        if info['full_content']: 
            info['status'] = 0
        elif info['abstract']:
            info['status'] = 1
            info['full_content'] = info['abstract']
        elif info['metadatas']:
            info['status'] = 2
            info['abstract'] = info['full_content'] = paper['title']
        else:
            info['status'] = 3
            info['full_content'] = paper['title']
        return info
    
    def _get_metadata(self, paper: Dict[str, Any]):
        # The following information to get:
        return {
            "id": paper['id'].replace(URL_DOMAIN, ""),
            "ids": paper['ids'],
            "title": paper['display_name'],
            "locations": [x['source'] for x in paper['locations']],
            "cited_by_count": paper['cited_by_count'],
            "counts_by_year": paper['counts_by_year'],
            "publication_date": paper['publication_date'],
        }
    
    async def __call__(self, citations: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        paper_content_map = {}  # key: citation key; value: citation info.
        keys, tasks = [], []
        for k, v in citations.items():
            keys.append(k)
            tasks.append(asyncio.create_task(self._search_paper_from_api(v)))
        for paper, info in zip(keys, await asyncio.gather(*tasks)):
            paper_content_map[paper] = info
        return {"paper_content_map": paper_content_map}
