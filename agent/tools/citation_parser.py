import re
import json
import asyncio
from typing import Dict, Any

from .tool_config import ToolConfig
from .paper_download import PaperDownload
from .utils import valid_check, index_to_abstract
from .openalex import openalex_search_paper, URL_DOMAIN, OPENALEX_SELECT
      

class CitationParser:
    """
    A class to download and parse all cited papers of the survey paper using GROBID service.
    Returns a Paper object with hierarchical structure.
    """

    SELECT = f"{OPENALEX_SELECT},best_oa_location,locations"
    
    def __init__(self, config: ToolConfig):
        self.n_workers = config.grobid_num_workers
        self.paper_downloader = PaperDownload()
    
    async def _search_paper_from_api(self, title: str | Dict[str, Any]) -> Dict[str, Any]:
        """
        paper = citation_info，可能要改成特定的类。
        """
        if isinstance(title, dict): title = title['title']
        # 获取所有正确结果      
        info = {
            "metadata": None, 
            "title": title,
            "abstract": "", 
            "full_content": {}
        }
        paper_title = re.sub(r"[:,.!?&]", "", title)
        if not paper_title: 
            info['status'] = 3
            info['full_content'] = title
            return info
        try:
            results = await openalex_search_paper("works", {"title.search": paper_title}, select=self.SELECT)      
        except Exception as e:
            print(f"Cannot get paper {title} by title.search {e}")
            results = {}
        print(f"get {len(results.get("results", []))} papers")
        for paper_info in results.get("results", []):
            if valid_check(paper_title, paper_info['title']):
                paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
                # download full content
                if not info['full_content'] and (paper := await self.paper_downloader.download_single_paper(paper_info)): 
                    info['full_content'] = paper['full_content']
                    if paper['abstract']: info['abstract'] = paper['abstract']
                # abstract
                if not info['abstract']: info['abstract'] = paper_info['abstract']
                # metadata
                del paper_info['locations'], paper_info['best_oa_location']
                info['metadata'] = paper_info

        # status == 0 -> OK
        # status == 1 -> fail to download paper
        # status == 2 -> fail to download paper and fetch abstract
        # status == 3 -> fail to get information of the citation. TODO: Fallback to WebSearch.
        if info['full_content']: 
            info['status'] = 0
        elif info['abstract']:
            info['status'] = 1
            info['full_content'] = info['abstract']
        elif info['metadata']:
            info['status'] = 2
            info['abstract'] = ""
        else:
            info['status'] = 3
            info['abstract'] = ""
        print(f"Paper {title} Status {info['status']}")
        return info
    
    async def __call__(self, citations: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        import tqdm
        paper_content_map = {}  # key: citation key; value: citation info.
        # keys, tasks = [], []
        # for k, v in citations.items():
        #     keys.append(k)
        #     tasks.append(asyncio.create_task(self._search_paper_from_api(v)))
        # for k, task in tqdm.tqdm(zip(keys, asyncio.as_completed(tasks)), total=len(tasks)):
        #     info = await task
        #     paper_content_map[k] = info
        for k, v in tqdm.tqdm(citations.items(), total=len(citations)):
            info = await self._search_paper_from_api(v)
            paper_content_map[k] = info
        return {"paper_content_map": paper_content_map}
