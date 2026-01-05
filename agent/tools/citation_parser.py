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
        
        results = await openalex_search_paper("works", {"title.search": paper_title}, select=self.SELECT)
      
        for paper_info in results.get("results", []):
            if valid_check(paper_title, paper_info['display_name']): 
                paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
                # download full content
                if not info['full_content'] and (paper := await self.paper_downloader.download_single_paper(info)): 
                    paper_info |= paper
                # abstract
                if not info['abstract'] and (abstract := index_to_abstract(paper_info['abstract_inverted_index'])):
                    info['abstract'] = abstract
                # metadata
                paper_info = self._get_metadata(paper_info)
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
    

async def main():
    config = ToolConfig()
    query = ""
    results = await CitationParser(config)(query)
    with open("debug/anchor_papers.json") as f:
        json.dump(results['anchor_papers'], f, ensure_ascii=False)
    with open("debug/topics.txt") as f:
        f.write("\n".join(results['golden_topics']))


if __name__ == "__main__":
    asyncio.run(main())
