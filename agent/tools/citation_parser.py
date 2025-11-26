import io
import logging
import requests
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor as TPE

from paper_parser import GROBIDParser
from utils import openalex_search_paper, valid_check, index_to_abstract, URL_DOMAIN


@dataclass
class HTMLSection:
    numbers: List[int]
    level: int
    title: str
    element: Optional[ET.Element]
    paragraphs: List[str]


class CitationProcessInput(BaseModel):
    citations: List[Dict[str, str]] = Field(description="List of papers cited in the survey.")


class CitationParser(BaseTool):
    """
    A class to download and parse all cited papers of the survey paper using GROBID service.
    Returns a Paper object with hierarchical structure.
    """
    
    name: str = "paper_parser"
    description: str = (
        "Downloads and parses the full text of cited papers. "
        "Attempts to fetch PDF via Grobid; falls back to OpenAlex abstract/title. "
        "Returns a map of {id: text}."
    )
    args_schema: type[BaseModel] = CitationProcessInput
    
    def __init__(self, grobid_url: str = "http://localhost:8070", n_workers: int = 10, **kwargs):
        """
        Initialize the GROBID parser.
        
        Args:
            grobid_url: URL of the GROBID service (default: localhost:8070)
        """
        super().__init__(**kwargs)
        self.n_workers = n_workers
        self.paper_parser = GROBIDParser(grobid_url) 

    def _download_paper_to_memory(self, url) -> Optional[tuple[str, io.BytesIO, str]]:
        """
        Downloads a PDF and parses it with Grobid without saving to disk.
        """
        if not url: return
        # 1. Download the PDF into memory
        try:
            pdf_response = requests.get(url, timeout=30)
            pdf_response.raise_for_status()
        except Exception as e:
            logging.error(f"PDF {url} 下载错误: {e}")
            return

        # 2. Create a file-like object from the bytes
        pdf_file_obj = io.BytesIO(pdf_response.content)
        pdf_file_obj.name = "temp_paper.pdf"  # Grobid expects a filename hint
        
        return pdf_file_obj.name, pdf_file_obj, 'application/pdf'
    
    def _search_paper_from_api(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: 改成填表，由于OpenAlex会有重复条目，所以需要手动去重，在多个同名条目中寻找共同的ID。
        # TODO: 可第一步统计所有正确结果，第二步从结果中选择去重。
        paper_title = paper['title']
        # 获取所有正确结果
        on_target = []
        results = openalex_search_paper("works", {"title.search": paper_title}).get("results", [])
        for paper_info in results:
            if valid_check(paper_title, paper_info.get("title", "")): 
                paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
                on_target.append(paper_info)
        
        # 从正确结果中统计。
        best_field = {
            "id": [],
            "authorship": [],
            "title": paper['title'], 
            "cited_by_count": 0,
            "counts_by_year": [],
            "abstract": "",
            'publication_date': "",
            "file_obj": None
        }
        for paper in on_target:
            # if not best_field['id']: best_field['id'] = paper['id']
            # best_field['cited_by_count'] = max(best_field['cited_by_count'], paper['cited_by_count'])
            best_field['id'].append(paper['id'])
            best_field['cited_by_count'] += paper['cited_by_count']
            if not best_field['authorship']:
                best_field['authorship'] = [x['author']['display_name'] for x in paper['authorship']]
            if not best_field['abstract'] and (abstract := index_to_abstract(paper['abstract_inverted_index'])):
                best_field['abstract'] = abstract
            if not best_field['file_obj'] and \
                (best_oa_location := paper.get("best_oa_location", {})) and \
                (file_obj := self._download_paper_to_memory(best_oa_location["pdf_url"])):
                best_field['file_obj'] = file_obj
            if not best_field['file_obj'] and paper_info.get("locations_count", 0):
                for l in paper_info.get("locations", []):
                    if (file_obj := self._download_paper_to_memory(l['pdf_url'])): 
                        best_field['file_obj'] = file_obj
                        break            
            if not best_field['publication_date'] or \
                (date := paper['publication_date']) < best_field['publication_date']:
                best_field['publication_date'] = date
        return best_field
    
    # def _get_paper_features(self, paper: Dict[str, Any]):
    #     # The following information to get:
    #     return {
    #         "id": paper['id'].replace(URL_DOMAIN, ""),
    #         "title": paper['title'], 
    #         "cited_by_count": paper['cited_by_count'],
    #         "counts_by_year": paper['counts_by_year'],
    #         "abstract": paper['abstract'],
    #         'publication_date': paper['publication_date'],
    #     }
    
    def _run(self, citations: Dict[str, Dict[str, str]]) -> str:
        paper_content_map = {}  # key: citation key; value: citation info.
        pending_jobs = []
        for k, v in citations.items():
            v['citation_key'] = k
            pending_jobs.append(v)
        with TPE(max_workers=self.n_workers) as executor:
            for paper, info in zip(pending_jobs, executor.map(self._search_paper_from_api, pending_jobs)):
                if info['file_obj']:
                    paper_content = self.paper_parser.parse_pdf(info['file_obj'])
                    info['full_content'] = paper_content.get_skeleton()
                    info['status'] = "ok" 
                elif info['abstract']:
                    info['status'] = "fail to download paper"
                    info['full_content'] = info['abstract']
                elif info['id']:
                    info['status'] = "fail to download paper and fetch abstract"
                    info['full_content'] = info['title']
                else:
                    info = {'status': "fail to get information of the citation. Please check its existance.", 
                            'content': paper['title']}
                paper_content_map[paper['citation_key']] = info     
        return paper_content_map
    
    async def _arun(self, citations: Dict[str, Dict[str, str]]) -> str:
        return await self._run(citations)
