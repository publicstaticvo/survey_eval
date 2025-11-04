import time
import random
import logging
import requests
from typing import Optional
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}


def s2api_search_paper(
        title: Optional[str] = None, 
        arxiv_id: Optional[str] = None, 
        fields: str = "paperId,title", 
        max_results: int = 5, 
        retry: int = 3, 
        unlimited_429: bool = True
    ):
    """使用Semantic Scholar API搜索"""
    assert title or arxiv_id, "MUST SPECIFY ONE OF title AND arxiv_id"
    # arxiv_id first
    if arxiv_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        params = {"fields": fields}
    else:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': title, 'limit': max_results, 'fields': fields}  
    
    while retry != 0:
        try:
            time.sleep(random.random())                
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.HTTPError as e:
            # if unlimited_429 and response.status_code == 429:
            #     logging.warning(f"Semantic Scholar Error: 429")
            if not unlimited_429 or response.status_code != 429:
                logging.error(f"Semantic Scholar Error: {e}, Retry: {retry}")
                retry -= 1
            time.sleep(1)
        except Exception as e:
            logging.error(f"Semantic Scholar {e}, Retry: {retry}")
            retry -= 1
            time.sleep(1)
    return {}


def s2api_search_paper_single_pass(
        title: Optional[str] = None, 
        arxiv_id: Optional[str] = None, 
        fields: str = "paperId,title", 
        max_results: int = 5
    ):
    """使用Semantic Scholar API搜索，不retry"""
    assert title or arxiv_id, "MUST SPECIFY ONE OF title AND arxiv_id"
    if arxiv_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        params = {"fields": fields}
    else:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': title, 'limit': max_results, 'fields': fields}  
    try:
        time.sleep(random.random())
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': title, 'limit': max_results, 'fields': fields}      
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logging.error(f"Single Pass {e}")
        return {}