import re
import time
import random
import logging
import requests
import unidecode
import Levenshtein
from typing import Optional, Union
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}
email_pool = [
    "dailyyulun@gmail.com",
    "fqpcvtjj@hotmail.com",
    "ts.yu@siat.ac.cn",
    "yutianshu2025@ia.ac.cn",
    "yutianshu25@ucas.ac.cn",
    "dailyyulun@163.com",
    "lundufiles@163.com"
]


def valid_check(query: str, target: str, ratio: float = 0.1) -> bool:
    def normalize(text: str) -> str:
        text = unidecode.unidecode(text)
        text = re.sub(r"[^0-9a-zA-Z\s]", "", text)
        return text.lower()

    distance = Levenshtein.distance(normalize(query), normalize(target))
    return distance <= ratio * len(query)


def request_template(url: str, headers: dict[str, str], parameters: dict[str, str], sleep: bool = True) -> dict:
    if sleep: time.sleep(2 * random.random())
    response = requests.get(url, headers=headers, params=parameters, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data


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
            return request_template(url, headers, params)
        except requests.exceptions.HTTPError as e:
            status_code = int(str(e)[:3])
            # if unlimited_429 and response.status_code == 429:
            #     logging.warning(f"Semantic Scholar Error: 429")
            if not unlimited_429 or status_code != 429:
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
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': title, 'limit': max_results, 'fields': fields}      
        return request_template(url, headers, params)
    except Exception as e:
        logging.error(f"Single Pass {e}")
        return {}
    

def openalex_search_paper(
        endpoint: str, 
        filter: Optional[dict[str, str]] = None, 
        max_results: int = -1, 
        add_mail: Union[bool, str] = True,
        retry: int = 3
    ) -> dict:
    assert max_results <= 200, "Per page is at most 200"
    while retry != 0:
        try:
            url = f"https://api.openalex.org/{endpoint}"
            if filter is not None:
                filter_string = ",".join([f"{k}:{v}" for k, v in filter.items()])
                request_parameters = {"filter": filter_string}
            else:
                request_parameters = {}
                if max_results >= 0:
                    request_parameters['sample'] = max_results
                    if max_results > 25:
                        request_parameters['per-page'] = 200
            if add_mail:
                request_parameters['mailto'] = add_mail if isinstance(add_mail, str) else random.choice(email_pool)
            return request_template(url, None, request_parameters)
        except requests.exceptions.RequestException as e:
            what = str(e)
            logging.error(f"OpenAlex {e}, Retry: {retry}")
            if any(f"{code} Client Error" in what for code in [400, 401, 403, 404]): return {}
            retry -= 1
            time.sleep(1)
    return {}