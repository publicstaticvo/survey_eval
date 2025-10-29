import time
import random
import logging
import requests
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}


def s2api_search_paper(title: str, fields: list[str], max_results: int = 5, retry: int = 3, unlimited_429: bool = True):
    """使用Semantic Scholar API搜索"""
    while retry != 0:
        try:
            time.sleep(random.random())
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {'query': title, 'limit': max_results, 'fields': fields}      
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.ReadTimeout:
            logging.error(f"Semantic Scholar Error: Read time out, Retry: {retry}")
            retry -= 1
            time.sleep(1)
        except Exception as e:
            # if unlimited_429 and response.status_code == 429:
            #     logging.error(f"Semantic Scholar Error: 429")
            if not unlimited_429 or response.status_code != 429:
                logging.error(f"Semantic Scholar Error: {e}, Retry: {retry}")
                retry -= 1
            time.sleep(1)
    return {}


def s2api_search_paper_single_pass(title: str, fields: list[str], max_results: int = 3):
    """使用Semantic Scholar API搜索，不retry"""
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