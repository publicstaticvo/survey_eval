import os
import re
import glob
import time
import tqdm
import arxiv
import random
import difflib
import logging
import requests

from constants import *
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor as TPE
from request_utils import openalex_search_paper, valid_check

arxiv_pattern = re.compile(r"(?<![0-9])[0-9]{4}\.[0-9]{4,5}(?![0-9])")
json_pattern = re.compile(r"\{.+?\}", re.DOTALL)
logging.basicConfig(filename="../logs/clean1.log", level=logging.INFO)
arxiv_logger = logging.getLogger('arxiv')
arxiv_logger.setLevel(logging.WARNING)


def yield_local(fn):
    with open(fn, "r+", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except:
                    pass


def print_json(d, fn):
    with open(fn, "w+", encoding="utf-8") as f:
        for x in d:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


class PaperDownloader:
    def __init__(self, s2api_key: List[str] = [], llm_key: Dict = {}, n_workers: int = 20):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        })
        random.shuffle(s2api_key)
        self.api_key = s2api_key
        self.llm_base_url = llm_key['domain']
        self.llm_api_key = llm_key['key']
        self.llm_model = "gpt-oss-120b"
        self.n_workers = n_workers
        self.error_titles = []
        self.times_429 = 0
        
    def search_arxiv(self, title: str, max_results: int = 3, retry: int = 5) -> Optional[str]:
        """搜索arXiv论文"""
        try:
            client = arxiv.Client(num_retries=retry)
            search = arxiv.Search(query=title, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            for result in client.results(search):
                if result: 
                    text = result.entry_id.split("/")[-1]
                    if "v" in text: text = text[:text.index("v")]
                    return text
        except:
            return
    
    def search_semantic_scholar(self, title: str, max_results: int = 10, retry: int = 5) -> Optional[Dict]:
        """使用Semantic Scholar API搜索"""
        title = title.replace("{", "").replace("}", "")
        while retry != 0:
            try:
                # time.sleep(0.5)
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {'query': title, 'limit': max_results, 'fields': 'paperId,title,openAccessPdf,url,citationCount'}
                # with self.session.with_api_key(self.api_key[self.times_429 % len(self.api_key)]) as temp_session:                
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data['total'] > 0:
                    for item in data['data']:
                        if difflib.SequenceMatcher(None, item['title'], title).ratio() < 0.9: continue
                        url = item.get('openAccessPdf', {}).get('url', "")
                        arxiv_key = arxiv_pattern.findall(url)
                        if arxiv_key: return f"arxiv:{arxiv_key[-1]}"
                        if url: return url
                return ""
            except requests.exceptions.ReadTimeout:
                logging.error(f"Semantic Scholar Error: Read time out, Retry: {retry}")
                retry -= 1
                time.sleep(1)
            except Exception as e:
                if response.status_code == 429:
                    logging.error(f"Semantic Scholar Error: 429")
                else:
                    logging.error(f"Semantic Scholar Error: {e}, Retry: {retry}")
                    retry -= 1
                time.sleep(1)
        return "Network Error +"
        
    def search_openalex(self, title: str, retry: int = 5) -> str:
        title = re.sub(r"[\{\}\(\)\[\]\$'`\"]", "", title)
        response = openalex_search_paper("works", {"title.search": title}, add_mail=True, retry=retry)
        for paper_info in response.get("results", []):
            if not valid_check(title, paper_info.get("title", "")): continue
            if (best_oa_location := paper_info.get("best_oa_location", {})):
                return best_oa_location["pdf_url"]
            locations_count = paper_info.get("locations_count", 0)
            if locations_count:
                for l in paper_info.get("locations", []):
                    if l['is_oa'] and l['pdf_url']:
                        return l['pdf_url']

    def download_pdf(self, pdf_url: str, filename: str) -> bool:
        """下载PDF文件"""
        try:
            response = self.session.get(pdf_url, timeout=30, stream=True)
            if response.status_code == 200:
                os.makedirs('downloaded_papers', exist_ok=True)
                filepath = os.path.join('downloaded_papers', filename)                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"PDF已下载: {filepath}")
                return True
        except Exception as e:
            logging.error(f"PDF下载错误: {e}")
        return False
    
    def comprehensive_search(self, title: str) -> Dict[str, str]:
        """综合搜索论文"""  
        # print("1. 搜索arXiv...")
        result = self.search_arxiv(f'ti:{title}')
        if result is not None: return {"source": "arXiv", "url": result}     
        result = self.search_arxiv(f'ti:"{title}"')
        if result is not None: return {"source": "arXiv", "url": result}     
        # print("2. 搜索Semantic Scholar...")
        result = self.search_semantic_scholar(title)
        if result == "Network Error +":
            return {"source": "Network Error", "url": ""}
        if result:
            if "arxiv" in result: return {"source": "arXiv", "url": result[6:]}
            return {"source": "s2", "url": result}
        return {"source": "", "url": ""}

    def get_paper_url(self, cite):
        arxiv_key = ""
        if 'journal' in cite:
            arxiv_key = arxiv_pattern.findall(cite['journal'])
            if arxiv_key: arxiv_key = arxiv_key[-1]
        if not arxiv_key and 'volume' in cite:
            arxiv_key = arxiv_pattern.findall(cite['volume'])
            if arxiv_key: arxiv_key = arxiv_key[-1]
        if arxiv_key:
            return 'inline arXiv', arxiv_key
        # print(f"request for {cite['title']}")
        result = self.comprehensive_search(cite['title'])
        if result['source']: return result['source'], result['url']
        else: return "", None

    def run_openalex(self, n_workers: int = 10):                      
        find, null = {}, []
        if n_workers <= 1:
            # for arg in ['arXiv', 's2', 'null']:
            for x in tqdm.tqdm(yield_local("../crawled_papers/citations/null.jsonl")):
                if x['title'] in find: continue
                url = self.search_openalex(x['title'])
                if url: find[x['title']] = url
                else: null.append(x['title'])
        else:
            def inner_run(title: str) -> Tuple[str, Dict]:
                return title, self.search_openalex(title)
            
            pending_results = set()
            for arg in ['arXiv', 's2', 'null']:
                for x in yield_local(f"../crawled_papers/citations/{arg}.jsonl"):
                    pending_results.add(x['title'])
            
            with TPE(max_workers=n_workers) as executor:
                for title, url in tqdm.tqdm(executor.map(inner_run, pending_results), total=len(pending_results)):
                    if url: find[title] = url
                    else: null.append(title)
            logging.info(f"Get {len(find)} New URLs")
        with open("../crawled_papers/citations/find.json", "w+", encoding='utf-8') as f: json.dump(find, f)
        with open("../crawled_papers/citations/null.json", "w+", encoding='utf-8') as f: json.dump(null, f)
    
    def run(self):
        null = []
        title_set = set()
        with open("../crawled_papers/citations/arXiv.jsonl", "a+") as f_arxiv, open("../crawled_papers/citations/s2.jsonl", "a+") as f_s2:
            for x in tqdm.tqdm(yield_local("../crawled_papers/citations/null.jsonl"), total=10065):
                if x['title'] in title_set: continue
                new_info = self.search_semantic_scholar(x['title'])
                if new_info:
                    if 'arxiv' in new_info:
                        x['source'] = 'arxiv'
                        x['volume'] = new_info[6:]
                        f_arxiv.write(json.dumps(x) + "\n")
                    else:
                        x['source'] = 's2'
                        x['url'] = new_info
                        f_s2.write(json.dumps(x) + "\n")
                    title_set.add(x['title'])
                else:
                    null.append(x)           
        print_json(null, "../crawled_papers/citations/null.jsonl")         

    def _request_llm_for_title(self, text, retry=3):
        message = [{"role": "user", "content": GET_TITLE_FROM_LATEX_PROMPT.format(content=text)}]
        while retry > 0:
            try:
                sampling_params = {
                    "model": self.llm_model,
                    "messages": message,
                    "temperature": 0.6,
                    "top_p": 0.95, "top_k": 20,
                    "max_tokens": 4096
                }
                headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.llm_api_key}'}
                url = f"{self.llm_base_url}/v1/chat/completions"
                response = requests.post(url, headers=headers, data=json.dumps(sampling_params), timeout=600)
                response.raise_for_status()
                message = json.loads(response.text)['choices'][0]['message']
                text = message['content']
                title = json_pattern.findall(text)
                title = json.loads(title[0])
                assert all(x in title for x in ["title", "arxiv_id"])
                return title
            except Exception as e:
                retry -= 1
                logging.error(f"Error: {e}, Retry: {retry}")
                time.sleep(5)    


if __name__ == "__main__":
    with open("../../api_key.json") as f:
        info = json.load(f)
        keys = info['semanticscholar']['key']
        llm_keys = info['cstcloud']
    PaperDownloader(keys, llm_keys).run_openalex(8)
