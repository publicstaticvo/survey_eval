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
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple
# from concurrent.futures import ThreadPoolExecutor as TPE

arxiv_pattern = re.compile(r"(?<![0-9])[0-9]{4}\.[0-9]{4,5}(?![0-9])")
json_pattern = re.compile(r"\{.+?\}", re.DOTALL)
logging.basicConfig(filename="../logs/clean2.log", level=logging.INFO)
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

    def old_run(self):
        files = glob.glob("../crawled_papers/cs/*/citations-clean.jsonl")
        # jobs = []
        arxiv_set, title_set = set(), set()
        info_to_title = {}
        with open("../crawled_papers/cited_papers.jsonl", "a+", encoding='utf-8') as fout, \
             open("../crawled_papers/cited_arxiv_ids.txt", "a+", encoding='utf-8') as fout_2:
            for i, f in enumerate(files):
                d = load_local(f)
                arxiv_id = Path(f).parent.name
                logging.info(f"File {i + 1} / {len(files)} - {arxiv_id}/citation.jsonl")
                for j, x in enumerate(d):
                    logging.info(f"  Sentence {j + 1} / {len(d)}")
                    for k in x['citation']:
                        cite = x['citation'][k]
                        if 'journal' in cite and cite['journal'] in ['arXiv', 'inline arXiv', 's2']:
                            title_set.add(cite['title'])
                            continue
                        if 'title' not in cite and 'info' in cite:
                            arxiv_in_info = arxiv_pattern.findall(cite['info'])
                            if arxiv_in_info:
                                arxiv_id = arxiv_in_info[0]
                                if arxiv_id not in arxiv_set:
                                    fout_2.write(arxiv_id + "\n")
                                    arxiv_set.add(arxiv_id)
                                cite['source'] = "inline arXiv"
                                cite['volume'] = arxiv_id
                                if 'title' in cite: 
                                    cite['title'] = cite['title'].strip()
                                    if cite['title'].endswith("."): cite['title'] = cite['title'][:-1]
                                    title_set.add(cite['title'])
                                continue
                            if "author" in cite: 
                                cite = {"info": f"{cite['author']}\n{cite['title']}\n{cite['info']}"}
                                x['citation'][k] = cite
                            if cite['info'] in info_to_title:
                                cite['title'] = info_to_title[cite['info']]
                                continue
                            title = self._request_llm_for_title(cite['info'])
                            if title: 
                                if not title['title']: continue
                                if title["arxiv_id"]:
                                    arxiv_id = arxiv_pattern.findall(title["arxiv_id"])
                                    if arxiv_id:
                                        arxiv_id = arxiv_id[0]
                                        if arxiv_id not in arxiv_set:
                                            fout_2.write(arxiv_id + "\n")
                                            arxiv_set.add(arxiv_id)
                                        cite['source'] = "inline arXiv"
                                        cite['volume'] = arxiv_id
                                        cite['title'] = title['title']
                                        if cite['title'].endswith("."): cite['title'] = cite['title'][:-1]
                                        title_set.add(cite['title'])
                                        continue
                                cite['title'] = title['title']
                                info_to_title[cite['info']] = cite['title']
                            else: continue
                        if 'title' not in cite: continue
                        cite['title'] = cite['title'].strip()
                        if cite['title'].endswith("."): cite['title'] = cite['title'][:-1]
                        if cite['title'] in title_set: continue
                        # cite['source'] = f
                        # cite['citation_key'] = k
                        # jobs.append(cite)
                        source, url = self.get_paper_url(cite)
                        cite['source'] = source
                        if source == "Network Error": continue
                        if "arXiv" in source:
                            fout_2.write(url + "\n")
                            arxiv_set.add(url)
                            cite['volume'] = url
                        elif source:
                            cite['url'] = url
                            fout.write(json.dumps({"title": cite['title'], 'url': url}) + "\n")
                        title_set.add(cite['title'])
                logging.info("Writing to clean.jsonl")
                print_json(d, f)

        # files_to_update = {}
        # with TPE(max_workers=self.n_workers) as executor:
        #     for cite in tqdm.tqdm(executor.map(self.get_paper_url, jobs), desc="Async Jobs"):
        #         if cite["source"] not in files_to_update:
        #             files_to_update[cite['source']] = {}
        #         files_to_update[cite['source']][cite['citation_key']] = {k: v for k, v in cite.items() if k not in ['source', 'citation_key']}
        # for f in tqdm.tqdm(files, desc="Reorganizing files"):
        #     citations = load_local(f)
        #     for x in citations:
        #         for k in x['citation']:
        #             x['citation'][k].update(files_to_update[f][k])
        #     print_json(citations, f"{f[:-6]}s-clean.jsonl")

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
    PaperDownloader(keys, llm_keys).run()
