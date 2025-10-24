import os
import re
import glob
import time
import tqdm
import arxiv
import random
import requests

from constants import *
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, List, Tuple
# from concurrent.futures import ThreadPoolExecutor as TPE

arxiv_pattern = re.compile(r"(?<![0-9])[0-9]{4}\.[0-9]{4,5}(?![0-9])")
json_pattern = re.compile(r"\{.+?\}", re.DOTALL)


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


class APISession(requests.Session):
    def __init__(self):
        super().__init__()
        self._original_headers = self.headers.copy()

    @contextmanager
    def with_api_key(self, api_key):
        try:
            self.headers["Authorization"] = f"Bearer {api_key}"
            yield self
        finally:
            self.headers.clear()
            self.headers.update(self._original_headers)


class PaperDownloader:
    def __init__(self, s2api_key: List[str] = [], llm_key: Dict = {}, n_workers: int = 20):
        self.session = APISession()
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
        
    def search_arxiv(self, title: str, max_results: int = 3, retry: int = 3) -> Optional[Dict]:
        """搜索arXiv论文"""
        max_retry = retry
        while retry > 0:
            try:
                client = arxiv.Client()
                search = arxiv.Search(query=f'ti:"{title}"', max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
                results = list(client.results(search))
                return results[0].entry_id.split("/")[-1] if results else None
            except Exception as e:
                print(f"Arxiv Error: {e}, Retry: {retry}")
                time.sleep(3 ** (max_retry - retry))
                retry -= 1
    
    def search_semantic_scholar(self, title: str, max_results: int = 3, retry: int = 5) -> Optional[Dict]:
        """使用Semantic Scholar API搜索"""
        while retry > 0:
            try:
                time.sleep(0.5)
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {'query': title, 'limit': max_results, 'fields': 'paperId,title,openAccessPdf,url,citationCount'}
                with self.session.with_api_key(self.api_key[self.times_429 % len(self.api_key)]) as temp_session:                
                    response = temp_session.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data['total'] > 0 and 'openAccessPdf' in data['data'][0]:
                    url = data['data'][0].get('openAccessPdf', {}).get('url', "")
                    arxiv_key = arxiv_pattern.findall(url)
                    if arxiv_key: return f"arxiv:{arxiv_key[-1]}"
                    return url
                return ""
            except requests.exceptions.ReadTimeout:
                print(f"Semantic Scholar Error: Read time out, Retry: {retry}")
                retry -= 1
                time.sleep(1.5)
            except Exception as e:
                if response.status_code == 429:
                    self.times_429 += 1
                    print(f"Semantic Scholar Error: 429, Retry: {retry}")
                else:
                    print(f"Semantic Scholar Error: {e}, Retry: {retry}")
                retry -= 1
                time.sleep(1.5)
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
                print(f"PDF已下载: {filepath}")
                return True
        except Exception as e:
            print(f"PDF下载错误: {e}")
        return False
    
    def comprehensive_search(self, title: str) -> Dict[str, str]:
        """综合搜索论文"""  
        # print("1. 搜索arXiv...")
        result = self.search_arxiv(title)
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
        # No arxiv link found. Try searching.
        result = self.comprehensive_search(cite['title'])
        if result['source']: return result['source'], result['url']
        else: return "", None

    def run(self):
        files = glob.glob("../crawled_papers/cs/*/citation.jsonl")
        # jobs = []
        arxiv_set, title_set = set(), set()
        info_to_title = {}
        with open("../crawled_papers/cited_papers.jsonl", "w+", encoding='utf-8') as fout, \
             open("../crawled_papers/cited_arxiv_ids.txt", "w+", encoding='utf-8') as fout_2, \
             open("../crawled_papers/error_titles.json", "w+", encoding='utf-8') as fout_3:
            for i, f in enumerate(files):
                d = load_local(f)
                arxiv_id = Path(f).parent.name
                print(f"File {i + 1} / {len(files)} - {arxiv_id}/citation.jsonl")
                for j, x in enumerate(d):
                    print(f"  Sentence {j + 1} / {len(d)}")
                    for k in x['citation']:
                        cite = x['citation'][k]
                        if "info" in cite:
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
                                        title_set.add(cite['title'])
                                        continue
                                cite['title'] = title['title']
                                info_to_title[cite['info']] = cite['title']
                            else: continue
                        if 'title' not in cite or cite['title'] in title_set: continue
                        # cite['source'] = f
                        # cite['citation_key'] = k
                        # jobs.append(cite)
                        source, url = self.get_paper_url(cite)
                        cite['source'] = source
                        if source == "Network Error":
                            fout_3.write(cite['title'] + "\n")
                        if "arXiv" in source:
                            arxiv_set.add(url)
                            cite['volume'] = url
                        elif source:
                            cite['url'] = url
                            fout.write(json.dumps({"title": cite['title'], 'url': url}) + "\n")
                        title_set.add(cite['title'])
                print_json(d, f"{f[:-6]}s-clean.jsonl")

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
                print(f"Error: {e}, Retry: {retry}")
                time.sleep(5)    


if __name__ == "__main__":
    with open("../../api_key.json") as f:
        info = json.load(f)
        keys = info['semanticscholar']['key']
        llm_keys = info['cstcloud']
    PaperDownloader(keys, llm_keys).run()
