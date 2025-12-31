import re
import os
import math
import json
import tqdm
import logging
import asyncio
import aiofiles
import itertools
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any
from sentence_transformers.util import cos_sim
from dateutil.relativedelta import relativedelta

from tools.to_openalex import to_openalex
from tools.tool_config import LLMServerInfo
from tools.utils import valid_check, clean_token
from tools.sbert_client import SentenceTransformerClient
from tools.dynamic_oracle_generator import QueryExpansionLLMClient
from tools.request_utils import openalex_search_paper, OPENALEX_SELECT, SessionManager


debug = False
output = "tests.jsonl" if debug else "final.jsonl"
logging.basicConfig(
    filename="test.log" if debug else "feature.log", 
    level=logging.INFO, 
    format="%(asctime)s-%(levelname)s-%(message)s",
)
FILELOCK = asyncio.Lock()
TASK_LEVEL_SEMAPHORE = asyncio.Semaphore(7)

with open("/data/tsyu/api_key.json") as f: json_key = json.load(f)
key = {}
for k in ['cstcloud', 'deepseek']:
    for m in json_key[k]['models']:
        key[m] = {"base_url": json_key[k]['domain'], "api_key": json_key[k]['key']}

model = "gpt-oss-120b"
llm_info = LLMServerInfo(base_url=key[model]['base_url'], api_key=key[model]['api_key'], model=model)
st = SentenceTransformerClient("http://172.18.36.90:8030/encode", 32)


async def search_paper_from_api(paper_title: str) -> tuple[str, dict]:
    try:
        results = await openalex_search_paper("works", {"title.search": clean_token(paper_title)}, add_email=False)
    except Exception as e:
        return paper_title, str(e)

    for x in results['results']:
        if valid_check(paper_title, x['title']): 
            return paper_title, x
    return paper_title, None


class DynamicOracleGenerator:
    
    def __init__(
            self, 
            num_oracle_papers: int, 
            sentence_transformer: SentenceTransformerClient,
            eval_date: datetime = datetime.now(),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.oracle = {}
        self.library = {}  # All papers searched by oracle
        self.eval_date = eval_date
        self.num_oracle_papers = num_oracle_papers
        self.sentence_transformer = sentence_transformer
        self.query_llm = QueryExpansionLLMClient(llm_info, {"temperature": 0, "max_tokens": 16384})

    def _citation_count_by_eval_date(self, paper: dict):
        """
        Calculate the cited by count on evaluation date.
        """
        eval_year = int(self.eval_date.year)
        if (citation_count := paper.get("cited_by_count", 0)):
            for x in paper.get("counts_by_year", []):
                if x['year'] > eval_year:
                    citation_count -= x['cited_by_count']
            return citation_count
        citation_count = 0
        for x in paper.get("counts_by_year", []):
            if x['year'] <= eval_year:
                citation_count += x['cited_by_count']
        return citation_count

    def _paper_age(self, paper):
        year = int(self.eval_date.strftime("%Y-%m-%d")[:4]) - int(paper['publication_date'][:4])
        assert year >= 0, (self.eval_date, paper['publication_date'])
        return year + 1

    async def _request_for_papers(self, query, uplimit) -> List[Dict[str, Any]]:
        filter_params = {
            "default.search": to_openalex(query),
            "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
        }
        select = f"{OPENALEX_SELECT},relevance_score"
        oracle = []
        # first query
        for page in range(1, (uplimit - 1) // 200 + 2):
            try:
                results = await openalex_search_paper("works", filter_params, per_page=200, select=select, page=page)
            except Exception as e:
                logging.info(f"An {e} cause oracle data miss in query {query}.")
                return []
            uplimit = results['count']
            oracle.extend(results['results'])

        return oracle
    
    async def _get_high_refs(self, num_high_refs: int = 100) -> list:
        corefs = {}
        for x in self.library.values():
            for y in x['referenced_works']:
                corefs[y] = corefs.get(y, 0) + 1

        # get information of top local-cited papers
        new_oracles = []
        high_refs = sorted(list(corefs.items()), key=lambda x: x[1], reverse=True)[:num_high_refs]
        high_refs_to_search = set()
        for x, _ in high_refs:
            if x in self.library: 
                self.library[x]['query'] = 'high+ref'
                new_oracles.append(x)
            else:
                high_refs_to_search.add(x)
        if debug:
            logging.info(f"{len(new_oracles)} high_refs already collected and {len(high_refs_to_search)} high_refs to be requested")
        
        if high_refs_to_search:
            try:
                results = await openalex_search_paper("works", filter={"openalex": "|".join(high_refs_to_search)}, per_page=200)
                for x in results['results']:
                    if valid_check(self.target, x['title']): continue
                    if not x['publication_date'] or self.eval_date.strftime("%Y-%M-%d") < x['publication_date']: continue
                    x['query'] = 'high+ref'
                    self.library[x['id']] = x    
                    new_oracles.append(x['id'])                
            except Exception as e:
                logging.error(f"get_high_refs {e}")
        return new_oracles
    
    async def _get_neighbors(self, prev_queries: List[str], query: str, num_top_seeds: int, batch_size: int = 50) -> list:

        async def _get_cites(work_ids, keyword, cited_by_threshold: int = 0) -> dict:
            paper_count = -1
            filters = {keyword: '|'.join(work_ids), "to_publication_date": self.eval_date.strftime("%Y-%m-%d")}
            if cited_by_threshold > 0:
                filters['cited_by_count'] = f">{cited_by_threshold}"
            papers = {}
            for page in range(1, 51):
                try:                  
                    results = await openalex_search_paper("works", filters, per_page=200, page=page, sort="cited_by_count")
                    if paper_count == -1: paper_count = results['count']
                    for x in results['results']:
                        if valid_check(self.target, x['title']): continue
                        papers[x['id']] = x
                except Exception as e:
                    logging.info(f"_request_papers {e}") 
                if paper_count >= 0 and page * 200 >= paper_count: break
            return papers
            
        # 1. 用self.register_key选出每个子领域的TOP论文; 用相似性选出跟query最像的论文
        paper_ids, sentences = [], []
        for q in prev_queries:
            for x, _ in self.register_key[q]:
                paper_ids.append(x)
                x = self.library[x]
                title = x['title']
                abstract = x.get("abstract", None)
                sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")
        sentences.append(query)
        if len(sentences) > num_top_seeds:
            embeddings = self.sentence_transformer.embed(sentences)
            cosine_scores = cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
            top_papers = sorted([(i, s) for i, s in zip(paper_ids, cosine_scores)], key=lambda x: x[1], reverse=True)[:num_top_seeds]
            top_papers = [x for x, _ in top_papers]
        else:
            top_papers = paper_ids
        # 2. 获取他们的邻居的metadata
        if debug:
            logging.info(f"query {query.lower()} Top papers {len(top_papers)}")
        # as_batch
        citation_hub = {}
        for i in range(0, len(top_papers), batch_size):
            citation_hub |= await _get_cites(top_papers[i:i + batch_size], "cites", 10)
            citation_hub |= await _get_cites(top_papers[i:i + batch_size], "cited_by", 10)
        if debug:
            logging.info(f"query {query.lower()} Get {len(citation_hub)} neighbors.")

        # 3. 统计同时出现在多个metadata的论文。由于batch请求自带去重，所以要手动计算。
        neighbors_count = {}
        top_papers = set(top_papers)
        # 对应cited_by：top_papers -> citation_hub
        for x in top_papers:
            for y in self.library[x]['referenced_works']:
                if y in citation_hub:
                    neighbors_count[y] = neighbors_count.get(y, 0) + 1
        # 对应cites：citation_hub -> top_papers
        for pid, x in citation_hub.items():
            for y in x['referenced_works']:
                if y in top_papers:
                    neighbors_count[pid] = neighbors_count.get(pid, 0) + 1
        
        # --- CRITICAL FIX STARTS HERE ---        
        # Separate neighbors into "High Confidence" (Co-cited) and "Long Tail" (Single citation)
        high_confidence, longtail_ids = [], []
        
        for x, count in neighbors_count.items():
            if count >= 2:
                # Structural Priority: These are connected to multiple seeds.
                # Boost score significantly so they appear at the top.
                score = count * 100 + math.log1p(self._citation_count_by_eval_date(citation_hub[x]))
                high_confidence.append((x, score))
                if debug: citation_hub[x]['score'] = score
            else:
                longtail_ids.append(x)
            if debug: citation_hub[x]['neighbors_count'] = count
        
        high_confidence.sort(key=lambda x: x[1], reverse=True)
        final_results = []
        # Filter existing library
        for x, _ in high_confidence:
            if x not in self.library:
                self.library[x] = citation_hub[x]
            final_results.append(x)

        # if not enough, process long tails
        longtail_results, sentences = [], []
        for paper_id in longtail_ids:
            x = citation_hub[paper_id]
            title = x['title']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")

        if sentences:
            sentences.append(query)
            embeddings = self.sentence_transformer.embed(sentences)
            cosine_scores = cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
            for x, sim in zip(longtail_ids, cosine_scores):
                global_cites = math.log1p(self._citation_count_by_eval_date(citation_hub[x]))
                final_score = (sim * 10) + (global_cites * 0.1) 
                longtail_results.append((x, final_score))
                if debug:
                    citation_hub[x]['sim'] = sim
                    citation_hub[x]['score'] = final_score
    
        longtail_results.sort(key=lambda x: x[1], reverse=True)
        for x, _ in longtail_results:
            if x not in self.library:
                self.library[x] = citation_hub[x]
            final_results.append(x)

        if debug:
            logging.info(f"query {query.lower()} Neighbors: {len(high_confidence)} co-cited, {len(longtail_results)} semantic rescue.")
        
        return final_results

    async def _collect_oracles(self, query: str):
        """
            Get self.num_oracle_papers selected queries. 
        """
        targets = {"high_ref": 50, "top_seed": 50, "total": self.num_oracle_papers}
        prev_queries, oracles = [], []  # 以便后面按顺序取
        prev_length = 0
        papers_for_each_query = 50
        while len(self.oracle) < self.num_oracle_papers:
            self.register_key = {}  # register which paper belongs to which query
            # 1. fetch 5 queries
            inputs = {"query": query, "prev_query": ("".join([f"\n   - {q}" for q in prev_queries[1:]])) if prev_queries else "No"}
            try:
                queries = await self.query_llm.call(inputs=inputs)
            except Exception as e:
                logging.info(f"QueryExpansion {e}")
                queries = []
            if not prev_queries: queries = [query] + queries
            logging.info(f"Get {len(queries)} queries: {queries}")
            if not (queries := [q for q in queries if q not in prev_queries]): continue
            # 2 request papers 要串行
            for q in queries:                
                try:
                    oracle = await self._request_for_papers(q, papers_for_each_query)
                    if oracle:
                        prev_queries.append(q)
                        oracles.append(oracle)  # oracles中的顺序应该与prev_queries中的query一一对应
                        for k in oracle:
                            if k['id'] not in self.library:
                                self.library[k['id']] = k
                except Exception as e:
                    logging.info(f"Query {q} has an {e}")
            # register paper to query
            register_key_reverse_map = {}
            for q, oracle in zip(prev_queries, oracles):
                for x in oracle:
                    if x['id'] in register_key_reverse_map:
                        # compare openalex relevance score to decide which query it belongs to
                        if x['relevance_score'] > register_key_reverse_map[x['id']]['relevance']:
                            register_key_reverse_map[x['id']]['query'] = q
                            register_key_reverse_map[x['id']]['relevance'] = x['relevance_score']
                    else:
                        # new paper
                        register_key_reverse_map[x['id']] = {'query': q, "relevance": x['relevance_score']}
            for q in prev_queries: self.register_key[q] = []
            for i, x in register_key_reverse_map.items():
                self.register_key[x['query']].append((i, x['relevance']))
            for q in prev_queries: 
                self.register_key[q] = sorted(self.register_key[q], key=lambda x: x[1], reverse=True)  
            if debug: 
                stat = {q: len(v) for q, v in self.register_key.items()}
                logging.info(f"query {query.lower()} Get papers distribution: {stat}, library_size: {len(self.library)}")
            # 3 get neighbors
            neighbors = await self._get_neighbors(prev_queries, query, targets['top_seed'])
            # 4 get high_ref foundations
            high_refs = await self._get_high_refs(targets['high_ref'])
            if debug:
                logging.info(f"query {query.lower()} Has {len(self.library)} papers in library.")    
            # 5 先检查这一轮一共找到多少不重复文献。若数量足够就组装并返回，否则重新循环。
            if len(self.library) >= self.num_oracle_papers:
                # 5.1 先选择high_corefs。
                selected = set(high_refs)
                if debug:
                    logging.info(f"query {query.lower()} Has {len(selected)} high_refs")

                # 5.2 queries_per_paper降到50后，可以直接包括所有搜到的论文。
                for q in self.register_key:
                    for x, _ in self.register_key[q]:
                        if x not in selected:
                            selected.add(x)
                            self.library[x]['query'] = q

                # 5.3 最后用neighbors填满。
                remain_target = self.num_oracle_papers - len(selected)
                if debug:
                    logging.info(f"query {query.lower()} Has {remain_target} papers remain")
                
                # 处理特殊的情况：邻居太少，query papers太多，常见于remain target先变成了负数的情况。
                if 2 * remain_target < self.num_oracle_papers - len(high_refs):
                    # 此时，令query和neighbors各占一半。做法是大重排。
                    max_num_neighbors = max(remain_target, 0) + len([x for x in neighbors if x not in selected])  # 一共可以有这么多neighbors
                    max_num_queries = self.num_oracle_papers - len(high_refs) - remain_target
                    num_neighbors = num_queries = (self.num_oracle_papers - len(high_refs)) // 2
                    if num_neighbors + num_queries + len(high_refs) != self.num_oracle_papers:
                        num_neighbors += 1
                    if max_num_neighbors < num_neighbors:
                        num_queries += (num_neighbors - max_num_neighbors)
                        num_neighbors = max_num_neighbors
                    elif max_num_queries < num_queries:
                        num_neighbors += (num_queries - max_num_queries)
                        num_queries = max_num_queries
                    
                    # 按照顺序先queries后neighbors
                    if num_queries + len(high_refs) < len(selected):
                        selected = set(high_refs)
                        paper_ids, sentences = [], []
                        for q in self.register_key:
                            for x, _ in self.register_key[q]:
                                if x not in selected:
                                    paper_ids.append(x)
                                    title = self.library[x]['title']
                                    abstract = self.library[x].get("abstract", None)
                                    sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")
                        sentences.append(query)
                        embeddings = self.sentence_transformer.embed(sentences)
                        cosine_scores = cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
                        sorted_query_papers = sorted(zip(paper_ids, cosine_scores), key=lambda x: x[1], reverse=True)
                        selected.update(x for x, _ in sorted_query_papers[:num_queries])

                    for x in neighbors:
                        if num_neighbors <= 0: break
                        if x not in selected:
                            selected.add(x)
                            num_neighbors -= 1
                            self.library[x]['query'] = "co+neighbor"
                else:
                    # 不需要修改query论文的数量。
                    for x in neighbors:
                        if remain_target <= 0: break
                        if x not in selected:
                            selected.add(x)
                            remain_target -= 1
                            self.library[x]['query'] = "co+neighbor"
                    
                for x in selected:
                    self.oracle[x] = self.library[x]
                assert len(self.oracle) == self.num_oracle_papers, (len(self.oracle), self.num_oracle_papers)
            if prev_length == len(self.library): 
                raise ValueError("No new papers. Perhaps you hit the OpenAlex rate limit.")
            prev_length = len(self.library)

    def _local_citation_and_pagerank(self):
        """
        Calculates local citation counts and PageRank for a list of papers.
        Modifies the dictionary in-place.
        """
        # 1. Create a lookup map and a set of valid IDs
        # This allows O(1) access to paper objects and quick existence checks
        # Initialize local_citation_count to 0 for all papers
        paper_map = {}
        for paper_id in self.oracle:  # paper_id = (title, first author)
            x = self.oracle[paper_id]
            paper_map[paper_id] = {"references": x['referenced_works'], "local_citation_count": 0}

        valid_ids = set(paper_map.keys())  # valid_id = set((title, first author))

        # 2. Construct the Graph
        # A citation network is a Directed Graph (DiGraph)
        # Direction: Source Paper -> Cites -> Target Paper
        citation_graph = nx.DiGraph()
        
        # Add all nodes first (to ensure papers with 0 links are included)
        citation_graph.add_nodes_from(valid_ids)

        # Iterate through papers to build edges and count citations
        for source_id, paper_data in paper_map.items():
            citations = paper_data['references']
            
            for target_id in citations:  # target_id = workid
                # strictly filter for 'local' papers only
                if target_id in self.oracle:
                    # Add Edge to Graph
                    citation_graph.add_edge(source_id, target_id)                    
                    # Increment Local Citation Count
                    # Note: If A cites B, B gets the citation count
                    paper_map[target_id]['local_citation_count'] += 1

        self.citation_graph = nx.to_dict_of_lists(citation_graph)

        # 3. Calculate PageRank
        # alpha=0.85 is the standard damping factor used by Google
        pagerank_scores = nx.pagerank(citation_graph, alpha=0.85)

        # 4. Update the original list with PageRank scores
        for paper_id, score in pagerank_scores.items():
            paper_map[paper_id]['local_pagerank'] = score
            self.oracle[paper_id]['features'][2] = paper_map[paper_id]['local_citation_count']
            self.oracle[paper_id]['features'][3] = score

    def _calculate_similarity(self, query: str):
        # vector similarity query and title & abstract
        sentences, paper_ids = [], []

        for paper_id in self.oracle:
            paper_ids.append(paper_id)
            x = self.oracle[paper_id]
            title = x['title']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")

        sentences.append(query)
        embeddings = self.sentence_transformer.embed(sentences)
        cosine_scores = cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
        for paper_id, score in zip(paper_ids, cosine_scores):
            self.oracle[paper_id]["features"][0] = score
    
    def _calculate_features_for_citations(self, citations: List[Dict[str, Any]], query: str, normalize_params: List[float]):
        """
        The format of param citations is:
        [{
            "metadata": metadata dict
            "title": "title",
            "abstract": "abstract", 
        }, ...]
        """
        # preparation
        self.citation_graph = {k: set(v) for k, v in self.citation_graph.items()}
        # normalization
        for citation in citations:
            j = citation['id']
            if j in self.oracle: 
                # The cited paper is in oracle papers
                feature = self.oracle[j]['features']
                citation['query'] = self.oracle[j]['query']
            citation['features'] = feature
        citations = [x for x in citations if 'query' in citation]
        logging.info(f"For query {query.lower()}, We have {len(citation)} oracle references")
        return citations
    
    async def run(self, query: Dict[str, Any], metadata_map: Dict[str, Any]) -> dict:
        """
        metadata_map: title(re.sub) -> paper metadata
        """
        avail_references, citations = set(), []
        for x in query['references']:
            if x in metadata_map and x not in avail_references:
                avail_references.add(x)
                citations.append(metadata_map[x])
        if debug:
            logging.info(f"query {query['query'].lower()} Get {len(avail_references)} avail positive examples")
        if not avail_references: return []
        self.target = query['title']
        self.positive_ids = [x['id'] for x in citations]

        # 1. Request for papers
        await self._collect_oracles(query['query'])
        # 2. Get post-calculate features
        # 2.1 sentence transformer cosine similarity
        for x in self.oracle.values():
            x['features'] = [0, 0, 0, 0, 0]
            x['features'][1] = self._citation_count_by_eval_date(x)
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        self._local_citation_and_pagerank()
        # negative sampling
        self._calculate_similarity(query['query'])
        # regularization
        max_citation_count = math.log1p(max(x['features'][1] for x in self.oracle.values()))
        max_local_citation_count = math.log1p(max(x['features'][2] for x in self.oracle.values()))
        # z-score regularization
        normalize_params = {"max_citations": max_citation_count, "max_local_citations": max_local_citation_count}
        citations = self._calculate_features_for_citations(citations, query['query'], normalize_params)
        for x in itertools.chain(self.oracle.values(), self.negatives.values(), self.hard_negatives.values()):
            if max_citation_count > 0:
                x['features'][1] = math.log1p(x['features'][1]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][2] = math.log1p(x['features'][2]) / max_local_citation_count
            x['features'][4] = x['features'][1] / math.log1p(self._paper_age(x))
        return citations


def load_local(fn):
    with open(fn, "r+", encoding="utf-8") as f:
        d = [json.loads(line.strip()) for line in f if line.strip()]
    return d


def print_json(d, fn):
    with open(fn, "w+", encoding="utf-8") as f:
        for x in d:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


async def get_metas():
    if os.path.exists("metadata.jsonl"):
        metadatas = load_local("metadata.jsonl")
        for x in metadatas:
            if 'key' in x:
                assert x['key'] == x['title']
        metadata_map = {x['title']: x for x in metadatas}
    else:
        metadata_map = {}
    if os.path.exists("missing.txt"): os.remove("missing.txt")
    queries = load_local("queries.jsonl")
    # 0. Get Citation Information
    pending_jobs = []
    for x in queries:
        pending_jobs.extend([re.sub(r"[:,.!?&]", "", y) for y in x['references']])
    pending_jobs = list(set(x for x in pending_jobs if x not in metadata_map))
    tasks = [asyncio.create_task(search_paper_from_api(x)) for x in pending_jobs]
    for task in tqdm.tqdm(asyncio.as_completed(tasks)):
        try:
            info = await task
            metadata_map[info['title']] = info
            async with FILELOCK:
                async with aiofiles.open("metadata.jsonl", "a+") as f:
                    await f.write(json.dumps(info, ensure_ascii=False) + "\n")
        except Exception as e:
            with open("missing.txt", "a+", encoding='utf-8') as f: f.write(f"{info}\n")


async def get_oracle(query, metadata_map):
    def _get_metadata(paper: Dict[str, Any]):
        # The following information to get:
        return {
            "id": paper['id'],
            "title": paper['title'],
            "cited_by_count": paper['cited_by_count'],
            "counts_by_year": paper['counts_by_year'],
            "publication_date": paper['publication_date'],
            "query": paper.get('query', None),
            "features": paper.get('features', None),
            "neighbors": paper.get('neighbors_count', None),
            "score": paper.get('score', None),
            "sim": paper.get('sim', None),
        }
    
    engine = DynamicOracleGenerator(1000, st, datetime.strptime(query['date'], "%Y-%m-%d") + relativedelta(years=1))
    async with TASK_LEVEL_SEMAPHORE:
        citations = await engine.run(query, metadata_map)
    citations = [{"id": p['id'], "title": p['title'], "features": p['features'], "query": p['query']} for p in citations]
    oracle = {k: _get_metadata(v) for k, v in engine.oracle.items()}
    if debug:
        database = {k: _get_metadata(v) for k, v in engine.library.items()}
        return {"query": query['query'], 'title': query['title'], "citations": citations, "oracle": oracle, "database": database}
    return {"query": query['query'], 'title': query['title'], "citations": citations, "oracle": oracle}


async def main():
    try:
        await SessionManager.init()
        metadatas = load_local("metadata.jsonl")
        metadata_map = {x['id']: x for x in metadatas}
        queries = load_local("rests.jsonl")[2:1000]
        tasks = [asyncio.create_task(get_oracle(q, metadata_map)) for q in queries]
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):            
            try:
                result = await task
                if result:
                    async with FILELOCK:
                        async with aiofiles.open(output, 'a+', encoding='utf-8') as f: 
                            await f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    logging.info("No result")
            except asyncio.exceptions.CancelledError:
                pass
            except Exception as e:
                logging.error(f"result error {e}")
    finally:
        await SessionManager.close()


async def search_a_survey(query: dict):
    _, survey_meta = await search_paper_from_api(clean_token(query['title']))
    if not survey_meta: return query, "no match", None
    if not survey_meta['referenced_works']: return query, "no ref", None
    if not survey_meta['publication_date']: 
        survey_meta['publication_date'] = query['date']
        if not survey_meta['publication_date']: return query, "no date", None
    # get reference information
    batch_size = 100
    tasks = []
    for i in range(0, len(survey_meta['referenced_works']), batch_size):
        batch = survey_meta['referenced_works'][i:i + batch_size]
        task = openalex_search_paper("works", filter={"openalex": "|".join(batch)}, per_page=100)
        tasks.append(asyncio.create_task(task))
    logging.info(f"{len(survey_meta['referenced_works'])} golden references for survey {query['title']}")
        
    metadata = {}
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
        except Exception as e:
            continue

        for x in result['results']:
            metadata[x['id']] = x

    return query, survey_meta, metadata


async def dataset_relabel():
    metadatas = {}
    try:
        await SessionManager.init()
        queries = load_local("rest.jsonl")
        tasks = [asyncio.create_task(search_a_survey(x)) for x in queries]
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                query, survey, papers = await task
                if isinstance(survey, dict):
                    metadatas |= papers
                    query['id'] = survey['id']
                    query['date'] = survey['publication_date']
                    query['references'] = survey['referenced_works']
                    async with FILELOCK:
                        async with aiofiles.open("requeries.jsonl", "a+") as f:
                            await f.write(json.dumps(query, ensure_ascii=False) + "\n")
                else:
                    logging.error(f"{query['title']} {survey}")
            except KeyError:
                raise
            except Exception as e:
                logging.error(f"{e}")
        logging.info(f"Collected {len(metadatas)} References")
        print_json(list(metadatas.values()), "metadata.jsonl")
    finally:
        await SessionManager.close()


async def get_redirect():   
    try:
        await SessionManager.init()
        with open("rest.json") as f:
            works = json.load(f)
        tasks = []
        for i in works:
            task = openalex_search_paper(f"works/{i}")
            tasks.append(asyncio.create_task(task))

        redirect = {}
        for work, task in tqdm.tqdm(zip(works, asyncio.as_completed(tasks)), total=len(tasks)):
            try:
                x = await task
            except Exception as e:
                logging.error(f"Get metadata {e}")
                continue

            if x:
                redirect[work] = x['id']
                async with FILELOCK:
                    async with aiofiles.open("metadata1.jsonl", "a+") as f:
                        await f.write(json.dumps(x, ensure_ascii=False) + "\n")
        logging.info(f"We have {len(redirect)} redirects")
        with open("redirect.json", "w+") as f:
            json.dump(redirect, f, indent=2)
    finally:
        await SessionManager.close()


async def test():
    try:
        await SessionManager.init()
        y = "Detecting and Managing Mental Health Issues within Young Adults. A Systematic Review on College Counselling in Italy"
        logging.info(clean_token(y))
        results = await search_paper_from_api(y)
        logging.info(results)
    finally:
        await SessionManager.close()    


if __name__ == "__main__":
    # TODO: 删除所有外面的async with ratelimit，都放到openalex_search_paper中。
    # TODO: 删除所有openalex_search_paper之后get_metadata的过程。
    # TODO: pass normalized params to SourceCritic,传入oracle_data即可。
    asyncio.run(main())