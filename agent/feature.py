import re
import os
import math
import json
import tqdm
import asyncio
import aiofiles
import itertools
import numpy as np
import networkx as nx
from datetime import datetime
from sentence_transformers import util
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor as TPE
from sklearn.feature_extraction.text import CountVectorizer

from tools.tool_config import LLMServerInfo
from tools.utils import index_to_abstract, valid_check
from tools.sbert_client import SentenceTransformerClient
from tools.dynamic_oracle_generator import QueryExpansionLLMClient
from tools.request_utils import openalex_search_paper, URL_DOMAIN, SessionManager


RATELIMIT = asyncio.Semaphore(10)
FILELOCK = asyncio.Lock()

with open("/data/tsyu/api_key.json") as f: json_key = json.load(f)
key = {}
for k in ['cstcloud', 'deepseek']:
    for m in json_key[k]['models']:
        key[m] = {"base_url": json_key[k]['domain'], "api_key": json_key[k]['key']}

model = "gpt-oss-120b"
llm_info = LLMServerInfo(base_url=key[model]['base_url'], api_key=key[model]['api_key'], model=model)
st = SentenceTransformerClient("http://localhost:8030/encode", 32)


def normalize(a: np.ndarray) -> np.ndarray:
    norm = np.sum(a * a)
    if norm == 0: return a
    return a / np.sqrt(norm)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    b = normalize(b)
    return float(np.sum(a * b))


def get_metadata(paper: Dict[str, Any]):
    # The following information to get:
    return {
        "id": paper['id'].replace(URL_DOMAIN, ""),
        "ids": paper['ids'],
        "title": paper['display_name'],
        "locations": [x['source'] for x in paper['locations']],
        "cited_by_count": paper['cited_by_count'],
        "counts_by_year": paper['counts_by_year'],
        "publication_date": paper['publication_date'],
        "referenced_works": [x.replace(URL_DOMAIN, "") for x in paper['referenced_works']]
    }


async def search_paper_from_api(paper_title: str) -> Dict[str, Any]:
    on_target = None
    try:
        async with RATELIMIT:
            results = await openalex_search_paper("works", {"default.search": paper_title}, add_email=False)
    except Exception as e:
        return paper_title, str(e)

    for paper_info in results.get("results", []):
        if valid_check(paper_title, paper_info['display_name']): 
            paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
            on_target = paper_info
            break
        # else:
        #     print(f"{paper_title}+{paper_info['display_name']}")
    if not on_target: return paper_title, [x['title'] for x in results.get("results", [])], len(results.get("results", []))
    return {
        "metadata": get_metadata(on_target), 
        "title": paper_title,
        "abstract": index_to_abstract(on_target['abstract_inverted_index']),
    }


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
        self.register_key = {"high_ref": []}  # register which paper belongs to which query
        self.negatives = {}
        self.hard_negatives = {}
        self.eval_date = eval_date
        self.num_oracle_papers = num_oracle_papers
        self.sentence_transformer = sentence_transformer
        self.query_llm = QueryExpansionLLMClient(llm_info, {"temperature": 0.6, "max_tokens": 16384})

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
        return (self.eval_date - datetime.strptime(paper['publication_date'], "%Y-%m-%d")).days + 1

    async def _request_for_papers(self, query, uplimit):
        filter_params = {
            "default.search": query,
            "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
        }
        oracle, oracle_ids = {}, []
        page, previous_length = 1, 0
        async with RATELIMIT:
            while len(self.oracle) < uplimit:
                results = await openalex_search_paper("works", filter_params, per_page=200, page=page)
                if not results:
                    print(f"An network issue cause oracle data miss in query {query}.")
                    break
                for x in results['results']:
                    x['id'] = x['id'].replace(URL_DOMAIN, "")
                    if valid_check(self.target, x['display_name']): continue
                    if (paper_id := x['id']) not in self.oracle and paper_id not in oracle:
                        # abstract
                        x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
                        del x['abstract_inverted_index']
                        # referenced_works
                        x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
                        # store
                        oracle[paper_id] = x
                        oracle_ids.append(paper_id)
                    if len(oracle) == uplimit: break
                # {"results": []} case
                if len(oracle) == previous_length: break
                page += 1 
                previous_length = len(oracle)
        return query, oracle, oracle_ids
    
    async def _get_high_corefs(self, threshold: float = 0.15):
        corefs = {}
        for x in self.library.values():
            for y in x['referenced_works']:
                corefs[y] = corefs.get(y, 0) + 1

        # get information of high co-citations
        threshold = int(threshold * len(self.library))
        high_corefs = [x for x, y in corefs.items() if y >= threshold]
        for x in high_corefs:
            if x in self.library: self.register_key['high_ref'].append(x['id'])
        high_corefs = set(high_corefs)
        for q in self.register_key:
            self.register_key[q] = [x for x in self.register_key[q] if x not in high_corefs]
        print(f"We found {len(self.register_key['high_ref'])} high cited papers in library.")
        
        async def get_paper_by_workid(work_id):
            async with RATELIMIT:
                x = await openalex_search_paper(f"works/{work_id}")

            if not x: return
            x['id'] = x['id'].replace(URL_DOMAIN, "")
            # abstract
            x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
            del x['abstract_inverted_index']
            # referenced_works
            x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
            return x
        
        new_oracles = {}
        tasks = [asyncio.create_task(get_paper_by_workid(x)) for x in high_corefs and x not in self.library]
        for task in asyncio.as_completed(tasks):
            x = await task
            if not x or valid_check(self.target, x['display_name']): continue
            new_oracles[x['id']] = x
            self.register_key['high_ref'].append(x['id'])
        print(f"We added {len(self.register_key['high_ref'])} high cited papers.")
        return new_oracles
    
    async def _negative_sampling(self, number: int, margin_citation_count: int):
        # corefs by oracle paper
        corefs = set()
        for x in self.library.values():
            corefs += set(x['referenced_works'])
        # 2 kind of negative samples: 1. Easy negatives that has 0 local prestige
        while len(self.negatives) < number:
            async with RATELIMIT:
                results = await openalex_search_paper(
                    "works", 
                    filter={"to_publication_date": self.eval_date.strftime("%Y-%m-%d")}, 
                    do_sample=True, 
                    per_page=200
                )
            for x in results.get('results', []):
                if valid_check(self.target, x['display_name']): continue
                x['id'] = x['id'].replace(URL_DOMAIN, "")
                if all((paper_id := x['id']) not in y for y in [self.oracle, corefs, self.positive_ids, self.negatives]):
                    x = get_metadata(x)
                    # feature - no local prestige and pagerank
                    x['features'] = [x['relevance_score'], 0, 0, 0, 0, 0]
                    x['features'][2] = math.log(1 + self._citation_count_by_eval_date(x))
                    # store
                    self.negatives[paper_id] = x
                    if len(self.negatives) == number: break
            print(f"Get {len(self.negatives)} Easy Negatives")
        # 2. Hard negatives with high global prestige but 0 relevancy
        M = int(margin_citation_count * 1.5)
        while len(self.hard_negatives) < number:
            async with RATELIMIT:
                results = await openalex_search_paper(
                    "works", 
                    filter={"cited_by_count": f">{M}", "to_publication_date": self.eval_date.strftime("%Y-%m-%d")}, 
                    do_sample=True, 
                    per_page=200
                )
            for x in results.get('results', []):
                if valid_check(self.target, x['display_name']): continue
                x['id'] = x['id'].replace(URL_DOMAIN, "")
                if all((paper_id := x['id']) not in y for y in [self.oracle, corefs, self.positive_ids, self.negatives, self.hard_negatives]):
                    x = get_metadata(x)
                    # store
                    if (real_citation_count := self._citation_count_by_eval_date(x)) >= margin_citation_count:
                        # feature - no local prestige and pagerank
                        x['features'] = [x['relevance_score'], 0, 0, 0, 0, 0]
                        x['features'][2] = math.log(1 + real_citation_count)
                        self.hard_negatives[paper_id] = x
                        if len(self.hard_negatives) == number: break
            print(f"Get {len(self.hard_negatives)} Hard Negatives")

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
            title = x['display_name']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")

        for paper_id in self.negatives:
            paper_ids.append(paper_ids)
            x = self.negatives[paper_id]
            title = x['display_name']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")

        for paper_id in self.hard_negatives:
            paper_ids.append(paper_ids)
            x = self.hard_negatives[paper_id]
            title = x['display_name']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")

        sentences.append(query)
        embeddings = self.sentence_transformer.embed(sentences)
        cosine_scores = util.cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
        for paper_id, score in zip(paper_ids, cosine_scores):
            if paper_id in self.oracle:
                self.oracle[paper_id]["feature"][1] = score
            elif paper_id in self.negatives:
                self.negatives[paper_id]["feature"][1] = score
            elif paper_id in self.hard_negatives:
                self.hard_negatives[paper_id]["feature"][1] = score
    
    def _calculate_features_for_citations(self, citations: List[Dict[str, Any]], query: str):
        """
        The format of param citations is:
        [{
            "metadata": metadata dict
            "title": "title",
            "abstract": "abstract", 
        }, ...]
        """
        # preparation
        num_oracles, num_citations = len(self.oracle), len(citations)
        self.topn = min(num_citations, num_oracles)
        max_citations, max_local_citations = 0, 0
        for x in self.oracle.values:
            max_citations = max(max_citations, x['features'][1])
            max_local_citations = max(max_local_citations, x['features'][2])
        max_citations = math.log(1 + max_citations)
        max_local_citations = math.log(1 + max_local_citations)
        not_oracle_papers, not_oracle_paper_abstract = [], []
        # normalization
        for i, citation in enumerate(citations):
            metadata = citation['metadata']
            j = metadata['id']
            if j in self.oracle: 
                # The cited paper is in oracle papers
                feature = self.oracle[j]['features']
            else:
                # Not in
                feature = [0, 0, 0, 0, 0]
                not_oracle_papers.append(i)
                title = metadata['title'] if metadata['title'] else citation['title']
                not_oracle_paper_abstract.append(f"{title}. {citation['abstract']}" if citation['abstract'] else title)
                # feature 2: global citations
                feature[1] = math.log(1 + self._citation_count_by_eval_date(metadata))
                if max_citations > 0: feature[1] /= max_citations
                # local co-citation: calcultate number of oracle papers cites this paper
                # local_pagerank = average(pageranks of oracle papers cites this paper)
                ids_set = set(metadata['id'])
                for n in self.citation_graph:
                    if set(ids_set) - set(self.citation_graph[n]):
                        feature[2] += 1
                        feature[3] += self.oracle[n]['features'][4]
                if feature[2] > 0: feature[3] /= feature[2]
                feature[2] = math.log(1 + feature[2])
                if max_local_citations > 0: feature[2] /= max_local_citations
                # feature 5: 
                feature[4] = feature[1] / math.log(1 + self._paper_age(metadata))
            citation['features'] = feature
        print(f"We have {len(not_oracle_papers)} not oracle references out of {num_citations} references")

        # embed together
        if not_oracle_papers:
            not_oracle_paper_abstract.append(query)
            not_oracle_paper_embeddings = self.sentence_transformer.embed(not_oracle_paper_abstract)
            not_oracle_cossims = cos_sim(not_oracle_paper_embeddings[-1:], not_oracle_paper_embeddings[:-1])[0].tolist()
            for i, s in zip(not_oracle_papers, not_oracle_cossims):
                citations[i]['features'][0] = s
        return citations
    
    async def _collect_oracles(self, query: str):
        """
            Get self.num_oracle_papers selected queries. 
        """
        def _average_cut(values: list, target: int):
            min_size = min(values)
            if min_size * len(values) >= target:
                result = [target // len(values) for _ in values]
                for i in range(target - sum(result)): result[i] += 1
                return result
            
            result = [min_size for _ in values]
            for i in range(len(values)): values[i] -= min_size
            remain_idx = [i for i, l in values if l > 0]
            values = [v for v in values if v > 0]
            add = _average_cut(values, target - sum(result))
            for i, v in zip(remain_idx, add): result[i] += v
            return result
            
        prev_queries = []
        while len(self.oracle) < self.num_oracle_papers:
            # 1. fetch 5 queries
            queries = await self.query_llm.call(inputs={"query": query, "prev_queries": prev_queries})
            if not (queries := [q for q in queries if q not in prev_queries]): continue
            prev_queries.extend(queries)
            print(f"Get {len(queries)} queries: {queries}")
            # 2 request papers
            tasks = [self._request_for_papers(q, self.num_oracle_papers) for q in queries]
            for task in asyncio.as_completed(tasks):
                try:
                    query, oracle, oracle_ids = await task
                    if oracle:
                        self.register_key[query] = oracle_ids
                        for k in oracle:
                            if k not in self.library:
                                self.library[k] = oracle[k]
                    print(f"Get {len(self.library)} papers in library by query {query}")
                except Exception as e:
                    print(f"Query {query} has an {e}")              
            # 3 get high_refs
            high_refs = await self._get_high_corefs()
            self.library.update(high_refs)
            # 4 select oracle papers
            if len(self.library) >= self.num_oracle_papers:
                # fill in self.library
                selected = self.register_key['high_ref']
                print(f"The distribution of oracle paper is: high cited papers {len(selected)}", end=" ")
                lengths = [len(self.register_key[q]) for q in prev_queries]
                amount = _average_cut(lengths, self.num_oracle_papers - len(selected))
                for q, a in zip(prev_queries, amount):
                    selected.extend(self.register_key[q][:a])
                    print(f"{a} papers from query {q}", end=" ")
                print()
                assert len(selected) == self.num_oracle_papers, (len(selected), self.num_oracle_papers)
    
    async def run(self, query: Dict[str, Any], metadata_map: Dict[str, Any]) -> dict:
        """
        metadata_map: title(re.sub) -> paper metadata
        """
        print(f"DynamicOracleGenerator::Request for oracle paper with query {query['query']}")
        avail_references, citations = set(), []
        for x in query['references']:
            x = re.sub(r"[:,.!?&]", "", x)
            if x in metadata_map and x not in avail_references:
                avail_references.add(x)
                citations.append(metadata_map[x])
        print(f"Get {len(avail_references)} avail positive examples from total {len(query['references'])} references")
        if not avail_references: return
        self.target = query['title']
        self.positive_ids = [x['metadata']['id'] for x in citations]
        
        # 1. Request for papers
        await self._collect_oracles(query['query'])
        # 2. Get post-calculate features
        print(f"DynamicOracleGenerator::Get features 1 3 4")
        # 2.1 sentence transformer cosine similarity
        self._calculate_similarity(query['query'])
        for x in self.oracle.values():
            x['features'] = [0, 0, 0, 0, 0]
            x['features'][1] = self._citation_count_by_eval_date(x)
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        self._local_citation_and_pagerank()
        print(f"DynamicOracleGenerator::Negative Sampling")
        # negative sampling
        mid_citation_count = sorted([x['features'][1] for x in self.oracle.values()])[len(self.oracle) // 2]
        await self._negative_sampling(len(avail_references), mid_citation_count)
        # regularization
        max_citation_count = math.log(1 + max(x['features'][1] for x in self.oracle.values()))
        max_local_citation_count = math.log(1 + max(x['features'][2] for x in self.oracle.values()))
        for x in itertools.chain(self.oracle.values(), self.negatives.values(), self.hard_negatives.values()):
            if max_citation_count > 0:
                x['features'][1] = math.log(1 + x['features'][1]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][2] = math.log(1 + x['features'][2]) / max_local_citation_count
            x['features'][4] = x['features'][1] / math.log(1 + self._paper_age(x))
        print("DynamicOracleGenerator::Calculate Citation Feature")  
        citations = self._calculate_features_for_citations(citations, query['query'])
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
    print(f"DynamicOracleGenerator::Get Citation Information {len(pending_jobs)}")
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
            "title": paper['display_name'],
            "cited_by_count": paper['cited_by_count'],
            "counts_by_year": paper['counts_by_year'],
            "publication_date": paper['publication_date'],
            "features": paper['features'] if 'features' in paper else None
        }
    
    engine = DynamicOracleGenerator(1000, st, datetime.strptime(query['date'], "%Y-%m-%d"))
    citations = await engine.run(query, metadata_map)
    citations = [{"id": p['metadata']['id'], "title": p['title'], "features": p['features']} for p in citations]
    oracle = {k: _get_metadata(v) for k, v in engine.oracle.items()}
    negatives = {k: _get_metadata(v) for k, v in engine.negatives.items()}
    hard_negatives = {k: _get_metadata(v) for k, v in engine.hard_negatives.items()}
    return {"citations": citations, "oracle": oracle, "easy_negatives": negatives, "hard_negatives": hard_negatives}


async def main():
    try:
        await SessionManager.init()
        metadatas = load_local("metadata.jsonl")
        metadata_map = {x['title']: x for x in metadatas}
        queries = load_local("queries.jsonl")[:1]
        tasks = [asyncio.create_task(get_oracle(q, metadata_map)) for q in queries]
        for task in asyncio.as_completed(tasks):
            result = await task
            if result:
                async with FILELOCK:
                    async with aiofiles.open("results.jsonl", 'a+', encoding='utf-8') as f: 
                        await f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                print("No result")
    finally:
        await SessionManager.close()


async def test():
    try:
        await SessionManager.init()
        y = "Influence of Bisphenol A on Type 2 Diabetes Mellitus"
        y = re.sub(r"[:,.!?&]", "", y)
        print(y)
        results = await search_paper_from_api(y)
        print(results)
    finally:
        await SessionManager.close()    


if __name__ == "__main__":
    asyncio.run(main())