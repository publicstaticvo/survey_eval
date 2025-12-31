import re
import math
import json
import logging
import asyncio
import lightgbm
import numpy as np
import networkx as nx
from bertopic import BERTopic
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import CountVectorizer

from .request_utils import AsyncLLMClient, openalex_search_paper, OPENALEX_SELECT, URL_DOMAIN, RateLimit
from .prompts import QUERY_EXPANSION_PROMPT
from .utils import index_to_abstract, extract_json
from .sbert_client import SentenceTransformerClient
from .to_openalex import to_openalex
from .tool_config import ToolConfig

debug = False
    

class QueryExpansionLLMClient(AsyncLLMClient):
    PROMPT: str = QUERY_EXPANSION_PROMPT
    def _availability(self, response):
        queries = extract_json(response)        
        return queries['queries']


class DynamicOracleGenerator:
    
    def __init__(self, config: ToolConfig):
        self.oracle = {}
        self.library = {}  # All papers searched by oracle
        self.register_key = {}
        self.letor = lightgbm.Booster(model_file=config.letor_path)
        self.eval_date = config.evaluation_date
        self.query_llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.num_oracle_papers = config.num_oracle_papers
        self.sentence_transformer = SentenceTransformerClient(config.sbert_server_url)

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
                new_oracles.append(x)
            else:
                high_refs_to_search.add(x)
        if debug:
            logging.info(f"{len(new_oracles)} high_refs already collected and {len(high_refs_to_search)} high_refs to be requested")
        
        if high_refs_to_search:
            try:
                results = await openalex_search_paper("works", filter={"openalex": "|".join(high_refs_to_search)}, per_page=200)
                for x in results['results']:
                    if not x['publication_date'] or self.eval_date.strftime("%Y-%M-%d") < x['publication_date']: continue
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
                    for x in results['results']: papers[x['id']] = x
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
        for paper_id in self.oracle:
            x = self.oracle[paper_id]
            paper_map[paper_id] = {"references": x['referenced_works'], "local_citation_count": 0}

        valid_ids = set(paper_map.keys())

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

        # 3. Calculate PageRank
        # alpha=0.85 is the standard damping factor used by Google
        pagerank_scores = nx.pagerank(citation_graph, alpha=0.85)

        # 4. Update the original list with PageRank scores
        for paper_id, score in pagerank_scores.items():
            paper_map[paper_id]['local_pagerank'] = score
            self.oracle[paper_id]['features'][2] = paper_map[paper_id]['local_citation_count']
            self.oracle[paper_id]['features'][3] = score

        return nx.to_dict_of_lists(citation_graph)

    def _calculate_similarity(self, query: str):
        # vector similarity query and title & abstract
        sentences = []
        for paper_id in self.oracle:
            x = self.oracle[paper_id]
            title = x['display_name']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")
        sentences.append(query)
        embeddings = self.sentence_transformer.embed(sentences)
        cosine_scores = cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
        for paper_id, score in zip(self.oracle, cosine_scores):
            self.oracle[paper_id]["features"][0] = score
    
    def _predict_paper_rank(self):
        paper_ids, features = [], []
        for k, v in self.oracle.items():
            paper_ids.append(k)
            features.append(v['features'])
        ranks = self.letor.predict(np.array(features)).tolist()
        for k, r in zip(paper_ids, ranks):
            self.oracle[k]['rank'] = r
    
    async def __call__(self, query: str) -> dict:
        logging.info(f"DynamicOracleGenerator::Request for oracle paper with query {query}")
        # 1. Request for papers
        await self._collect_oracles(query)
        # calculate features
        # feature 0: cosine similarity (-1~1)
        # feature 1: citation count == global prestige (regularized to 0~1)
        # feature 2: local citation count == local_prestige (regularized to 0~1)
        # feature 3: local pagerank (regularized to 0~1)
        # feature 4: citation velocity == emengence
        for x in self.oracle.values():
            x['features'] = [0, 0, 0, 0, 0]
            x['features'][1] = self._citation_count_by_eval_date(x)
        # 2. Get post-calculate features
        logging.info(f"DynamicOracleGenerator::Get features 1 3 4")
        # 2.1 sentence transformer cosine similarity
        self._calculate_similarity(query)
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        self._local_citation_and_pagerank()
        # regularization
        max_citation_count = math.log1p(max(x['features'][1] for x in self.oracle.values()))
        max_local_citation_count = math.log1p(max(x['features'][2] for x in self.oracle.values()))
        for x in self.oracle.values():
            if max_citation_count > 0:
                x['features'][1] = math.log1p(x['features'][1]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][2] = math.log1p(x['features'][2]) / max_local_citation_count
            x['features'][4] = x['features'][1] / math.log1p(self._paper_age(x))
        # 3. Get rank
        logging.info(f"DynamicOracleGenerator::Get rank")
        self._predict_paper_rank()
        return {"oracle_papers": self.oracle}
