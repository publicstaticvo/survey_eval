import re
import math
import json
import logging
import asyncio
import networkx as nx
from bertopic import BERTopic
from datetime import datetime
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor as TPE

from .request_utils import AsyncLLMClient, openalex_search_paper, URL_DOMAIN, RateLimit
from .prompts import SUBTOPIC_GENERATION_PROMPT, QUERY_EXPANSION_PROMPT
from .sbert_client import SentenceTransformerClient
from .utils import index_to_abstract, extract_json
from .tool_config import ToolConfig


class SubtopicLLMClient(AsyncLLMClient):
    def _availability(self, response):
        subtopic_map = extract_json(response)
        return list(subtopic_map.values())
    

class QueryExpansionLLMClient(AsyncLLMClient):
    PROMPT: str = QUERY_EXPANSION_PROMPT
    def _availability(self, response):
        queries = extract_json(response)
        return [re.sub(r"[:.,!?&]", "", query) for query in queries['queries']]


class DynamicOracleGenerator:
    
    def __init__(self, config: ToolConfig):
        self.oracle = {}
        self.library = {}  # All papers searched by oracle
        self.register_key = {"high_ref": []}  # register which paper belongs to which query
        self.eval_date = config.evaluation_date
        self.query_llm = QueryExpansionLLMClient(config.llm_server_info, config.sampling_params)
        self.subtopic_llm = SubtopicLLMClient(config.llm_server_info, config.sampling_params)
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
        return (self.eval_date - datetime.strptime(paper['publication_date'], "%Y-%m-%d")).days
    
    def _fetch_metadata_from_response(self, response):
        if not response: return
        response['id'] = response['id'].replace(URL_DOMAIN, "")
        response['abstract'] = index_to_abstract(response['abstract_inverted_index'])
        del response['abstract_inverted_index']
        # referenced_works
        response['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in response['referenced_works']]
        return response

    async def _request_for_papers(self, query: str, uplimit: int):
        filter_params = {
            "default.search": query,
            "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
        }
        oracle, oracle_ids = {}, []
        page, previous_length = 1, 0
        async with RateLimit.OPENALEX_SEMAPHORE:
            while len(self.oracle) < uplimit:
                results = await openalex_search_paper("works", filter_params, per_page=200, page=page)
                if not results:
                    print(f"An network issue cause oracle data miss in query {query}.")
                    break
                for x in results['results']:
                    x['id'] = x['id'].replace(URL_DOMAIN, "")
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
        
        async def get_paper_by_workid(work_id):
            async with RateLimit.OPENALEX_SEMAPHORE:
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
            if not x: continue
            new_oracles[x['id']] = x
            self.register_key['high_ref'].append(x['id'])
        return new_oracles
       
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
            # TODO: Z-scores

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
            self.oracle[paper_id]["feature"][1] = score

    def _cluster_with_bertopic(self):
        paper_titles = [x['display_name'] for x in self.oracle.values()]
        vectorizer_model = CountVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))
        topic_model = BERTopic(
            embedding_model=self.sentence_transformer,
            vectorizer_model=vectorizer_model,
            min_topic_size=10,  # Minimum papers per topic
            nr_topics='auto',  # Automatically determine number of topics
            calculate_probabilities=True,
            verbose=True
        )
        topics_of_papers, _ = topic_model.fit_transform(paper_titles)
        topics = {-1: {"keywords": ["N/A"], "paper_titles": []}}
        for topic_id, title in zip(topics_of_papers, paper_titles):
            if topic_id not in topics:
                topics[topic_id] = {"keywords": topic_model.get_topic(topic_id), "paper_titles": []}
            topics[topic_id]['paper_titles'].append(title)
        return topics
    
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
                except:
                    pass
            # 3 get high_refs
            high_refs = await self._get_high_corefs()
            self.library.update(high_refs)
            # 4 select oracle papers
            if len(self.library) >= self.num_oracle_papers:
                # fill in self.library
                selected = self.register_key['high_ref']
                lengths = [len(self.register_key[q]) for q in prev_queries]
                amount = _average_cut(lengths, self.num_oracle_papers - len(selected))
                for q, a in zip(prev_queries, amount):
                    selected.extend(self.register_key[q][:a])
                assert len(selected) == self.num_oracle_papers, (len(selected), self.num_oracle_papers)
    
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
        citation_graph = self._local_citation_and_pagerank()
        # regularization
        max_citation_count = math.log(1 + max(x['features'][1] for x in self.oracle.values()))
        max_local_citation_count = math.log(1 + max(x['features'][2] for x in self.oracle.values()))
        for x in self.oracle.values():
            if max_citation_count > 0:
                x['features'][1] = math.log(1 + x['features'][1]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][2] = math.log(1 + x['features'][2]) / max_local_citation_count
            x['features'][4] = x['features'][1] / math.log(1 + self._paper_age(x))
        # 3. Get subtopics
        logging.info(f"DynamicOracleGenerator::Get subtopics")
        # 3.1 Cluster with BERTopic
        clusters = self._cluster_with_bertopic()
        # 3.2 Get subtopic names by LLM
        inputs = {}
        cluster_count = 0
        for k, v in clusters.items():
            if k == -1: continue
            cluster_count += 1
            inputs[f'Cluster {cluster_count}'] = v['keywords']
            
        message = SUBTOPIC_GENERATION_PROMPT.format(query=query, clusters=json.dumps(inputs, indent=2, ensure_ascii=False))
        self.subtopics = await self.subtopic_llm.call(messages=message)
        logging.info(f"DynamicOracleGenerator::Return")
        return {"oracle_papers": self.oracle, "adjacent_graph": citation_graph, "subtopics": self.subtopics}
