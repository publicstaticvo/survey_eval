import re
import math
import json
import tqdm
import itertools
import numpy as np
import networkx as nx
from bertopic import BERTopic
from datetime import datetime
from sentence_transformers import util
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor as TPE
from sklearn.feature_extraction.text import CountVectorizer

from survey_eval.agent.tools.utils import openalex_search_paper, index_to_abstract, URL_DOMAIN, valid_check
from survey_eval.agent.tools.prompts import SUBTOPIC_GENERATION_PROMPT
from survey_eval.agent.tools.llm_server import ConcurrentLLMClient
from survey_eval.agent.tools.tool_config import LLMServerInfo
from survey_eval.agent.tools.sbert_client import SentenceTransformerClient


def normalize(a: np.ndarray) -> np.ndarray:
    norm = np.sum(a * a)
    if norm == 0: return a
    return a / np.sqrt(norm)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    b = normalize(b)
    return float(np.sum(a * b))


class SubtopicLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\{.+?\}", re.DOTALL)
    PROMPT: str = SUBTOPIC_GENERATION_PROMPT

    def __init__(self, llm, sampling_params, retry = 5):
        super().__init__(llm, sampling_params, 1, retry)

    def _pattern_check(self, output):
        try:
            subtopic_map = json.loads(self.format_pattern.findall(output)[-1])
            return list(subtopic_map.values())
        except:
            return

    def run_llm(self, inputs):
        # Should only return one subtopic name
        message = self.PROMPT.format(**inputs)
        while (pattern := self._pattern_check(super().run_llm(message))) is None: pass
        return pattern


def get_metadata(paper: Dict[str, Any]):
    # The following information to get:
    return {
        "id": paper['id'].replace(URL_DOMAIN, ""),
        "ids": paper['ids'],
        "title": paper['display_name'],
        "authors": paper['authorship'],
        "locations": [x['source'] for x in paper['locations']],
        "cited_by_count": paper['cited_by_count'],
        "counts_by_year": paper['counts_by_year'],
        "publication_date": paper['publication_date'],
        "referenced_works": [x.replace(URL_DOMAIN, "") for x in paper['referenced_works']],
    }


def search_paper_from_api(paper_title: str) -> Dict[str, Any]:
    on_target = None
    results = openalex_search_paper("works", {"title.search": paper_title}).get("results", [])
    for paper_info in results:
        if valid_check(paper_title, paper_info['display_name']): 
            paper_info['id'] = paper_info['id'].replace(URL_DOMAIN, "")
            on_target = paper_info
            break
    if not on_target: return
    return {
        "metadata": get_metadata(on_target), 
        "title": paper_title,
        "abstract": index_to_abstract(on_target['abstract_inverted_index']),
    }


metadata_map = {}


class DynamicOracleGenerator:
    
    def __init__(
            self, 
            num_oracle_papers: int, 
            llm_model: SubtopicLLMClient, 
            sentence_transformer: SentenceTransformerClient,
            eval_date: datetime = datetime.now(), 
            **kwargs
        ):
        super().__init__(**kwargs)
        self.oracle = {}
        self.negatives = {}
        self.oracle_ids = []
        self.hard_negatives = {}
        self.eval_date = eval_date
        self.llm_model = llm_model
        self.num_oracle_papers = num_oracle_papers
        self.sentence_transformer = sentence_transformer

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

    def _request_for_papers(self, query):
        # key现在是标题+一作。
        filter_params = {
            "default.search": query,
            "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
        }
        page = 1
        while len(self.oracle) < self.num_oracle_papers:
            results = openalex_search_paper("works", filter_params, per_page=200, page=page, retry=5)
            for x in results.get('results'):
                x['id'] = x['id'].replace(URL_DOMAIN, "")
                if (paper_id := x['id']) not in self.oracle:
                    # abstract
                    x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
                    del x['abstract_inverted_index']
                    # referenced_works
                    x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
                    # reorganize counts_by_year
                    x['counts_by_year'] = {y['year']: y['cited_by_count'] for y in x['counts_by_year']}
                    # store
                    self.oracle[paper_id] = x
                    self.oracle_ids.append(paper_id)
                    if len(self.oracle) == self.num_oracle_papers: break
            page += 1
        # Get high referenced paper of oracle papers
        self._get_high_corefs()
        # calculate features
        # feature 0: openalex relevance score (0~1)
        # feature 1: cosine similarity (-1~1)
        # feature 2: citation count == global prestige (regularized to 0~1)
        # feature 3: local citation count == local_prestige (regularized to 0~1)
        # feature 4: local pagerank (regularized to 0~1)
        # feature 5: citation velocity == emengence
        for x in self.oracle.values():
            x['features'] = [x['relevance_score'], 0, 0, 0, 0, 0]
            x['features'][2] = self._citation_count_by_eval_date(x)
    
    def _get_high_corefs(self, threshold: float = 0.05):
        self.min_openalex_relevance = self.oracle[self.oracle_ids[-1]]['relevance_score']
        self.corefs = {}
        for x in self.oracle.values():
            for y in x['referenced_works']:
                if y not in self.oracle:
                    self.corefs[y] = self.corefs.get(y, 0) + 1
        # get information of high co-citations
        threshold = int(threshold * len(self.oracle))
        high_corefs = [x for x, y in self.corefs.items() if y >= threshold]  # work_ids
        
        def get_paper_by_workid(work_id):
            x = openalex_search_paper(f"works/{work_id}")
            if not x: return
            x = get_metadata(x)
            x['relevance_score'] = self.min_openalex_relevance / 2
            # set min relevance score
            x['relevance_score'] = self.min_openalex_relevance
            return x

        for work in high_corefs:
            x = get_paper_by_workid(work)
            if not x: continue
            assert x['id'] not in self.oracle
            self.oracle[x['id']] = x
            self.oracle_ids.append(x['id'])
    
    def _negative_sampling(self, number: int, margin_citation_count: int):
        # 2 kind of negative samples: 1. Easy negatives that has 0 local prestige
        while len(self.negatives) < number:
            results = openalex_search_paper("works", do_sample=True, per_page=200)
            for x in results.get('results', []):
                x['id'] = x['id'].replace(URL_DOMAIN, "")
                if all((paper_id := x['id']) not in y for y in [self.oracle, self.corefs, self.negatives]):
                    x = get_metadata(x)
                    x['relevance_score'] = self.min_openalex_relevance / 2
                    # feature - no local prestige and pagerank
                    x['feature'] = [x['relevance_score'], 0, 0, 0, 0, 0]
                    x['feature'][2] = math.log(1 + self._citation_count_by_eval_date(x))
                    # store
                    self.negatives[paper_id] = x
                    if len(self.negatives) == number: break
        # 2. Hard negatives with high global prestige but 0 relevancy
        M = int(margin_citation_count * 1.5)
        while len(self.hard_negatives) < number:
            results = openalex_search_paper("works", filter={"cited_by_count": f">{M}"}, do_sample=True, per_page=200)
            for x in results.get('results', []):
                x['id'] = x['id'].replace(URL_DOMAIN, "")
                if all((paper_id := x['id']) not in y for y in [self.oracle, self.corefs, self.negatives, self.hard_negatives]):
                    x = get_metadata(x)
                    x['relevance_score'] = self.min_openalex_relevance / 2
                    # store
                    if (real_citation_count := self._citation_count_by_eval_date(x)) >= margin_citation_count:
                        # feature - no local prestige and pagerank
                        x['feature'] = [x['relevance_score'], 0, 0, 0, 0, 0]
                        x['feature'][2] = math.log(1 + real_citation_count)
                        self.hard_negatives[paper_id] = x
                        if len(self.hard_negatives) == number: break

    def _local_citation_and_pagerank(self):
        """
        Calculates local citation counts and PageRank for a list of papers.
        Modifies the dictionary in-place.
        """
        # 1. Create a lookup map and a set of valid IDs
        # This allows O(1) access to paper objects and quick existence checks
        # Initialize local_citation_count to 0 for all papers
        paper_map = {}
        for paper_id in self.oracle_ids:  # paper_id = (title, first author)
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
            self.oracle[paper_id]['features'][3] = paper_map[paper_id]['local_citation_count']
            self.oracle[paper_id]['features'][4] = score

    def _calculate_similarity(self, query: str):
        # vector similarity query and title & abstract
        sentences, paper_ids = [], []

        for paper_id in self.oracle_ids:
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
    
    def _calculate_features_for_citations(self, citations: List[Dict[str, Any]], query: str):
        """
        The format of param citations is:
        {
            "metadatas": list of metadatas [{"id": ., "year": ., "citation_count": .,}]
            "title": "title",
            "abstract": "abstract", 
        }
        The key of oracle_data['dynamic_oracle_data'] is (title, first_author).
        """
        # preparation
        num_oracles, num_citations = len(self.oracle), len(citations)
        self.topn = min(num_citations, num_oracles)
        min_relevance, max_citations, max_local_citations = 10, 0, 0
        for x in self.oracle.values:
            min_relevance = min(min_relevance, x['feature'][0])
            max_citations = max(max_citations, x['feature'][2])
            max_local_citations = max(max_local_citations, x['feature'][3])
        max_citations = math.log(1 + max_citations)
        max_local_citations = math.log(1 + max_local_citations)
        query_embedding = normalize(self.sentence_transformer.embed([query])[0])
        # normalization
        for citation in citations:
            metadata = citation['metadata']
            i = metadata['id']
            if i in self.oracle: 
                # The cited paper is in oracle papers
                feature = self.oracle[i]['feature']
            else:
                # Not in
                feature = [min_relevance / 2, 0, 0, 0, 0, 0]
                feature[1] = cos_sim(query_embedding, self.sentence_transformer.embed([citation['abstract']])[0])
                # feature 2: global citations
                feature[2] = math.log(1 + self._citation_count_by_eval_date(metadata))
                if max_citations > 0: feature[2] /= max_citations
                # local co-citation: calcultate number of oracle papers cites this paper
                # local_pagerank = average(pageranks of oracle papers cites this paper)
                ids_set = set(metadata['id'])
                for n in self.citation_graph:
                    if set(ids_set) - set(self.citation_graph[n]):
                        feature[3] += 1
                        feature[4] += self.oracle[n]['feature'][4]
                if feature[3] > 0: feature[4] /= feature[3]
                feature[3] = math.log(1 + feature[3])
                if max_local_citations > 0: feature[3] /= max_local_citations
                # feature 5: 
                feature[5] = feature[2] / math.log(1 + self._paper_age(metadata))
            citations['feature'] = feature
        return citations
    
    def run(self, query: Dict[str, Any]) -> dict:
        print(f"DynamicOracleGenerator::Request for oracle paper with query {query['query']}")
        # 1. Request for papers
        self._request_for_papers(query['query'])
        # 2. Get post-calculate features
        print(f"DynamicOracleGenerator::Get features 1 3 4")
        # 2.1 sentence transformer cosine similarity
        self._calculate_similarity(query['query'])
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        self._local_citation_and_pagerank()
        # negative sampling
        mid_citation_count = sorted([x['features'][2] for x in self.oracle.values()])[len(self.oracle) // 2]
        self._negative_sampling(len(query['references']), mid_citation_count)
        # regularization
        max_citation_count = math.log(1 + max(x['features'][2] for x in self.oracle.values()))
        max_local_citation_count = math.log(1 + max(x['features'][3] for x in self.oracle.values()))
        for x in itertools.chain(self.oracle.values(), self.negatives.values(), self.hard_negatives.values()):
            if max_citation_count > 0:
                x['features'][2] = math.log(1 + x['features'][2]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][3] = math.log(1 + x['features'][3]) / max_local_citation_count
            x['features'][5] = x['features'][2] / math.log(1 + self._paper_age(x))
        print("DynamicOracleGenerator::Calculate Citation Feature")
        reference_metadata = [metadata_map[x] for x in query['references']]
        self._calculate_features_for_citations(reference_metadata, query['query'])
        return reference_metadata


def load_local(fn):
    with open(fn, "r+", encoding="utf-8") as f:
        d = [json.loads(line.strip()) for line in f if line.strip()]
    return d

def print_json(d, fn):
    with open(fn, "w+", encoding="utf-8") as f:
        for x in d:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


with open("../api_key.json") as f: json_key = json.load(f)
key = {}
for k in ['cstcloud', 'deepseek']:
    for m in json_key[k]['models']:
        key[m] = {"base_url": json_key[k]['domain'], "api_key": json_key[k]['key']}

model = "gpt-oss-120b"
llm_info = LLMServerInfo(base_url=key[model]['base_url'], api_key=key[model]['api_key'], model=model)
llm = SubtopicLLMClient(llm=llm_info, sampling_params={'temperature': 0.0, "max_tokens": 16384})
st = SentenceTransformerClient("http://localhost:8030/encode", 64)


def get_oracle(query):
    worker = DynamicOracleGenerator(1000, llm, st, datetime.strptime(query['date'], "%Y-%m-%d"))
    results = worker.run(query)
    return results
    

def main():
    queries = load_local("queries.jsonl")
    # 0. Get Citation Information
    print("DynamicOracleGenerator::Get Citation Information")
    pending_jobs = []
    for x in queries:
        pending_jobs.extend(x['references'])
    pending_jobs = list(set(pending_jobs))
    with TPE(max_workers=10) as executor:
        for paper, info in zip(pending_jobs, executor.map(search_paper_from_api, pending_jobs)):
            metadata_map[paper] = info    
    # get_oracle
    results = get_oracle(queries[0])
    # multiprocessing
    # results = []
    # with TPE(max_workers=20) as pool:
    #     for result in tqdm.tqdm(pool.map(get_oracle, [x['greedy'] for x in queries]), total=len(queries)):
    #         results.append(result)
    with open("results.jsonl", 'w+', encoding='utf-8') as f: json.dump(results, f, indent=2, ensure_ascii=False)
