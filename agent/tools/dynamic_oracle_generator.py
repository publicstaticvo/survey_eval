import re
import math
import json
import logging
import networkx as nx
from bertopic import BERTopic
from datetime import datetime
from pydantic import BaseModel, Field
from sentence_transformers import util
from langchain_core.tools import BaseTool
from sbert_client import SentenceTransformerClient
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor as TPE

from utils import openalex_search_paper, index_to_abstract, URL_DOMAIN
from prompts import SUBTOPIC_GENERATION_PROMPT
from llm_server import ConcurrentLLMClient


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


class OracleInput(BaseModel):
    query: str = Field(..., description="Research query.")


class DynamicOracleGenerator(BaseTool):
    name = "dynamic_oracle_generator"
    description = "Generates seed set and features."
    args_schema: type[BaseModel] = OracleInput
    
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
        self.oracle_ids = []
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
    
    def _fetch_metadata_from_response(self, response):
        if not response: return
        response['id'] = response['id'].replace(URL_DOMAIN, "")
        response['abstract'] = index_to_abstract(response['abstract_inverted_index'])
        del response['abstract_inverted_index']
        # referenced_works
        response['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in response['referenced_works']]
        # reorganize counts_by_year
        response['counts_by_year'] = {y['year']: y['cited_by_count'] for y in response['counts_by_year']}
        return response

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
        min_openalex_relevance = self.oracle[self.oracle_ids[-1]]['relevance_score']
        corefs = {}
        for x in self.oracle.values():
            for y in x['referenced_works']:
                if y not in self.oracle:
                    corefs[y] = corefs.get(y, 0) + 1
        # get information of high co-citations
        threshold = int(threshold * len(self.oracle))
        high_corefs = [x for x, y in corefs.items() if y >= threshold]  # work_ids
        
        def get_paper_by_workid(work_id):
            x = openalex_search_paper(f"works/{work_id}")
            if not x: return
            x['id'] = x['id'].replace(URL_DOMAIN, "")
            # abstract
            x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
            del x['abstract_inverted_index']
            # referenced_works
            x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
            # reorganize counts_by_year
            x['counts_by_year'] = {y['year']: y['cited_by_count'] for y in x['counts_by_year']}
            # set min relevance score
            x['relevance_score'] = min_openalex_relevance
            return x

        with TPE(max_workers=min(10, len(high_corefs))) as pool:
            for x in pool.map(get_paper_by_workid, high_corefs):
                if not x: continue
                assert x['id'] not in self.oracle
                self.oracle[x['id']] = x
                self.oracle_ids.append(x['id'])
    
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

        # 3. Calculate PageRank
        # alpha=0.85 is the standard damping factor used by Google
        pagerank_scores = nx.pagerank(citation_graph, alpha=0.85)

        # 4. Update the original list with PageRank scores
        for paper_id, score in pagerank_scores.items():
            paper_map[paper_id]['local_pagerank'] = score
            self.oracle[paper_id]['features'][3] = paper_map[paper_id]['local_citation_count']
            self.oracle[paper_id]['features'][4] = score

        return nx.to_dict_of_lists(citation_graph)

    def _calculate_similarity(self, query: str):
        # vector similarity query and title & abstract
        sentences = []
        for paper_id in self.oracle_ids:
            x = self.oracle[paper_id]
            title = x['display_name']
            abstract = x.get("abstract", None)
            sentences.append(f"{title}. {abstract}" if abstract else f"{title}.")
        sentences.append(query)
        embeddings = self.sentence_transformer.embed(sentences)
        cosine_scores = util.cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
        for paper_id, score in zip(self.oracle_ids, cosine_scores):
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
    
    def _run(self, query: str) -> dict:
        logging.info(f"DynamicOracleGenerator::Request for oracle paper with query {query}")
        # 1. Request for papers
        self._request_for_papers(query)
        # 2. Get post-calculate features
        logging.info(f"DynamicOracleGenerator::Get features 1 3 4")
        # 2.1 sentence transformer cosine similarity
        self._calculate_similarity(query)
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        citation_graph = self._local_citation_and_pagerank()
        # regularization
        max_citation_count = math.log(1 + max(x['features'][2] for x in self.oracle.values()))
        max_local_citation_count = math.log(1 + max(x['features'][3] for x in self.oracle.values()))
        for x in self.oracle.values():
            if max_citation_count > 0:
                x['features'][2] = math.log(1 + x['features'][2]) / max_citation_count
            if max_local_citation_count > 0:
                x['features'][3] = math.log(1 + x['features'][3]) / max_local_citation_count
            x['features'][5] = x['features'][2] / math.log(1 + self._paper_age(x))
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
        self.subtopics = self.llm_model.run_llm({"query": query, "clusters": json.dumps(inputs, indent=2, ensure_ascii=False)})
        logging.info(f"DynamicOracleGenerator::Return")
        return {"oracle_papers": self.oracle, "adjacent_graph": citation_graph, "subtopics": self.subtopics}
    
    async def _arun(self, query: str) -> str:
        return await self._run(query)
