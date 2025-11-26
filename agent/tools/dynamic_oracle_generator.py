import re
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

from utils import openalex_search_paper, index_to_abstract, URL_DOMAIN
from llm_server import ConcurrentLLMClient


class SubtopicLLMClient(ConcurrentLLMClient):

    format_pattern: re.Pattern = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
    PROMPT: str = """..."""

    def __init__(self, llm, sampling_params, n_workers, retry = 5):
        super().__init__(llm, sampling_params, n_workers, retry)

    def _pattern_check(self, output):
        try:
            return self.format_pattern.findall(output)[-1]
        except:
            return

    def run_llm(self, inputs):
        # Should only return one subtopic name
        message = self.PROMPT.format(**inputs)
        while (pattern := self._pattern_check(super().run_llm(message))) is None: pass
        return pattern


class DynamicOracleInput(BaseModel):
    query: str = Field(description="The research query topic (e.g., 'Federated Learning in Edge Computing').")


class DynamicOracleGenerator(BaseTool):
    name = "dynamic_oracle_generator"
    description = (
        "Generates a 'Dynamic Oracle' for evaluation. "
        "1. Retrieves top 1000 papers from OpenAlex to build a seed set. "
        "2. Calculates prestige features (PageRank, Co-citation) and extracts essential subtopics. "
        "Use this ONCE at the start."
    )
    args_schema: type[BaseModel] = DynamicOracleInput
    
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

    def _citation_velocity(self, paper):
        paper_age = self._paper_age(paper)
        cited_by = self._citation_count_by_eval_date(paper)
        return cited_by / paper_age
    
    def _local_citation_and_pagerank(self):
        """
        Calculates local citation counts and PageRank for a list of papers.
        Modifies the dictionary in-place.
        """
        # 1. Create a lookup map and a set of valid IDs
        # This allows O(1) access to paper objects and quick existence checks
        # Initialize local_citation_count to 0 for all papers
        paper_map = {}
        for paper_id in self.oracle_ids:
            x = self.oracle[paper_id]
            references = x['referenced_works'] if x['referenced_works'] else []
            paper_map[paper_id] = {"references": references, "local_citation_count": 0}
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
            
            for target_id in citations:
                # strictly filter for 'local' papers only
                if target_id in valid_ids:
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

    def _request_for_papers(self, query):
        # TODO: 用标题+一作姓氏作为paper fingerprint，并统计相同的条目：引用数量求和。
        filter_params = {
            "default.search": query,
            "to_publication_date": self.eval_date.strftime("%Y-%m-%d"), 
        }
        page = 1
        while len(self.oracle) < self.num_oracle_papers:
            results = openalex_search_paper("works", filter_params, per_page=200, page=page, retry=100)
            for x in results.get('results'):
                if (paper_id := x['id'].replace(URL_DOMAIN, "")) not in self.oracle:
                    if x['abstract_inverted_index']: 
                        x['abstract'] = index_to_abstract(x['abstract_inverted_index'])
                    del x['abstract_inverted_index']
                    x['referenced_works'] = [y.replace(URL_DOMAIN, "") for y in x['referenced_works']]
                    cited_by_count = self._citation_count_by_eval_date(x)
                    # feature 0: openalex relevance score
                    # feature 1: cosine similarity
                    # feature 2: citation count == global prestige
                    # feature 3: local citation count == local_prestige
                    # feature 4: local pagerank
                    # feature 5: citation velocity == emengence
                    x['features'] = [x['relevance_score'], 0, cited_by_count, 0, 0, self._citation_velocity(x)]
                    self.oracle[paper_id] = x
                    self.oracle_ids.append(paper_id)
                    if len(self.oracle) == self.num_oracle_papers:
                        break
            page += 1
        self.min_openalex_relevance = self.oracle[self.oracle_ids[-1]]['relevance_score']

    def _calculate_similarity(self, query: str):
        sentences = []
        for paper_id in self.oracle_ids:
            x = self.oracle[paper_id]
            sentences.append(x.get("abstract", x['display_name']))
        sentences.append(query)
        embeddings = self.sentence_transformer.embed(sentences)
        cosine_scores = util.cos_sim(embeddings[-1:], embeddings[:-1])[0].tolist()
        for paper_id, score in zip(self.oracle_ids, cosine_scores):
            self.oracle[paper_id]["feature"][1] = score

    def _cluster_with_bertopic(self):
        paper_titles = [x['display_name'] for x in self.oracle.values()]
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=2,  # Minimum document frequency
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
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
    
    def _run(self, query: str, run_manager=None) -> dict:
        logging.info(f"DynamicOracleGenerator::Request for oracle paper with query {query}")
        # 1. Request for papers
        self._request_for_papers(query)
        # 2. Get post-calculate features
        logging.info(f"DynamicOracleGenerator::Get features 1 3 4")
        # 2.1 sentence transformer cosine similarity
        self._calculate_similarity(query)
        # 2.3 local citation count == local_prestige && 2.4 local pagerank
        self._local_citation_and_pagerank()
        # 3. Get subtopics
        logging.info(f"DynamicOracleGenerator::Get subtopics")
        # 3.1 Cluster with BERTopic
        clusters = self._cluster_with_bertopic()
        # 3.2 Get subtopic names by LLM
        inputs = []
        for k, v in clusters.items():
            if k == -1: continue
            title_str = "\n".join(v['paper_titles'])
            keywords = ", ".join(v['keywords'])
            inputs.append({"titles": title_str, "keywords": keywords})
        self.subtopics = self.llm_model.run_parallel(inputs)
        logging.info(f"DynamicOracleGenerator::Return")
        return {"oracle_papers": self.oracle, "subtopics": self.subtopics}
    
    async def _arun(self, query: str, run_manager=None) -> str:
        return await self._run(query, run_manager)
