import json
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from sbert_client import SentenceTransformerClient


def normalize(a: np.ndarray) -> np.ndarray:
    norm = np.sum(a * a)
    if norm == 0: return a
    return a / np.sqrt(norm)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    b = normalize(b)
    return float(np.sum(a * b))


class SourceCriticInput(BaseModel):
    cited_paper_ids: List[str] = Field(..., description="List of paper IDs cited in the survey.")
    oracle_data: Dict[str, Any] = Field(..., description="Output from DynamicOracleGenerator.")


class SourceSelectionCritic(BaseTool):
    name = "source_selection_critic"
    description = (
        "Evaluates the quality of the bibliography. "
        "Uses a LETOR model to compare the agent's citations against the "
        "ranked Dynamic Oracle list. Returns nDCG score and missing important papers."
    )
    args_schema: type[BaseModel] = SourceCriticInput

    def __init__(
            self, 
            sentence_transformer: SentenceTransformerClient, 
            letor_model,
            eval_date: datetime = datetime.now(),
            topn: int = 0,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.sentence_transformer = sentence_transformer
        self.letor_model = letor_model
        self.eval_date = eval_date
        self.topn = topn

    def _paper_age(self, paper):
        return (self.eval_date - datetime.strptime(paper['publication_date'], "%Y-%m-%d")).days

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
    
    def calculate_rank_biased_recall(self, agent_paper_ids, oracle_ranked_list):
        """
        Calculates Rank-Biased Recall @ Top-N.
        
        :param agent_paper_ids: Set of paper IDs cited by the agent.
        :param oracle_ranked_list: List of paper IDs from LETOR, sorted by score (desc).
        :param self.topn: The cutoff for the 'Essential' list (e.g., 100).
        :return: Float (0.0 to 1.0)
        """
        topn = self.topn if self.topn else len(oracle_ranked_list)

        # 1. Slice the Oracle to just the Top-N essential papers
        oracle_topn = oracle_ranked_list[:topn]
        
        # 2. Calculate the Ideal Discounted Cumulative Gain (IDCG)
        # This is the max score if the agent found ALL Top-N papers.
        idcg = sum([1.0 / math.log2(rank + 2) for rank in range(len(oracle_topn))])
        
        # 3. Calculate Actual Discounted Cumulative Gain (DCG)
        dcg = 0.0
        for rank, paper_id in enumerate(oracle_topn):
            if paper_id in agent_paper_ids:
                # Agent found this important paper! Add its weight.
                dcg += 1.0 / math.log2(rank + 2)
                
        # 4. Normalize
        if idcg == 0: return 0.0
        return dcg / idcg

    def _run(self, citations: dict, query: str, oracle_data: dict):
        """
        The format of param citations is:
        {
            "metadatas": list of metadatas [{"id": ., "year": ., "citation_count": .,}]
            "abstract": "abstract", 
            "full_content": "full_content" or "abstract" or "title",
            "status": 0-3
        }
        status == 0 -> OK
        status == 1 -> fail to download paper
        status == 2 -> fail to download paper and fetch abstract
        status == 3 -> fail to get information of the citation. Please check its existance.
        The key of oracle_data['dynamic_oracle_data'] is (title, first_author).
        """
        # preparation
        oracle_data = oracle_data['dynamic_oracle_data']
        num_oracles, num_citations = len(oracle_data), len(citations)
        self.topn = min(num_citations, num_oracles)
        min_relevance, min_pagerank, max_citations = 10, 10, 0
        for x in oracle_data.values:
            min_relevance = min(min_relevance, x['feature'][0])
            min_pagerank = min(min_pagerank, x['feature'][4])
            max_citations = max(max_citations, x['feature'][2])
        max_citations = math.log(1 + max_citations)
        query_embedding = normalize(self.sentence_transformer.embed([query])[0])
        # try to search in oracle papers
        map_workid_to_oracle = {}  # transfer the key from (title, author) to work_id
        oracle_citation_feature_map, not_oracle_citation_feature_map = {}, {}  # citation key -> feature
        # TODO: Should we change the citation key from "b123" to "Yu et al."?
        for v in oracle_data.values():
            for i in v['id']: map_workid_to_oracle[v['id'][i]] = v
        unfound_papers = []
        # normalization
        for citation_key, citation in citations.items():
            if citation['status'] == 3:
                # process unfound papers
                unfound_papers.append(citation['full_content'])
                continue
            metadata = citation['metadatas']
            for i in metadata:
                if i['id'] in map_workid_to_oracle: 
                    # The cited paper is in oracle papers
                    oracle_citation_feature_map[citation_key] = map_workid_to_oracle[i]['feature']
                    map_workid_to_oracle[i]['citation_key'] = citation_key
                    break
            else:
                # Not in
                feature = [min_relevance / 2, 0, 0, 0, min_pagerank, 0]
                feature[1] = cos_sim(query_embedding, self.sentence_transformer.embed([citation['abstract']])[0])
                feature[2] = math.log(1 + self._citation_count_by_eval_date(metadata)) / max_citations
                feature[5] = feature[2] / math.log(1 + self._paper_age(metadata))
                not_oracle_citation_feature_map[citation_key] = feature
        # rank all oracle papers and non-oracle cited papers
        paper_features, paper_ids = [], []
        for paper_id, paper in oracle_data.items():
            paper_features.append(paper['feature'])
            paper_ids.append(paper_id)
        for citation_key, citation in not_oracle_citation_feature_map.items():
            paper_features.append(citation['feature'])
            paper_ids.append(citation_key)
        ranks = self.letor_model.rank(paper_features)
        # calculate metrics, incorrect papers and missing papers
        # ranked based recall
        oracle_ranked_list = sorted(range(num_oracles), key=lambda i: ranks[i], reverse=True)
        cited_paper_ranked_list = list(range(num_oracles, len(ranks)))
        cited_paper_ranks = ranks[num_oracles:]
        for i in range(num_oracles):
            if "citation_key" in oracle_data[paper_ids[i]]:  # cited oracle paper
                cited_paper_ranked_list.append(i)
                cited_paper_ranks.append(ranks[i])
        cited_paper_ranked_list = sorted(cited_paper_ranked_list, key=lambda i: ranks[i], reverse=True)
        ranked_based_recall = self.calculate_rank_biased_recall(cited_paper_ranked_list, oracle_ranked_list)
        # weighted precision
        weighted_precision = sum(cited_paper_ranks) / num_citations
        # missing papers: Papers that are not cited in top N of the oracle list.
        missing = []
        for i in range(num_oracles):
            if "citation_key" in oracle_data[paper_ids[i]]:
                missing.append([oracle_data[paper_ids[i]], ranks[i]])
        missing = sorted(missing, key=lambda x: x[1], reverse=True)
        missing = {x[0]: x[1] for x in missing if x[1] > oracle_ranked_list[self.topn]}
        # false positives: All papers that has low ranks and are not in the oracle list.
        incorrect = [[not_oracle_citation_feature_map[paper_ids[i]], ranks[i]] for i in range(num_oracles, len(ranks))]
        incorrect = sorted(incorrect, key=lambda x: x[1], reverse=True)
        incorrect = {x[0]: x[1] for x in incorrect if x[1] < oracle_ranked_list[self.topn]}
        for m in unfound_papers:
            incorrect[m] = "paper not found"
        return {
            "ranked_based_recall": ranked_based_recall,
            "weighted_precision": weighted_precision,
            "missing_important_papers": missing,
            "incorrect_papers": incorrect,
        }

    async def _arun(self, citations: dict, query: str, oracle_data: dict):
        return await self._run(citations, query, oracle_data)
