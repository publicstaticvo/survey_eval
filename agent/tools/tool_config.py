from dataclasses import dataclass, field
from collections.abc import Mapping
from datetime import datetime
from typing import Any
import yaml


GREEDY_PARAMS = {
    'temperature': 0.0, "max_tokens": 8192, "seed": 42,
    "top_p": 1.0,      # 设置为1，不进行核采样
    "top_k": 1,        # 或设置为1，确保总是选择最可能的token
    "repetition_penalty": 1.0,  # 设置为1，禁用重复惩罚
    "length_penalty": 1.0,      # 设置为1，禁用长度惩罚
    "no_repeat_ngram_size": 0,  # 设置为0，禁用n-gram重复惩罚
}


@dataclass(frozen=True)
class LLMServerInfo:
    base_url: str = "https://uni-api.cstcloud.cn"
    api_key: str = "7868485c0ca1d66880fdb72e06b77ebfc6daf07faf61638e81e7a79adf7e309d"
    model: str = "gpt-oss-120b"
    

@dataclass(frozen=True)
class ToolConfig:
    # Agent
    agent_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    agent_max_tokens: int = 16384
    # General
    evaluation_date: datetime = field(default_factory=lambda: datetime.strptime("2024-12-31", "%Y-%m-%d"))
    # Sentence Transformer
    sbert_server_url: str = "http://172.18.36.90:8030"
    # external LLM
    llm_server_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    sampling_params: Mapping[str, Any] = field(default_factory=lambda: GREEDY_PARAMS)
    # dynamic oracle
    num_oracle_papers: int = 1000
    letor_path: str = "backup/ranker.txt"
    # citation parser
    grobid_url: str = "http://172.18.36.90:8070"
    grobid_num_workers: int = 10
    # factual correctness - reranking
    rerank_server_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    rerank_n_documents: int = 5
    # source selection
    topn: int = 0
    # topic coverage
    topic_weak_sim_threshold: float = 0.45
    topic_sim_threshold: float = 0.55
    # fact check
    mean_cov_weight: float = 0.7
    non_compat_punishment: float = 0.6
    confidence_threshold: float = 0.6
    # quality
    sentence_similarity_threshold: float = 0.97
    paragraph_similarity_threshold: float = 0.92
    redundancy_ngram: int = 5

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f: config = yaml.safe_load(f)
        return cls(
            agent_info=LLMServerInfo(
                base_url=config['agent']['base_url'],
                api_key=config['agent']['api_key'],
                model=config['agent']['model'],
            ),
            agent_max_tokens=config['agent']['max_tokens'],
            evaluation_date=datetime.strftime(config['general']['evaluation_date']),
            num_oracle_papers=config['dynamic_oracle']['num_oracle_papers'],
            letor_path=config['dynamic_oracle']['letor_path'],
            llm_server_info=LLMServerInfo(
                base_url=config['external_llm']['base_url'],
                api_key=config['external_llm']['api_key'],
                model=config['external_llm']['model'],
            ),
            sbert_server_url=config['sbert']['base_url'],
            grobid_url=config['citation_parser']['grobid_url'],
            rerank_server_info=LLMServerInfo(
                base_url=config['rerank']['base_url'],
                api_key=config['rerank']['api_key'],
                model=config['rerank']['model'],
            ),
            rerank_n_documents=config['rerank']['num_documents'],
            topn=config['source_selection']['topn'],
            topic_weak_sim_threshold=config['topic_coverage']['topic_weak_sim_threshold'],
            topic_sim_threshold=config['topic_coverage']['topic_sim_threshold'],
            sentence_similarity_threshold=config['quality']['sentence_similarity_threshold'],
            paragraph_similarity_threshold=config['quality']['paragraph_similarity_threshold'],
            redundancy_ngram=config['quality']['redundancy_ngram'],
        )