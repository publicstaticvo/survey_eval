from dataclasses import dataclass, field
from ruamel.yaml import YAML
from typing import Any
from collections.abc import Mapping
from datetime import datetime


@dataclass(frozen=True)
class LLMServerInfo:
    base_url: str
    api_key: str | None = None
    model: str = "whatever"
    

@dataclass(frozen=True)
class ToolConfig:
    # Agent
    agent_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    agent_max_tokens: int = 16384
    # General
    evaluation_date: datetime = field(default_factory=lambda: datetime.now())
    # Sentence Transformer
    sbert_server_url: str = "http://localhost:8000"
    # external LLM
    llm_num_workers: int = 20
    llm_server_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    sampling_params: Mapping[str, Any] = field(default_factory=lambda: {'temperature': 0.0, "max_tokens": 16384})
    # dynamic oracle
    num_oracle_papers: int = 1000
    # citation parser
    grobid_url: str = "http://localhost:8070"
    grobid_num_workers: int = 10
    # source selection
    topn: int = 0
    # topic coverage
    topic_similarity_threshold: float = 0.
    # quality
    redundancy_similarity_threshold: float = 0.95
    redundancy_ngram: int = 5

    @classmethod
    def from_yaml(cls, config_path):
        loader = YAML(typ='safe')
        with open(config_path) as f: config = loader.load(f)
        return cls(
            agent_info=LLMServerInfo(
                base_url=config['agent']['base_url'],
                api_key=config['agent']['api_key'],
                model=config['agent']['model'],
            ),
            agent_max_tokens=config['agent']['max_tokens'],
            evaluation_date=datetime.strftime(config['general']['evaluation_date']),
            num_oracle_papers=config['dynamic_oracle']['num_oracle_papers'],
            llm_server_info=LLMServerInfo(
                base_url=config['external_llm']['base_url'],
                api_key=config['external_llm']['api_key'],
                model=config['external_llm']['model'],
            ),
            llm_num_workers=config['llm']['n_workers'],
            sbert_server_url=config['sbert']['base_url'],
            grobid_url=config['citation_parser']['grobid_url'],
            grobid_num_workers=config['citation_parser']['n_workers'],
            topn=config['source_selection']['topn'],
            topic_similarity_threshold=config['topic_coverage']['topic_similarity_threshold'],
            redundancy_similarity_threshold=config['quality']['redundancy_similarity_threshold'],
            redundancy_ngram=config['quality']['redundancy_ngram'],
        )