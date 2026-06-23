from dataclasses import dataclass, field
from collections.abc import Mapping
from datetime import datetime
from typing import Any
import yaml


GREEDY_PARAMS = {
    'temperature': 0.0, "max_tokens": 16384, "seed": 42,
    "top_p": 1.0,      # 璁剧疆涓?锛屼笉杩涜鏍搁噰鏍?
    "top_k": 1,        # 鎴栬缃负1锛岀‘淇濇€绘槸閫夋嫨鏈€鍙兘鐨則oken
    "repetition_penalty": 1.0,  # 璁剧疆涓?锛岀鐢ㄩ噸澶嶆儵缃?
    "length_penalty": 1.0,      # 璁剧疆涓?锛岀鐢ㄩ暱搴︽儵缃?
    "no_repeat_ngram_size": 0,  # 璁剧疆涓?锛岀鐢╪-gram閲嶅鎯╃綒
}


@dataclass(frozen=True)
class LLMServerInfo:
    base_url: str = "https://uni-api.cstcloud.cn"
    api_key: str = "7868485c0ca1d66880fdb72e06b77ebfc6daf07faf61638e81e7a79adf7e309d"
    model: str = "deepseek-v4-flash"


@dataclass(frozen=True)
class ToolConfig:
    # Agent
    agent_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    agent_max_tokens: int = 16384
    # General
    evaluation_date: datetime = field(default_factory=lambda: datetime.strptime("2026-06-30", "%Y-%m-%d"))
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
    grobid_parse_mode: str = "casual"
    proxy_url: str = "http://localhost:7890"
    arxiv_proxy_url: str = "http://localhost:7890"
    # factual correctness - reranking
    rerank_server_info: LLMServerInfo = field(default_factory=LLMServerInfo)
    rerank_n_documents: int = 5
    # source selection
    topn: int = 0
    # topic coverage
    topic_weak_sim_threshold: float = 0.45
    topic_sim_threshold: float = 0.55
    topic_papers_search_limit: int = 10
    topic_coverage_search_limit: int = 10
    new_paper_topic_similarity_threshold: float = 0.55
    new_paper_reference_overlap_threshold: float = 0.6
    citation_velocity_keep_ratio: float = 0.4
    minimum_reference_survey_citations: int = 10
    use_openalex_count_by_year: bool = True
    # fact check
    background_reference_similarity_threshold: float = 0.6
    mean_cov_weight: float = 0.7
    non_compat_punishment: float = 0.6
    confidence_threshold: float = 0.6
    contribution_similarity_threshold: float = 0.65
    internal_consistency_sentence_ratio_threshold: float = 0.4
    # quality
    sentence_similarity_threshold: float = 0.97
    paragraph_similarity_threshold: float = 0.92
    redundancy_ngram: int = 5
    # websearch
    websearch_url: str = "https://google.serper.dev/search"
    websearch_apikey: str = "6a58924ede5e53c3e3d72ef428236db7654b88ec"
    # openalex
    openalex_rate_limit_enabled: bool = True
    openalex_requests_per_second: float = 100.0
    openalex_max_concurrency: int = 3
    openalex_api_keys: list[str] = field(default_factory=list)
    default_academic_search_engine: str = "semantic scholar"
    semantic_scholar_api_key: str = ""

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f: config = yaml.safe_load(f)
        evaluation_date = config['general']['evaluation_date']
        if isinstance(evaluation_date, datetime):
            parsed_evaluation_date = evaluation_date
        else:
            parsed_evaluation_date = datetime.strptime(str(evaluation_date), "%Y-%m-%d")
        proxy_url = (
            config.get('proxy', {}).get('url')
            or config.get('general', {}).get('proxy_url')
            or config.get('citation_parser', {}).get('proxy_url')
            or config.get('citation_parser', {}).get('arxiv_proxy_url')
            or 'http://localhost:7890'
        )
        return cls(
            agent_info=LLMServerInfo(
                base_url=config['agent']['base_url'],
                api_key=config['agent']['api_key'],
                model=config['agent']['model'],
            ),
            agent_max_tokens=config['agent']['max_tokens'],
            evaluation_date=parsed_evaluation_date,
            num_oracle_papers=config['dynamic_oracle']['num_oracle_papers'],
            letor_path=config['dynamic_oracle']['letor_path'],
            llm_server_info=LLMServerInfo(
                base_url=config['external_llm']['base_url'],
                api_key=config['external_llm']['api_key'],
                model=config['external_llm']['model'],
            ),
            sbert_server_url=config['sbert']['base_url'],
            grobid_url=config['citation_parser']['grobid_url'],
            grobid_parse_mode=config.get('citation_parser', {}).get('grobid_parse_mode', 'casual'),
            proxy_url=proxy_url,
            arxiv_proxy_url=proxy_url,
            rerank_server_info=LLMServerInfo(
                base_url=config['rerank']['base_url'],
                api_key=config['rerank']['api_key'],
                model=config['rerank']['model'],
            ),
            rerank_n_documents=config['rerank']['num_documents'],
            topn=config['source_selection']['topn'],
            topic_weak_sim_threshold=config['topic_coverage']['topic_weak_sim_threshold'],
            topic_sim_threshold=config['topic_coverage']['topic_sim_threshold'],
            topic_papers_search_limit=config.get('topic_papers', {}).get('search_limit', 10),
            topic_coverage_search_limit=config.get('topic_coverage', {}).get('search_limit', 10),
            new_paper_topic_similarity_threshold=config.get('topic_coverage', {}).get('new_paper_topic_similarity_threshold', 0.55),
            new_paper_reference_overlap_threshold=config.get('source_selection', {}).get('new_paper_reference_overlap_threshold', 0.6),
            citation_velocity_keep_ratio=config.get('source_selection', {}).get('citation_velocity_keep_ratio', 0.4),
            minimum_reference_survey_citations=config.get('source_selection', {}).get('minimum_reference_survey_citations', 10),
            use_openalex_count_by_year=config.get('source_selection', {}).get('use_openalex_count_by_year', True),
            background_reference_similarity_threshold=config.get('fact_check', {}).get(
                'background_reference_similarity_threshold',
                0.6,
            ),
            sentence_similarity_threshold=config['quality']['sentence_similarity_threshold'],
            paragraph_similarity_threshold=config['quality']['paragraph_similarity_threshold'],
            redundancy_ngram=config['quality']['redundancy_ngram'],
            websearch_url=config['websearch']['url'],
            websearch_apikey=config['websearch']['api_key'],
            openalex_rate_limit_enabled=config.get('openalex', {}).get('rate_limit_enabled', True),
            openalex_requests_per_second=config.get('openalex', {}).get('requests_per_second', 100.0),
            openalex_max_concurrency=config.get('openalex', {}).get('max_concurrency', 3),
            openalex_api_keys=config.get('openalex', {}).get('api_keys', []),
            default_academic_search_engine=config.get('academic_search', {}).get(
                'default_engine',
                config.get('default_academic_search_engine', 'openalex'),
            ),
            semantic_scholar_api_key=config.get('semantic_scholar', {}).get('api_key', ''),
        )
