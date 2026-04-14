from importlib import import_module


_EXPORTS = {
    "AgentState": ".agent_state",
    "FinalAggregate": ".aggregate_review",
    "ArgumentStructureEvaluator": ".argument_eval",
    "AnchorSurveyFetch": ".anchor_surveys",
    "ClaimSegmentation": ".claim_segmentation",
    "CitationParser": ".citation_parser",
    "DynamicOracleGenerator": ".dynamic_oracle_generator",
    "FactualCorrectnessCritic": ".fact_check",
    "GoldenTopicGenerator": ".golden_topics",
    "MinimalCompletionCheck": ".minimum_completion",
    "QualityCritic": ".programmatic_quality",
    "QueryExpand": ".query_expand",
    "MissingPaperCheck": ".source_critic",
    "StructureCheck": ".structure_eval",
    "TopicCoverageCritic": ".topic_coverage",
    "ToolConfig": ".tool_config",
    "WebSearchFallback": ".websearch",
    "PaperParser": ".grobidpdf.paper_parser",
    "SentenceTransformerClient": ".sbert_client",
    "SessionManager": ".request_utils",
    "LatexPaperParser": ".latex_parser.tex_parser",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
