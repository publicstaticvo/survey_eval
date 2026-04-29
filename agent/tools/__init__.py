from importlib import import_module


_EXPORTS = {
    "AgentState": ".utility.agent_state",
    "FinalAggregate": ".aggregate_review",
    "ArgumentStructureEvaluator": ".eval.argument_eval",
    "AnchorSurveyFetch": ".preprocess.anchor_surveys",
    "ClaimSegmentation": ".preprocess.claim_segmentation",
    "CitationParser": ".preprocess.citation_parser",
    "DynamicCandidatePool": ".preprocess.dynamic_candidate_pool",
    "FactualCorrectnessCritic": ".eval.fact_check",
    "GoldenTopicGenerator": ".preprocess.golden_topics",
    "minimum_completion": ".eval.minimum_completion",
    "QualityCritic": ".eval.programmatic_quality",
    "QueryExpand": ".preprocess.query_expand",
    "MissingPaperCheck": ".eval.missing_papers",
    "StructureCheck": ".eval.structure_eval",
    "TopicCoverageCritic": ".eval.topic_coverage",
    "ToolConfig": ".utility.tool_config",
    "WebSearchFallback": ".preprocess.websearch",
    "PaperParser": ".utility.grobidpdf.paper_parser",
    "SentenceTransformerClient": ".utility.sbert_client",
    "SessionManager": ".utility.request_utils",
    "LatexPaperParser": ".utility.latex_parser.tex_parser",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
