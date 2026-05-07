from importlib import import_module


_EXPORTS = {
    "AgentState": ".utility.agent_state",
    "FinalAggregate": ".aggregate_review",
    "ArgumentStructureEvaluator": ".eval.argument_eval",
    "AnchorSurveySource": ".preprocess.build_sources",
    "AnchorSurveyFetch": ".preprocess.build_sources",
    "ClaimSegmentation": ".preprocess.claim_segmentation",
    "CitationParser": ".preprocess.citation_parser",
    "DirectSeedGraphSource": ".preprocess.build_sources",
    "DynamicCandidatePool": ".preprocess.dynamic_candidate_pool",
    "FactualCorrectnessCritic": ".eval.fact_check",
    "GoldenTopicGenerator": ".golden_topics",
    "TopicRecord": ".golden_topics",
    "SelfScopeEvidenceExtractor": ".topic_self_evidence",
    "LetorCandidateScorer": ".preprocess.dynamic_candidate_pool",
    "LiteratureCandidateDeduplicate": ".preprocess.dynamic_candidate_pool",
    "minimum_completion": ".eval.minimum_completion",
    "QualityCritic": ".eval.programmatic_quality",
    "QueryExpand": ".preprocess.query_expand",
    "MissingPaperCheck": ".missing_papers",
    "RecentSemanticSource": ".preprocess.build_sources",
    "StructureCheck": ".eval.structure_eval",
    "TopicCoverageCritic": ".topic_coverage",
    "MissingTopicLLMClient": ".topic_coverage",
    "ToolConfig": ".utility.tool_config",
    "WebSearchFallback": ".preprocess.websearch",
    "PaperParser": ".utility.grobidpdf.paper_parser",
    "SentenceTransformerClient": ".utility.sbert_client",
    "SemanticScholar": ".utility.s2",
    "SessionManager": ".utility.request_utils",
    "get_academic_engine": ".utility.academic_engine",
    "LatexPaperParser": ".utility.latex_parser.tex_parser",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
