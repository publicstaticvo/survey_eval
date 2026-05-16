from importlib import import_module


_EXPORTS = {
    "AgentState": ".utility.agent_state",
    "FinalAggregate": ".aggregate_review",
    "ArgumentStructureEvaluator": ".eval.argument_eval",
    "GetReferenceSurveys": ".preprocess.get_reference_surveys",
    "ClaimSegmentation": ".preprocess.claim_segmentation",
    "CitationParser": ".preprocess.citation_parser",
    "FactualCorrectnessCritic": ".eval.fact_check",
    "GoldenTopicGenerator": ".preprocess.golden_topics",
    "SelfScopeEvidenceExtractor": ".preprocess.extract_scope",
    "minimum_completion": ".eval.minimum_completion",
    "QualityCritic": ".eval.programmatic_quality",
    "MissingPaperCheck": ".eval.missing_papers",
    "StructureCheck": ".eval.structure_eval",
    "TopicCoverageCritic": ".eval.topic_coverage",
    "MissingTopicLLMClient": ".eval.topic_coverage",
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
