from importlib import import_module


_EXPORTS = {
    "FinalAggregate": ".aggregate_review",
    "GetReferenceSurveys": ".preprocess.get_reference_surveys",
    "BuildLiteraturePool": ".preprocess.literature_pool",
    "ClaimSegmentation": ".fact.claim_segmentation",
    "CitationCorrectnessCheck": ".fact.citation_check",
    "SentenceClassification": ".preprocess.sentences",
    "ContributionClassification": ".preprocess.contribution_classify",
    "CitationParser": ".preprocess.citation_parser",
    "ContributionConsistency": ".contribution.contribution_consistent",
    "InternalConsistency": ".contribution.internal_consistent",
    "FactualCorrectnessCritic": ".fact.fact_check",
    "SelfScopeEvidenceExtractor": ".preprocess.extract_scope",
    "minimum_completion": ".preprocess.minimum_completion",
    "QualityCritic": ".eval.programmatic_quality",
    "MissingPaperCheck": ".scope.missing_papers",
    "TopicCoverage": ".scope.topic_coverage",
    "TopicCoverageCritic": ".scope.topic_coverage",
    "QueryExpand": ".scope.topic_papers",
    "TopicSpecificPapers": ".scope.topic_papers",
    "FindAllEntities": ".scope.uncited_entities",
    "UncitedEntities": ".scope.uncited_entities",
    "UncitedProspective": ".scope.uncited_prospective",
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
