from importlib import import_module


_EXPORTS = {
    "FinalAggregate": ".aggregate_review",
    "ContributionConsistency": ".contribution.contribution_consistent",
    "InternalConsistency": ".contribution.internal_consistent",
    "CitationCorrectnessCheck": ".fact.citation_check",
    "SingleFactCorrectness": ".fact.fact_check_single",
    "ClaimVerifier": ".fact.fact_check",
    "UncitedClaimVerifier": ".fact.uncited_claims",
    "ClaimSegmentation": ".preprocess.claim_segmentation",
    "CitationParser": ".preprocess.citation_parser",
    "ContributionClassification": ".preprocess.contribution_classify",
    "FindAllEntities": ".preprocess.find_all_entities",
    "GetReferenceSurveys": ".preprocess.get_reference_surveys",
    "BuildLiteraturePool": ".preprocess.literature_pool",
    "SentenceClassification": ".preprocess.sentences",
    "PaperContentClassification": ".preprocess.paper_content_classify",
    "minimum_completion": ".preprocess.minimum_completion",
    "MissingPaperCheck": ".scope.missing_papers",
    "TopicCoverage": ".scope.topic_coverage",
    "UncitedEntities": ".scope.uncited_entities",
    "ToolConfig": ".utility.tool_config",
    "WebSearchFallback": ".preprocess.websearch",
    "XMLPaperParser": ".utility.xml_parser",
    "SentenceTransformerClient": ".utility.sbert_client",
    "SemanticScholar": ".utility.s2",
    "SessionManager": ".utility.request_utils",
    "get_academic_engine": ".utility.academic_engine",
    "LatexPaperParser": ".utility.latex_parser.tex_parser",
    "Paper": ".utility.paper_elements",
    "Section": ".utility.paper_elements",
    "Sentence": ".utility.paper_elements",
    "Paragraph": ".utility.paper_elements",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
