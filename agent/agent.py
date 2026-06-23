from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from .tools.aggregate_review import FinalAggregate
    from .tools.contribution.contribution_consistent import ContributionConsistency
    from .tools.contribution.internal_consistent import InternalConsistency
    from .tools.fact.citation_check import CitationCorrectnessCheck
    from .tools.fact.fact_check import CitedClaimVerifier, FactualCorrectnessCritic
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.preprocess.literature_pool import BuildLiteraturePool
    from .tools.preprocess.minimum_completion import minimum_completion
    from .tools.preprocess.sentences import SentenceClassification
    from .tools.eval.programmatic_quality import QualityCritic
    from .tools.scope.missing_papers import MissingPaperCheck
    from .tools.scope.topic_coverage import TopicCoverageCritic
    from .tools.scope.uncited_entities import UncitedEntities
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
except ImportError:
    from tools.aggregate_review import FinalAggregate
    from tools.contribution.contribution_consistent import ContributionConsistency
    from tools.contribution.internal_consistent import InternalConsistency
    from tools.fact.citation_check import CitationCorrectnessCheck
    from tools.fact.fact_check import CitedClaimVerifier, FactualCorrectnessCritic
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.preprocess.literature_pool import BuildLiteraturePool
    from tools.preprocess.minimum_completion import minimum_completion
    from tools.preprocess.sentences import SentenceClassification
    from tools.eval.programmatic_quality import QualityCritic
    from tools.scope.missing_papers import MissingPaperCheck
    from tools.scope.topic_coverage import TopicCoverageCritic
    from tools.scope.uncited_entities import UncitedEntities
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


class GoldenTopicGenerator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GoldenTopicGenerator is not available in this worktree")


@dataclass
class SurveyEvaluationAgent:
    config: ToolConfig
    output_dir: str | Path | None = None

    def __post_init__(self):
        self._normalize_paths()
        self.logger = logging.getLogger(__name__)
        self.minimum_completion = minimum_completion
        try:
            self.golden_topics = GoldenTopicGenerator(self.config)
        except Exception:
            self.golden_topics = None
        self.citation_parser = CitationParser(self.config)
        self.citation_check = CitationCorrectnessCheck()
        self.sentence_classification = SentenceClassification(self.config)
        self.claim_segmentation = ClaimSegmentation(self.config)
        self.fact_check = FactualCorrectnessCritic(self.config)
        self.cited_claim_verifier = CitedClaimVerifier(self.config)
        self.literature_pool = BuildLiteraturePool(self.config)
        self.entity_extractor = UncitedEntities(self.config)
        self.source_critic = MissingPaperCheck(self.config)
        self.topic_coverage = TopicCoverageCritic(self.config)
        self.quality_eval = QualityCritic(self.config)
        self.contribution_consistency = ContributionConsistency(self.config)
        self.internal_consistency = InternalConsistency(self.config)
        self.final_aggregate = FinalAggregate()

    def _normalize_paths(self):
        letor_path = Path(self.config.letor_path)
        if not letor_path.is_absolute():
            candidate = Path(__file__).resolve().parent / letor_path
            if candidate.exists():
                object.__setattr__(self.config, "letor_path", str(candidate))

    def _output_root(self) -> Path | None:
        if self.output_dir is None:
            return None
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _save_module(self, name: str, data: Any):
        root = self._output_root()
        if root is None:
            return
        path = root / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info("saved %s", path)

    async def _safe_golden_topics(self, query: str, review_paper: dict[str, Any]) -> dict[str, Any]:
        if self.golden_topics is None:
            return {"query": query, "reference_data": {}, "reference_topics": [], "self_topics": {}}
        return await self.golden_topics(query, review_paper)

    def _extract_reference_surveys(self, golden_topic_data: dict[str, Any]) -> Any:
        return (golden_topic_data.get("reference_data") or {}).get("reference_surveys")

    async def evaluate(self, query: str, review_paper: Dict[str, Any], few_shot_examples: Dict[str, str] | None = None):
        self.logger.info("start survey evaluation: %s", query)
        minimum_check = self.minimum_completion(review_paper)
        result = {
            "query": query,
            "minimum_check": minimum_check["minimum_check"],
            "preprocessing": {},
            "evaluations": {},
            "aggregate_review": None,
            "errors": [],
        }
        self._save_module("00_minimum_check", minimum_check)
        if minimum_check["minimum_check"]["status"] != "pass":
            self.logger.info("minimum check failed")
            return result

        parse_task = asyncio.create_task(self.citation_parser(review_paper.get("citations", {})))
        sentence_task = asyncio.create_task(self.sentence_classification(review_paper))
        quality_task = asyncio.create_task(self.quality_eval._run(review_paper))
        topic_task = asyncio.create_task(self._safe_golden_topics(query, review_paper))
        citation_data, classified_paper, quality_data, golden_topic_data = await asyncio.gather(
            parse_task,
            sentence_task,
            quality_task,
            topic_task,
        )
        self.logger.info("preprocessing complete: %d citations", len(citation_data.get("paper_content_map", {})))
        self._save_module("01_citation_parser", citation_data)
        self._save_module("02_classified_paper", classified_paper)
        self._save_module("03_quality", quality_data)
        self._save_module("04_golden_topics", golden_topic_data)

        paper_content_map = citation_data["paper_content_map"]
        citation_correctness = await self.citation_check(review_paper.get("citations", {}), paper_content_map)
        self.logger.info("citation check complete")
        self._save_module("05_citation_check", citation_correctness)

        fact_data = await self.cited_claim_verifier(classified_paper, paper_content_map)
        self.logger.info("fact verification complete: %d targets", fact_data.get("checked_count", 0))
        self._save_module("06_fact_check", fact_data)

        literature_pool = await self.literature_pool(query, classified_paper, paper_content_map)
        self.logger.info("literature pool complete: %d papers", len(literature_pool.get("literature_pool", {})))
        self._save_module("07_literature_pool", literature_pool)

        entity_data = await self.entity_extractor(classified_paper)
        self.logger.info("entity extraction complete: %d uncited entities", len(entity_data.get("uncited_entities", [])))
        self._save_module("08_uncited_entities", entity_data)

        reference_surveys = self._extract_reference_surveys(golden_topic_data)
        source_data = await self.source_critic(
            classified_paper,
            paper_content_map,
            reference_surveys=reference_surveys,
            literature_pool=literature_pool,
            neutral_opinion_claims=fact_data.get("neutral_opinion_claims", []),
            entity_data=entity_data,
        )
        self.logger.info("missing paper check complete: %d candidates", len(source_data.get("source_evals", {}).get("missing_papers", [])))
        self._save_module("09_missing_papers", source_data)

        topic_data = await self.topic_coverage(query, classified_paper, reference_surveys=reference_surveys)
        self.logger.info("topic coverage complete")
        self._save_module("10_topic_coverage", topic_data)

        internal_data = await self.internal_consistency(classified_paper)
        self.logger.info("internal consistency complete")
        self._save_module("11_internal_consistency", internal_data)

        claim_data = await self.claim_segmentation(classified_paper)
        self._save_module("12_claim_segmentation", claim_data)

        contribution_data = {}
        if golden_topic_data.get("self_topics"):
            try:
                contribution_data = self.contribution_consistency(classified_paper, golden_topic_data.get("self_topics", {}))
            except Exception as exc:
                contribution_data = {"checks": [], "consistent": True, "error": str(exc)}
        self._save_module("13_contribution_consistency", contribution_data)

        result["preprocessing"] = {
            "golden_topics": golden_topic_data,
            "paper_content_map": paper_content_map,
            "classified_paper": classified_paper,
            "claims": claim_data.get("claims", []),
            "claim_errors": claim_data.get("errors", []),
            "literature_pool": literature_pool,
            "entity_data": entity_data,
        }
        result["evaluations"] = {
            "fact_checks": fact_data.get("fact_checks", []),
            "source_evals": source_data.get("source_evals", {}),
            "topic_evals": topic_data.get("topic_evals", {}),
            "internal_evals": internal_data,
            "contribution_evals": contribution_data,
            "quality_evals": quality_data.get("quality_evals", {}),
            "citation_evals": citation_correctness.get("citation_evals", {}),
        }
        result["aggregate_review"] = self.final_aggregate(result)
        self._save_module("14_aggregate_review", result["aggregate_review"])
        self._save_module("result", result)
        self.logger.info("survey evaluation complete")
        return result


async def evaluate_survey(
    query: str,
    review_paper: Dict[str, Any],
    config: ToolConfig | None = None,
    few_shot_examples=None,
    output_dir: str | Path | None = None,
):
    config = config or ToolConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await SessionManager.init()
    try:
        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs") / stamp
        agent = SurveyEvaluationAgent(config, output_dir=output_dir)
        return await agent.evaluate(query, review_paper, few_shot_examples=few_shot_examples)
    finally:
        await SessionManager.close()