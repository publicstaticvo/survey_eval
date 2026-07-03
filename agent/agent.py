from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_MISSING = object()

try:
    from .tools.aggregate_review import FinalAggregate
    from .tools.contribution.contribution_consistent import ContributionConsistency
    from .tools.contribution.internal_consistent import InternalConsistency
    from .tools.fact.fact_check import CitedClaimVerifier
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.preprocess.get_reference_surveys import GetReferenceSurveys
    from .tools.preprocess.literature_pool import BuildLiteraturePool
    from .tools.preprocess.minimum_completion import minimum_completion
    from .tools.preprocess.paper_content_classify import PaperContentClassification
    from .tools.scope.missing_papers import MissingPaperCheck
    from .tools.scope.topic_coverage import TopicCoverageCritic
    from .tools.scope.uncited_entities import UncitedEntities
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
except ImportError:
    from tools.aggregate_review import FinalAggregate
    from tools.contribution.contribution_consistent import ContributionConsistency
    from tools.contribution.internal_consistent import InternalConsistency
    from tools.fact.fact_check import CitedClaimVerifier
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.preprocess.literature_pool import BuildLiteraturePool
    from tools.preprocess.get_reference_surveys import GetReferenceSurveys
    from tools.preprocess.minimum_completion import minimum_completion
    from tools.preprocess.paper_content_classify import PaperContentClassification
    from tools.scope.missing_papers import MissingPaperCheck
    from tools.scope.topic_coverage import TopicCoverageCritic
    from tools.scope.uncited_entities import UncitedEntities
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


@dataclass
class SurveyEvaluationAgent:
    config: ToolConfig
    output_dir: str | Path | None = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.minimum_completion = minimum_completion
        self.citation_parser = CitationParser(self.config)
        self.paper_content_classification = PaperContentClassification(self.config)
        self.get_reference_surveys = GetReferenceSurveys(self.config)
        self.claim_segmentation = ClaimSegmentation(self.config)
        self.cited_claim_verifier = CitedClaimVerifier(self.config)
        self.literature_pool = BuildLiteraturePool(self.config)
        self.entity_extractor = UncitedEntities(self.config)
        self.source_critic = MissingPaperCheck(self.config)
        self.topic_coverage = TopicCoverageCritic(self.config)
        self.contribution_consistency = ContributionConsistency(self.config)
        self.internal_consistency = InternalConsistency(self.config)
        self.final_aggregate = FinalAggregate()

    def _output_root(self) -> Path | None:
        if self.output_dir is None:
            return None
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _module_path(self, name: str) -> Path | None:
        root = self._output_root()
        if root is None:
            return None
        return root / f"{name}.json"

    def _load_module(self, name: str) -> Any:
        path = self._module_path(name)
        if path is None or not path.exists():
            return _MISSING
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.logger.info("loaded cached %s", path)
        return data

    def _save_module(self, name: str, data: Any):
        path = self._module_path(name)
        if path is None:
            return
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info("saved %s", path)

    async def _run_or_load_module(self, name: str, runner):
        cached = self._load_module(name)
        if cached is not _MISSING:
            self.logger.info("skip %s: cached output exists", name)
            return cached
        data = runner()
        if inspect.isawaitable(data):
            data = await data
        self._save_module(name, data)
        return data


    def _neutral_opinion_claims(self, fact_data: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "claim": item.get("claim", ""),
                "claim_type": item.get("claim_type", ""),
                "citation_keys": item.get("citation_keys", []),
                "references": item.get("references", []),
            }
            for item in fact_data.get("fact_checks", [])
            if item.get("judgment") == "NEUTRAL" and item.get("reason", "") == ""
        ]

    async def evaluate(self, query: str, review_paper: Dict[str, Any], few_shot_examples: Dict[str, str] | None = None):
        self.logger.info("start survey evaluation: %s", query)
        cached_result = self._load_module("result")
        if cached_result is not _MISSING:
            self.logger.info("skip full evaluation: cached result exists")
            return cached_result

        minimum_check = await self._run_or_load_module("00_minimum_check", lambda: self.minimum_completion(review_paper))
        result = {
            "query": query,
            "minimum_check": minimum_check["minimum_check"],
            "preprocessing": {},
            "evaluations": {},
            "aggregate_review": None,
            "errors": [],
        }
        if minimum_check["minimum_check"]["status"] != "pass":
            self.logger.info("minimum check failed")
            return result

        parse_task = asyncio.create_task(
            self._run_or_load_module("01_citation_parser", lambda: self.citation_parser(review_paper.get("citations", {})))
        )
        sentence_task = asyncio.create_task(
            self._run_or_load_module("02_classified_paper", lambda: self.paper_content_classification(query, review_paper))
        )
        reference_survey_task = asyncio.create_task(
            self._run_or_load_module("03_get_reference_surveys", lambda: self.get_reference_surveys(query))
        )
        citation_data, classified_paper, reference_surveys = await asyncio.gather(parse_task, sentence_task, reference_survey_task)
        paper_content_map = citation_data["paper_content_map"]
        self.logger.info("preprocessing complete: %d citations", len(citation_data.get("paper_content_map", {})))

        literature_pool = await self._run_or_load_module(
            "08_literature_pool",
            lambda: self.literature_pool(query, classified_paper, paper_content_map),
        )
        self.logger.info("literature pool complete: %d papers", len(literature_pool.get("literature_pool", {})))

        entity_data = await self._run_or_load_module(
            "05_uncited_entities",
            lambda: self.entity_extractor(classified_paper, paper_content_map=paper_content_map, literature_pool=literature_pool),
        )
        self.logger.info("entity extraction complete: %d uncited entities", len(entity_data.get("uncited_entities", [])))
        # self._save_module("02_classified_paper", classified_paper)

        claim_data = await self._run_or_load_module(
            "06_claim_segmentation",
            lambda: self.claim_segmentation(classified_paper),
        )

        fact_data = await self._run_or_load_module(
            "07_fact_check",
            lambda: self.cited_claim_verifier(claim_data.get("claims", []), paper_content_map),
        )
        self.logger.info("fact verification complete: %d targets", fact_data.get("checked_count", 0))

        neutral_opinion_claims = self._neutral_opinion_claims(fact_data)
        source_data = await self._run_or_load_module(
            "09_missing_papers",
            lambda: self.source_critic(
                classified_paper,
                paper_content_map,
                reference_surveys=reference_surveys,
                literature_pool=literature_pool,
                neutral_opinion_claims=neutral_opinion_claims,
                entity_data=entity_data,
            ),
        )
        self.logger.info("missing paper check complete: %d candidates", len(source_data.get("source_evals", {}).get("missing_papers", [])))

        topic_data = await self._run_or_load_module(
            "10_topic_coverage",
            lambda: self.topic_coverage(query, classified_paper, reference_surveys=reference_surveys),
        )
        self.logger.info("topic coverage complete")

        internal_data = await self._run_or_load_module(
            "11_internal_consistency",
            lambda: self.internal_consistency(classified_paper),
        )
        self.logger.info("internal consistency complete")

        contribution_data = await self._run_or_load_module(
            "13_contribution_consistency",
            lambda: self.contribution_consistency(classified_paper),
        )

        result["preprocessing"] = {
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
        }
        result["aggregate_review"] = await self._run_or_load_module(
            "14_aggregate_review",
            lambda: self.final_aggregate(result),
        )
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




