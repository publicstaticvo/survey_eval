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
DEBUG_REBUILD_EMPTY_CACHED_MODULES = True

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
    from .tools.scope.topic_coverage import TopicCoverage
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
    from tools.scope.topic_coverage import TopicCoverage
    from tools.scope.uncited_entities import UncitedEntities
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


@dataclass
class SurveyEvaluationAgent:
    config: ToolConfig
    output_dir: str | Path | None = None
    run_modules: set[int] | None = None
    force: bool = False

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
        self.topic_coverage = TopicCoverage(self.config)
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


    def _paper_summary(self, paper: Any) -> Any:
        if not isinstance(paper, dict) or not paper.get("title"):
            return paper
        return {
            "title": paper.get("title", ""),
            "id": paper.get("id") or paper.get("paperId", ""),
            "ids": paper.get("ids", []),
        }

    def _simplify_paper_outputs(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._simplify_paper_outputs(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("title") and any(key in value for key in ("id", "paperId", "ids", "abstract")):
            return self._paper_summary(value)
        simplified = {}
        for key, item in value.items():
            if key == "paper" and isinstance(item, dict):
                simplified[key] = self._paper_summary(item)
            else:
                simplified[key] = self._simplify_paper_outputs(item)
        return simplified

    def _output_view(self, name: str, data: Any) -> Any:
        if name not in {"04_literature_pool", "05_uncited_entities", "08_missing_papers", "09_topic_coverage"}:
            return data
        return self._simplify_paper_outputs(data)

    def _module_outputs(self, names: list[str]) -> dict[str, str]:
        root = self._output_root()
        if root is None:
            return {}
        outputs = {}
        for name in names:
            path = self._module_path(name)
            if path is not None:
                outputs[name] = str(path)
        return outputs

    def _result_count(self, value: Any) -> int | None:
        if isinstance(value, (list, dict, tuple, set)):
            return len(value)
        return None

    def _final_result_summary(
        self,
        query: str | list[str],
        minimum_check: dict[str, Any],
        citation_data: dict[str, Any],
        classified_paper: dict[str, Any],
        reference_surveys: Any,
        literature_pool: dict[str, Any],
        entity_data: dict[str, Any],
        claim_data: dict[str, Any],
        fact_data: dict[str, Any],
        source_data: dict[str, Any],
        topic_data: dict[str, Any],
        internal_data: Any,
        contribution_data: Any,
        aggregate_review: dict[str, Any],
    ) -> dict[str, Any]:
        source_evals = source_data.get("source_evals", {}) if isinstance(source_data, dict) else {}
        topic_evals = topic_data.get("topic_evals", {}) if isinstance(topic_data, dict) else {}
        literature_papers = literature_pool.get("literature_pool", {}) if isinstance(literature_pool, dict) else {}
        citation_graph = literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {}
        counts = {
            "paper_content_map": self._result_count(citation_data.get("paper_content_map", {})),
            "classified_sections": self._result_count(classified_paper.get("sections", [])) if isinstance(classified_paper, dict) else None,
            "reference_surveys": self._result_count(reference_surveys),
            "claims": self._result_count(claim_data.get("claims", [])) if isinstance(claim_data, dict) else None,
            "claim_errors": self._result_count(claim_data.get("errors", [])) if isinstance(claim_data, dict) else None,
            "fact_checks": self._result_count(fact_data.get("fact_checks", [])) if isinstance(fact_data, dict) else None,
            "literature_pool": self._result_count(literature_papers),
            "citation_graph": self._result_count(citation_graph),
            "uncited_entities": self._result_count(entity_data.get("uncited_entities", [])) if isinstance(entity_data, dict) else None,
            "missing_papers": self._result_count(source_evals.get("missing_papers", [])) if isinstance(source_evals, dict) else None,
            "uncited_prospective": self._result_count(source_evals.get("uncited_prospective", {})) if isinstance(source_evals, dict) else None,
            "missing_functional_types": self._result_count(topic_evals.get("missing_functional_types", [])) if isinstance(topic_evals, dict) else None,
            "missing_content_tags": self._result_count(topic_evals.get("missing_content_tags", [])) if isinstance(topic_evals, dict) else None,
            "internal_checks": self._result_count(internal_data.get("checks", [])) if isinstance(internal_data, dict) else None,
            "contribution_checks": self._result_count(contribution_data.get("checks", [])) if isinstance(contribution_data, dict) else None,
            "aggregate_weaknesses": self._result_count(aggregate_review.get("weaknesses", [])) if isinstance(aggregate_review, dict) else None,
            "aggregate_comments": self._result_count(aggregate_review.get("comments", [])) if isinstance(aggregate_review, dict) else None,
        }
        return {
            "query": query,
            "minimum_check": minimum_check,
            "module_outputs": self._module_outputs([
                "00_minimum_check",
                "01_citation_parser",
                "02_classified_paper",
                "03_get_reference_surveys",
                "04_literature_pool",
                "05_uncited_entities",
                "06_claim_segmentation",
                "07_fact_check",
                "08_missing_papers",
                "09_topic_coverage",
                "10_internal_consistency",
                "11_contribution_consistency",
                "12_aggregate_review",
            ]),
            "counts": {key: value for key, value in counts.items() if value is not None},
            "aggregate_review": {
                "weakness_count": counts.get("aggregate_weaknesses", 0),
                "comment_count": counts.get("aggregate_comments", 0),
            },
            "errors": [],
        }

    def _load_module(self, name: str) -> Any:
        path = self._module_path(name)
        if name == "04_literature_pool" and path is not None:
            cache_path = path.with_name("04_literature_pool.cache.json")
            if cache_path.exists():
                path = cache_path
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
        if name == "04_literature_pool":
            cache_path = path.with_name("04_literature_pool.cache.json")
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._output_view(name, data), f, ensure_ascii=False, indent=2, default=str)
        self.logger.info("saved %s", path)

    def _module_number(self, name: str) -> int | None:
        prefix = str(name).split("_", 1)[0]
        return int(prefix) if prefix.isdigit() else None

    def _module_selected(self, name: str) -> bool:
        number = self._module_number(name)
        return self.run_modules is None or number is None or number in self.run_modules

    def _is_empty_cached_module(self, data: Any) -> bool:
        if data is None:
            return True
        if isinstance(data, (str, bytes)):
            return not data
        if isinstance(data, (list, tuple, set)):
            return len(data) == 0
        if isinstance(data, dict):
            return len(data) == 0 or all(self._is_empty_cached_module(value) for value in data.values())
        return False

    async def _run_or_load_module(self, name: str, runner, refresh_cached=None):
        selected = self._module_selected(name)
        cached = self._load_module(name)
        if cached is not _MISSING and DEBUG_REBUILD_EMPTY_CACHED_MODULES and self._is_empty_cached_module(cached):
            self.logger.info("rebuild %s: cached output is empty", name)
            cached = _MISSING
        if cached is not _MISSING and not (self.force and selected):
            if name == "01_citation_parser" and refresh_cached is not None:
                refreshed = refresh_cached(cached)
                if inspect.isawaitable(refreshed):
                    refreshed = await refreshed
                cached = refreshed
                self._save_module(name, cached)
            self.logger.info("skip %s: cached output exists", name)
            return cached
        if not selected:
            self.logger.info("skip %s: not selected by run_modules", name)
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

    async def evaluate(self, query: str | list[str], review_paper: Dict[str, Any], few_shot_examples: Dict[str, str] | None = None):
        queries = query if isinstance(query, list) else [item.strip() for item in str(query or "").split(",") if item.strip()]
        query_text = " ".join(queries)
        self.logger.info("start survey evaluation: %s", query_text)
        cached_result = self._load_module("result")
        if cached_result is not _MISSING and not self.force and self.run_modules is None:
            if not any(key in cached_result for key in ("preprocessing", "evaluations")):
                self.logger.info("skip full evaluation: cached compact result exists")
                return cached_result
            self.logger.info("ignore legacy verbose result cache and rebuild compact result")

        minimum_check = await self._run_or_load_module("00_minimum_check", lambda: self.minimum_completion(review_paper))
        result = {
            "query": query_text,
            "minimum_check": minimum_check["minimum_check"],
            "preprocessing": {},
            "evaluations": {},
            "aggregate_review": None,
            "errors": [],
        }
        if minimum_check["minimum_check"]["status"] != "pass":
            self.logger.info("minimum check failed")
            # return result

        parse_task = asyncio.create_task(
            self._run_or_load_module(
                "01_citation_parser",
                lambda: self.citation_parser(review_paper.get("citations", {})),
                refresh_cached=lambda cached: self.citation_parser.refresh_status3(review_paper.get("citations", {}), cached),
            )
        )
        sentence_task = asyncio.create_task(
            self._run_or_load_module("02_classified_paper", lambda: self.paper_content_classification(query_text, review_paper))
        )
        reference_survey_task = asyncio.create_task(
            self._run_or_load_module("03_get_reference_surveys", lambda: self.get_reference_surveys(query_text))
        )
        citation_data, classified_paper, reference_surveys = await asyncio.gather(parse_task, sentence_task, reference_survey_task)
        paper_content_map = citation_data["paper_content_map"]
        self.logger.info("preprocessing complete: %d citations", len(citation_data.get("paper_content_map", {})))

        literature_pool = await self._run_or_load_module(
            "04_literature_pool",
            lambda: self.literature_pool(query_text, classified_paper, paper_content_map),
        )
        self.logger.info("literature pool complete: %d papers", len(literature_pool.get("literature_pool", {})))

        entity_data = await self._run_or_load_module(
            "05_uncited_entities",
            lambda: self.entity_extractor(classified_paper, paper_content_map=paper_content_map, literature_pool=literature_pool),
        )
        self.logger.info("entity extraction complete: %d uncited entities", len(entity_data.get("uncited_entities", [])))

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
            "08_missing_papers",
            lambda: self.source_critic(
                classified_paper,
                paper_content_map,
                reference_surveys=reference_surveys,
                literature_pool=literature_pool,
                citation_graph=literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {},
                neutral_opinion_claims=neutral_opinion_claims,
                entity_data=entity_data,
            ),
        )
        self.logger.info("missing paper check complete: %d candidates", len(source_data.get("source_evals", {}).get("missing_papers", [])))

        topic_data = await self._run_or_load_module(
            "09_topic_coverage",
            lambda: self.topic_coverage(queries, classified_paper, reference_surveys=reference_surveys),
        )
        self.logger.info("topic coverage complete")

        internal_data = await self._run_or_load_module(
            "10_internal_consistency",
            lambda: self.internal_consistency(classified_paper),
        )
        self.logger.info("internal consistency complete")

        contribution_data = await self._run_or_load_module(
            "11_contribution_consistency",
            lambda: self.contribution_consistency(classified_paper),
        )

        aggregate_input = {
            "query": query_text,
            "minimum_check": minimum_check["minimum_check"],
            "preprocessing": {
                "classified_paper": classified_paper,
            },
            "evaluations": {
                "fact_checks": fact_data.get("fact_checks", []),
                "source_evals": source_data.get("source_evals", {}),
                "topic_evals": topic_data.get("topic_evals", {}),
                "internal_evals": internal_data,
                "contribution_evals": contribution_data,
            },
            "errors": [],
        }
        aggregate_review = await self._run_or_load_module(
            "12_aggregate_review",
            lambda: self.final_aggregate(aggregate_input),
        )
        result = self._final_result_summary(
            query=query_text,
            minimum_check=minimum_check["minimum_check"],
            citation_data=citation_data,
            classified_paper=classified_paper,
            reference_surveys=reference_surveys,
            literature_pool=literature_pool,
            entity_data=entity_data,
            claim_data=claim_data,
            fact_data=fact_data,
            source_data=source_data,
            topic_data=topic_data,
            internal_data=internal_data,
            contribution_data=contribution_data,
            aggregate_review=aggregate_review,
        )
        self._save_module("result", result)
        self.logger.info("survey evaluation complete")
        return result


async def evaluate_survey(
    query: str | list[str],
    review_paper: Dict[str, Any],
    config: ToolConfig | None = None,
    few_shot_examples=None,
    output_dir: str | Path | None = None,
    run_modules: str | set[int] | None = None,
    force: bool = False,
):
    config = config or ToolConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await SessionManager.init()
    try:
        return await evaluate_survey_with_session(
            query,
            review_paper,
            config=config,
            few_shot_examples=few_shot_examples,
            output_dir=output_dir,
            run_modules=run_modules,
            force=force,
        )
    finally:
        await SessionManager.close()


async def evaluate_survey_with_session(
    query: str | list[str],
    review_paper: Dict[str, Any],
    config: ToolConfig | None = None,
    few_shot_examples=None,
    output_dir: str | Path | None = None,
    run_modules: str | set[int] | None = None,
    force: bool = False,
):
    config = config or ToolConfig()
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / stamp
    module_set = None if run_modules in (None, "") else ({int(item.strip()) for item in str(run_modules).split(",") if item.strip()} if not isinstance(run_modules, set) else run_modules)
    agent = SurveyEvaluationAgent(config, output_dir=output_dir, run_modules=module_set, force=force)
    return await agent.evaluate(query, review_paper, few_shot_examples=few_shot_examples)
