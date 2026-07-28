from __future__ import annotations

import asyncio
import inspect
import json, re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_MISSING = object()
DEBUG_REBUILD_EMPTY_CACHED_MODULES = True
PAPER_CLASSIFICATION_STEPS = {
    "2.1": "sentence",
    "2.2": "section",
    "2.3": "content",
    "2.4": "contribution",
    "2.5": "entities",
}

try:
    from .tools.aggregate_review import FinalAggregate
    from .tools.contribution.contribution_consistent import ContributionConsistency
    from .tools.contribution.internal_consistent import InternalConsistency
    from .tools.fact.fact_check import ClaimVerifier
    from .tools.preprocess.citation_parser import CitationParser
    from .tools.preprocess.claim_segmentation import ClaimSegmentation
    from .tools.preprocess.get_reference_surveys import GetReferenceSurveys
    from .tools.preprocess.literature_pool import BuildLiteraturePool
    from .tools.preprocess.minimum_completion import minimum_completion
    from .tools.preprocess.paper_content_classify import PaperContentClassification
    from .tools.scope.missing_papers import MissingPaperCheck
    from .tools.scope.topic_coverage import TopicCoverage
    from .tools.scope.uncited_entities import UncitedEntities
    from .tools.utility.paper_elements import Paper
    from .tools.utility.request_utils import SessionManager
    from .tools.utility.tool_config import ToolConfig
except ImportError:
    from tools.aggregate_review import FinalAggregate
    from tools.contribution.contribution_consistent import ContributionConsistency
    from tools.contribution.internal_consistent import InternalConsistency
    from tools.fact.fact_check import ClaimVerifier
    from tools.preprocess.citation_parser import CitationParser
    from tools.preprocess.claim_segmentation import ClaimSegmentation
    from tools.preprocess.literature_pool import BuildLiteraturePool
    from tools.preprocess.get_reference_surveys import GetReferenceSurveys
    from tools.preprocess.minimum_completion import minimum_completion
    from tools.preprocess.paper_content_classify import PaperContentClassification
    from tools.scope.missing_papers import MissingPaperCheck
    from tools.scope.topic_coverage import TopicCoverage
    from tools.scope.uncited_entities import UncitedEntities
    from tools.utility.paper_elements import Paper
    from tools.utility.request_utils import SessionManager
    from tools.utility.tool_config import ToolConfig


@dataclass
class SurveyEvaluationAgent:
    config: ToolConfig
    output_dir: str | Path | None = None
    run_modules: set[str] | None = None
    force: bool = False

    def __post_init__(self):
        self.minimum_completion = minimum_completion
        self.citation_parser = CitationParser(self.config)
        self.paper_content_classification = PaperContentClassification(self.config)
        self.get_reference_surveys = GetReferenceSurveys(self.config)
        self.claim_segmentation = ClaimSegmentation(self.config)
        self.claim_verifier = ClaimVerifier(self.config)
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

    def _strip_private_keys(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._strip_private_keys(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self._strip_private_keys(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        return value

    def _output_view(self, name: str, data: Any) -> Any:
        if name == "06_claim_segmentation":
            return self._strip_private_keys(data)
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
        classified_paper: Paper,
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
            "classified_sections": self._result_count(classified_paper.children),
            "reference_surveys": self._result_count(reference_surveys),
            "claims": self._result_count(claim_data.get("claims", [])) if isinstance(claim_data, dict) else None,
            "claim_errors": self._result_count(claim_data.get("errors", [])) if isinstance(claim_data, dict) else None,
            "fact_checks": self._result_count(fact_data.get("fact_checks", [])) if isinstance(fact_data, dict) else None,
            "literature_pool": self._result_count(literature_papers),
            "citation_graph": self._result_count(citation_graph),
            "uncited_entities": self._result_count(entity_data.get("uncited_entities", [])) if isinstance(entity_data, dict) else None,
            "missing_papers": self._result_count(source_evals.get("missing_papers", [])) if isinstance(source_evals, dict) else None,
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
    
    def _organize_queries(self, query: str | list[str]) -> list[str]:
        values = query if isinstance(query, list) else str(query or "").split(",")
        return [
            re.sub(r"\s+", " ", str(value or "")).strip().casefold()
            for value in values
            if str(value or "").strip()
        ]

    def _normalize_cached_section_functional_types(self, data: Any) -> tuple[Any, bool]:
        changed = False
        if isinstance(data, list):
            normalized = []
            for item in data:
                value, item_changed = self._normalize_cached_section_functional_types(item)
                normalized.append(value)
                changed = changed or item_changed
            return normalized, changed
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                if key == "functional_type" and value == "CONTRAST":
                    normalized[key] = "EVALUATION"
                    changed = True
                    continue
                next_value, item_changed = self._normalize_cached_section_functional_types(value)
                normalized[key] = next_value
                changed = changed or item_changed
            return normalized, changed
        return data, False
    def _jsonable(self, data: Any) -> Any:
        if isinstance(data, Paper):
            return self._jsonable(data.get_skeleton())
        if isinstance(data, dict):
            return {key: self._jsonable(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._jsonable(item) for item in data]
        if isinstance(data, tuple):
            return [self._jsonable(item) for item in data]
        if isinstance(data, set):
            return [self._jsonable(item) for item in sorted(data, key=str)]
        return data

    def _is_paper_skeleton(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        return any(key in data for key in ("sections", "paragraphs", "limitation", "appendix")) or isinstance(data.get("abstract"), dict)

    def _paper_from_cached_skeleton(self, data: Any) -> Any:
        if isinstance(data, Paper) or data is None:
            return data
        if self._is_paper_skeleton(data):
            return Paper.from_skeleton(data)
        if isinstance(data, dict) and "full_content" in data:
            return self._paper_from_cached_skeleton(data.get("full_content"))
        return data

    def _hydrate_full_content_payload(self, data: Any) -> Any:
        if isinstance(data, Paper) or data is None:
            return data
        if self._is_paper_skeleton(data):
            return {"full_content": Paper.from_skeleton(data)}
        if isinstance(data, dict) and "full_content" in data:
            data["full_content"] = self._paper_from_cached_skeleton(data.get("full_content"))
        return data

    def _hydrate_citation_parser_cache(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        paper_content_map = data.get("paper_content_map")
        if not isinstance(paper_content_map, dict):
            return data
        for item in paper_content_map.values():
            if isinstance(item, dict):
                item["full_content"] = self._paper_from_cached_skeleton(item.get("full_content"))
        return data

    def _hydrate_reference_surveys_cache(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        reference_surveys = data.get("reference_surveys")
        if not isinstance(reference_surveys, list):
            return data
        for item in reference_surveys:
            if not isinstance(item, dict):
                continue
            item["full_content"] = self._hydrate_full_content_payload(item.get("full_content"))
        return data

    def _hydrate_cached_module(self, name: str, data: Any) -> Any:
        if name == "01_citation_parser":
            return self._hydrate_citation_parser_cache(data)
        if name == "02_classified_paper":
            return self._paper_from_cached_skeleton(data)
        if name == "03_get_reference_surveys":
            return self._hydrate_reference_surveys_cache(data)
        return data

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
        data, cache_normalized = self._normalize_cached_section_functional_types(data)
        if cache_normalized:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        data = self._hydrate_cached_module(name, data)
        logging.debug("loaded cached %s", path)
        return data

    def _save_module(self, name: str, data: Any):
        path = self._module_path(name)
        if path is None:
            return
        if name == "04_literature_pool":
            cache_path = path.with_name("04_literature_pool.cache.json")
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._jsonable(data), f, ensure_ascii=False, indent=2, default=str)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._output_view(name, self._jsonable(data)), f, ensure_ascii=False, indent=2, default=str)
        logging.info("saved %s", path)

    @staticmethod
    def _normalize_module_key(value: Any) -> str:
        parts = str(value).strip().split(".")
        normalized = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            normalized.append(str(int(part)) if part.isdigit() else part)
        return ".".join(normalized)

    def _module_key(self, name: str) -> str | None:
        prefix = str(name).split("_", 1)[0]
        return self._normalize_module_key(prefix) if prefix.isdigit() else None

    def _module_selected(self, name: str) -> bool:
        if self.run_modules is None:
            return True
        key = self._module_key(name)
        if key is None:
            return True
        return key in self.run_modules or any(item.startswith(f"{key}.") for item in self.run_modules)

    def _selected_paper_classification_steps(self) -> list[str] | None:
        if self.run_modules is None or "2" in self.run_modules: return
        return [step for key, step in PAPER_CLASSIFICATION_STEPS.items() if key in self.run_modules]

    async def _run_paper_content_classification(self, query: str, review_paper: Paper) -> Paper:
        steps = self._selected_paper_classification_steps()
        if not steps:
            return await self.paper_content_classification(query, review_paper)
        cached = self._load_module("02_classified_paper")
        paper = cached if isinstance(cached, Paper) else review_paper
        if cached is _MISSING:
            logging.warning("02_classified_paper cache missing; run selected 02 substeps on the input paper")
        logging.info("run 02_classified_paper substeps: %s", ", ".join(steps))
        return await self.paper_content_classification.run_steps(query, paper, steps)

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
            logging.info("rebuild %s: cached output is empty", name)
            cached = _MISSING
        if cached is not _MISSING and not (self.force and selected):
            if name == "01_citation_parser" and refresh_cached is not None:
                refreshed = refresh_cached(cached)
                if inspect.isawaitable(refreshed):
                    refreshed = await refreshed
                cached = refreshed
                self._save_module(name, cached)
            logging.debug("skip %s: cached output exists", name)
            return cached
        if not selected:
            logging.debug("skip %s: not selected by run_modules", name)
            return {}
        data = runner()
        if inspect.isawaitable(data):
            data = await data
        self._save_module(name, data)
        return data

    async def evaluate(self, query: str | list[str], review_paper: Paper, few_shot_examples: Dict[str, str] | None = None):
        # queries = query if isinstance(query, list) else [item.strip() for item in str(query or "").split(",") if item.strip()]
        queries = self._organize_queries(query)
        query_text = " ".join(queries)
        logging.info("start survey evaluation: %s", query_text)
        cached_result = self._load_module("result")
        if cached_result is not _MISSING and not self.force and self.run_modules is None:
            if not any(key in cached_result for key in ("preprocessing", "evaluations")):
                logging.info("skip full evaluation: cached compact result exists")
                return cached_result
            logging.info("ignore legacy verbose result cache and rebuild compact result")

        minimum_check = await self._run_or_load_module("00_minimum_check", lambda: self.minimum_completion(review_paper))
        result = {
            "query": query_text,
            "minimum_check": minimum_check.get("minimum_check"),
            "preprocessing": {},
            "evaluations": {},
            "aggregate_review": None,
            "errors": [],
        }
        # if minimum_check["minimum_check"]["status"] != "pass":
        #     logging.info("minimum check failed")
            # return result

        parse_task = asyncio.create_task(
            self._run_or_load_module(
                "01_citation_parser",
                lambda: self.citation_parser(review_paper.references),
                refresh_cached=lambda cached: self.citation_parser.refresh_status3(review_paper.references, cached),
            )
        )
        sentence_task = asyncio.create_task(
            self._run_or_load_module("02_classified_paper", lambda: self._run_paper_content_classification(query_text, review_paper))
        )
        reference_survey_task = asyncio.create_task(
            self._run_or_load_module("03_get_reference_surveys", lambda: self.get_reference_surveys(query_text))
        )
        citation_data, classified_paper, reference_surveys = await asyncio.gather(parse_task, sentence_task, reference_survey_task)
        paper_content_map = citation_data["paper_content_map"]
        logging.info("preprocessing complete: %d citations", len(citation_data.get("paper_content_map", {})))

        literature_pool = await self._run_or_load_module(
            "04_literature_pool",
            lambda: self.literature_pool(queries, classified_paper, paper_content_map),
        )
        logging.info("literature pool complete: %d papers", len(literature_pool.get("literature_pool", {})))

        entity_data = await self._run_or_load_module(
            "05_uncited_entities",
            lambda: self.entity_extractor(classified_paper, paper_content_map=paper_content_map),
        )
        logging.info("entity extraction complete: %d uncited entities", len(entity_data.get("uncited_entities", [])))

        claim_data = await self._run_or_load_module(
            "06_claim_segmentation",
            lambda: self.claim_segmentation(classified_paper),
        )

        fact_data = await self._run_or_load_module(
            "07_fact_check",
            lambda: self.claim_verifier(claim_data, paper_content_map, entity_data=entity_data),
        )
        logging.info("fact verification complete: %d targets", fact_data.get("checked_count", 0))

        source_data = await self._run_or_load_module(
            "08_missing_papers",
            lambda: self.source_critic(
                queries,
                classified_paper,
                paper_content_map,
                reference_surveys=reference_surveys,
                entity_data=entity_data,
                literature_pool=literature_pool,
                citation_graph=literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {},
            ),
        )
        logging.info("missing paper check complete: %d candidates", len(source_data.get("source_evals", {}).get("missing_papers", [])))

        topic_data = await self._run_or_load_module(
            "09_topic_coverage",
            lambda: self.topic_coverage(
                queries,
                classified_paper,
                reference_surveys=reference_surveys,
                literature_pool=literature_pool,
                citation_graph=literature_pool.get("citation_graph", {}) if isinstance(literature_pool, dict) else {},
            ),
        )
        logging.info("topic coverage complete")

        internal_data = await self._run_or_load_module(
            "10_internal_consistency",
            lambda: self.internal_consistency(classified_paper),
        )
        logging.info("internal consistency complete")

        contribution_data = await self._run_or_load_module(
            "11_contribution_consistency",
            lambda: self.contribution_consistency(classified_paper, queries),
        )

        # aggregate_input = {
        #     "query": query_text,
        #     "minimum_check": minimum_check["minimum_check"],
        #     "preprocessing": {
        #         "classified_paper": classified_paper,
        #     },
        #     "evaluations": {
        #         "fact_checks": fact_data.get("fact_checks", []),
        #         "source_evals": source_data.get("source_evals", {}),
        #         "topic_evals": topic_data.get("topic_evals", {}),
        #         "internal_evals": internal_data,
        #         "contribution_evals": contribution_data,
        #     },
        #     "errors": [],
        # }
        # aggregate_review = await self._run_or_load_module(
        #     "12_aggregate_review",
        #     lambda: self.final_aggregate(aggregate_input),
        # )
        # result = self._final_result_summary(
        #     query=query_text,
        #     minimum_check=minimum_check["minimum_check"],
        #     citation_data=citation_data,
        #     classified_paper=classified_paper,
        #     reference_surveys=reference_surveys,
        #     literature_pool=literature_pool,
        #     entity_data=entity_data,
        #     claim_data=claim_data,
        #     fact_data=fact_data,
        #     source_data=source_data,
        #     topic_data=topic_data,
        #     internal_data=internal_data,
        #     contribution_data=contribution_data,
        #     aggregate_review=aggregate_review,
        # )
        # self._save_module("result", result)
        logging.info("survey evaluation complete")
        return result


async def evaluate_survey(
    query: str | list[str],
    review_paper: Paper,
    config: ToolConfig | None = None,
    few_shot_examples=None,
    output_dir: str | Path | None = None,
    run_modules: str | set[str] | set[int] | None = None,
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
    review_paper: Paper,
    config: ToolConfig | None = None,
    few_shot_examples=None,
    output_dir: str | Path | None = None,
    run_modules: str | set[str] | set[int] | None = None,
    force: bool = False,
):
    config = config or ToolConfig()
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / stamp
    if run_modules in (None, ""):
        module_set = None
    elif isinstance(run_modules, set):
        module_set = {SurveyEvaluationAgent._normalize_module_key(item) for item in run_modules}
    else:
        module_set = {
            SurveyEvaluationAgent._normalize_module_key(item)
            for item in str(run_modules).split(",")
            if item.strip()
        }
    agent = SurveyEvaluationAgent(config, output_dir=output_dir, run_modules=module_set, force=force)
    return await agent.evaluate(query, review_paper, few_shot_examples=few_shot_examples)
