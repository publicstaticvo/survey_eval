import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

if __package__:
    from .tools.anchor_surveys import AnchorSurveyFetch
    from .tools.argument_eval import ArgumentStructureEvaluator
    from .tools.citation_parser import CitationParser
    from .tools.claim_segmentation import ClaimSegmentation
    from .tools.dynamic_oracle_generator import DynamicOracleGenerator
    from .tools.fact_check import FactualCorrectnessCritic
    from .tools.golden_topics import GoldenTopicGenerator
    from .tools.minimum_completion import MinimalCompletionCheck
    from .tools.programmatic_quality import QualityCritic
    from .tools.query_expand import QueryExpand
    from .tools.request_utils import SessionManager
    from .tools.source_critic import MissingPaperCheck
    from .tools.structure_eval import StructureCheck
    from .tools.tool_config import ToolConfig
    from .tools.topic_coverage import TopicCoverageCritic
else:
    from tools.anchor_surveys import AnchorSurveyFetch
    from tools.argument_eval import ArgumentStructureEvaluator
    from tools.citation_parser import CitationParser
    from tools.claim_segmentation import ClaimSegmentation
    from tools.dynamic_oracle_generator import DynamicOracleGenerator
    from tools.fact_check import FactualCorrectnessCritic
    from tools.golden_topics import GoldenTopicGenerator
    from tools.minimum_completion import MinimalCompletionCheck
    from tools.programmatic_quality import QualityCritic
    from tools.query_expand import QueryExpand
    from tools.request_utils import SessionManager
    from tools.source_critic import MissingPaperCheck
    from tools.structure_eval import StructureCheck
    from tools.tool_config import ToolConfig
    from tools.topic_coverage import TopicCoverageCritic


@dataclass
class SurveyEvaluationAgent:
    config: ToolConfig

    def __post_init__(self):
        self._normalize_paths()
        self.minimum_completion = MinimalCompletionCheck(self.config)
        self.query_expand = QueryExpand(self.config)
        self.anchor_surveys = AnchorSurveyFetch(self.config)
        self.golden_topics = GoldenTopicGenerator(self.config)
        self.dynamic_oracle = DynamicOracleGenerator(self.config)
        self.citation_parser = CitationParser(self.config)
        self.claim_segmentation = ClaimSegmentation(self.config)
        self.fact_check = FactualCorrectnessCritic(self.config)
        self.source_critic = MissingPaperCheck(self.config)
        self.topic_coverage = TopicCoverageCritic(self.config)
        self.structure_eval = StructureCheck(self.config)
        self.argument_eval = ArgumentStructureEvaluator(self.config)
        self.quality_eval = QualityCritic(self.config)

    def _normalize_paths(self):
        letor_path = Path(self.config.letor_path)
        if not letor_path.is_absolute():
            candidate = Path(__file__).resolve().parent / letor_path
            if candidate.exists():
                object.__setattr__(self.config, "letor_path", str(candidate))

    async def _fact_check_claims(self, claims, paper_content_map):
        tasks = []
        for claim in claims:
            citation_key = claim["citation_key"]
            cited_paper = paper_content_map.get(citation_key)
            if not cited_paper or cited_paper.get("status", 3) >= 3:
                continue
            tasks.append(asyncio.create_task(self.fact_check(claim["claim_text"], cited_paper)))
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
            except Exception as exc:
                results.append(
                    {
                        "fact_check": {
                            "claim": "",
                            "judgment": "NEUTRAL",
                            "evidence": "",
                            "reason": str(exc),
                            "score": 0.0,
                            "material": "error",
                        }
                    }
                )
                continue
            results.append(result)
        return results

    async def evaluate(self, query: str, review_paper: Dict[str, Any], few_shot_examples: Dict[str, str] | None = None):
        minimum_check = await self.minimum_completion(review_paper)
        result = {
            "query": query,
            "minimum_check": minimum_check["minimum_check"],
            "preprocessing": {},
            "evaluations": {},
            "aggregate_review": None,
            "errors": [],
        }
        if minimum_check["minimum_check"]["status"] != "pass":
            return result

        query_data = await self.query_expand(query)
        anchor_data = await self.anchor_surveys(query_data["core"], query)
        golden_topic_task = asyncio.create_task(self.golden_topics(query, anchor_data.get("anchor_surveys", {}), query_data["library"]))
        oracle_task = asyncio.create_task(self.dynamic_oracle(query, query_data["queries"], query_data["library"]))

        parse_task = asyncio.create_task(self.citation_parser(review_paper.get("citations", {})))
        claim_task = asyncio.create_task(self.claim_segmentation(review_paper))
        quality_task = asyncio.create_task(self.quality_eval._run(review_paper))
        golden_topic_data, oracle_data, citation_data, claim_data, quality_data = await asyncio.gather(
            golden_topic_task,
            oracle_task,
            parse_task,
            claim_task,
            quality_task,
            return_exceptions=False,
        )

        paper_content_map = citation_data["paper_content_map"]
        claims = claim_data["claims"]
        fact_checks = await self._fact_check_claims(claims, paper_content_map)

        topic_data = await self.topic_coverage(golden_topic_data.get("golden_topics", []), review_paper)
        missing_topics = [
            item["topic"]
            for item in topic_data["topic_evals"].get("topic_coverage", [])
            if item["status"] == "missing"
        ]
        source_data = self.source_critic(
            paper_content_map,
            oracle_data,
            anchor_data.get("anchor_papers", {}),
            golden_topic_data.get("golden_topics", []),
        )
        structure_data = await self.structure_eval(
            review_paper,
            anchor_data.get("anchor_papers", {}),
            missing_topics,
        )
        argument_data = await self.argument_eval(
            review_paper,
            minimum_check_details=minimum_check["minimum_check"].get("details", {}),
            few_shot_examples=few_shot_examples,
        )

        result["preprocessing"] = {
            "query_expand": query_data,
            "anchor_surveys": anchor_data,
            "golden_topics": golden_topic_data,
            "oracle_data": oracle_data,
            "paper_content_map": paper_content_map,
            "claims": claims,
        }
        result["evaluations"] = {
            "fact_checks": [item["fact_check"] for item in fact_checks],
            "source_evals": source_data["source_evals"],
            "topic_evals": topic_data["topic_evals"],
            "structure_evals": structure_data["structure_evals"],
            "argument_evals": argument_data["argument_evals"],
            "quality_evals": quality_data["quality_evals"],
        }
        result["errors"].extend(claim_data.get("errors", []))
        return result


async def evaluate_survey(query: str, review_paper: Dict[str, Any], config: ToolConfig | None = None, few_shot_examples=None):
    config = config or ToolConfig()
    await SessionManager.init()
    try:
        agent = SurveyEvaluationAgent(config)
        return await agent.evaluate(query, review_paper, few_shot_examples=few_shot_examples)
    finally:
        await SessionManager.close()
