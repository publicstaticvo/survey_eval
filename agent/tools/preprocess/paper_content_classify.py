from __future__ import annotations

import asyncio
from typing import Any

from ..utility.paper_elements import Paper
from ..utility.tool_config import ToolConfig
from .content_parser_paragraph import ParagraphContentParser
from .section_classify_hierarchical import HierarchicalSectionClassification
from .sentences import SentenceClassification
from .multilabel_extract import MultiLabelExtraction


# Active entry points used by agent.py.
ACTIVE_ENTRY_POINTS = {
    "2.1": "multilabel_extract.py:MultiLabelExtraction",
    "2.2": "section_classify_hierarchical.py:HierarchicalSectionClassification",
    "2.3": {
        "topics": "content_parser_paragraph.py:ParagraphContentParser",
        "entities": "content_parser_paragraph.py:ParagraphContentParser (objects)",
    },
}
class PaperContentClassification:
    """Run paper content classifiers and expose per-module error accounting."""

    def __init__(self, config: ToolConfig):
        self.sentence_classification = SentenceClassification(config)
        self.multilabel_extraction = MultiLabelExtraction(config)
        self.section_classification = HierarchicalSectionClassification(config)
        self.content_parser = ParagraphContentParser(config)
        self.last_report: dict[str, Any] = {
            "module": "paper_content_classification",
            "success_count": 0,
            "error_count": 0,
            "modules": {},
            "errors": [],
        }

    def _add_module_report(self, name: str, detector: Any, error: Exception | None = None) -> None:
        if error is not None:
            report = {"module": name, "success_count": 0, "error_count": 1, "errors": [{"error": repr(error)}]}
        else:
            report = getattr(detector, "last_report", None) or {
                "module": name,
                "success_count": 0,
                "error_count": 0,
                "errors": [],
            }
        self.last_report["modules"][name] = report

    async def _run_single(
        self,
        name: str,
        detector: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            result = await detector(*args, **kwargs)
        except Exception as exc:
            self._add_module_report(name, detector, exc)
            return exc
        self._add_module_report(name, detector)
        return result

    def _finalize_report(self) -> None:
        reports = list(self.last_report["modules"].values())
        self.last_report["success_count"] = sum(int(report.get("success_count", 0)) for report in reports)
        self.last_report["error_count"] = sum(int(report.get("error_count", 0)) for report in reports)
        self.last_report["errors"] = [
            {"module": name, **error}
            for name, report in self.last_report["modules"].items()
            for error in report.get("errors", [])
        ]

    @staticmethod
    def _merge_unique(existing: list[Any], additions: list[Any]) -> list[Any]:
        merged = list(existing)
        for item in additions:
            if item not in merged:
                merged.append(item)
        return merged

    @classmethod
    def _merge_claim_maps(
        cls,
        existing: dict[str, list[Any]],
        additions: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        merged = {key: list(value) for key, value in existing.items()}
        for section_key, claims in additions.items():
            merged[section_key] = cls._merge_unique(merged.get(section_key, []), claims)
        return merged

    async def run_steps(
        self,
        query: str,
        paper: Paper,
        steps: list[str],
        only_missing: bool = False,
    ) -> Paper:
        """Run the active 2.1--2.3 preprocessing graph concurrently.

        2.1 is the six-function paragraph extractor, 2.2 is hierarchical
        section classification, and 2.3 extracts topics and introduced objects
        with ContentParser. Downstream detectors consume these verified outputs.
        """
        self.last_report = {
            "module": "paper_content_classification",
            "success_count": 0,
            "error_count": 0,
            "modules": {},
            "errors": [],
        }
        requested = set(steps)
        jobs: list[tuple[str, Any]] = []

        def add(name: str, detector: Any):
            if name not in {job_name for job_name, _ in jobs}:
                jobs.append((name, detector))

        # 2.1: six label extraction; 2.2: hierarchical section labels.
        if "multilabel" in requested:
            add("2.1.multilabel", self.multilabel_extraction)
        if "section" in requested:
            add("2.2.section", self.section_classification)

        # 2.3: ContentParser covers all paragraphs and materializes verified objects.
        # Keep old names as aliases for cached or legacy invocations.
        if requested.intersection({"content", "structure_topics_entities", "entities"}):
            add("2.3.content", self.content_parser)

        async def execute(name: str, detector: Any):
            output = await self._run_single(name, detector, paper, only_missing=only_missing)
            return name, detector, output

        results = await asyncio.gather(
            *(execute(name, detector) for name, detector in jobs),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self.last_report["errors"].append({"module": "preprocess", "error": repr(result)})
                continue
            name, detector, output = result
            if isinstance(output, Exception):
                continue
            if isinstance(output, tuple):
                # Preserve the legacy contribution/textual return contract.
                if name.endswith("contribution"):
                    paper, claims = output
                    paper.contribution_claims = self._merge_claim_maps({}, claims) if not only_missing else self._merge_claim_maps(paper.contribution_claims, claims)
                elif name.endswith("textual"):
                    paper, claims = output
                    paper.contribution_claims = self._merge_claim_maps(paper.contribution_claims, claims)
            elif isinstance(output, Paper):
                paper = output

        self._finalize_report()
        return paper

    async def __call__(self, query: str, paper_content: Paper) -> Paper:
        """Run the active 2.1, 2.2, and 2.3 preprocessing entry points."""
        return await self.run_steps(
            query,
            paper_content,
            ["multilabel", "section", "content"],
        )
    def get_last_report(self) -> dict[str, Any]:
        return self.last_report