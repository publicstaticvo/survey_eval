from __future__ import annotations

from typing import Any

from ..utility.tool_config import ToolConfig
from .content_parser import ContentParser
from .contribution_classify import ContributionClassification
from .find_all_entities import FindAllEntities
from .section_classify import SectionClassification
from .sentences import SentenceClassification


class PaperContentClassification:
    """Run paper content classifiers in the order required by downstream checks."""

    def __init__(self, config: ToolConfig):
        self.sentence_classification = SentenceClassification(config)
        self.section_classification = SectionClassification(config)
        self.contribution_classification = ContributionClassification(config)
        self.content_parser = ContentParser(config)
        self.find_all_entities = FindAllEntities(config)

    async def __call__(self, query: str, paper_content: dict[str, Any]) -> dict[str, Any]:
        paper = await self.sentence_classification(paper_content)
        paper = await self.section_classification(paper)
        paper = await self.content_parser(paper)
        paper, contribution_claims = await self.contribution_classification(paper)
        paper["contribution_claims"] = contribution_claims
        paper = await self.find_all_entities(query, paper)
        return paper


