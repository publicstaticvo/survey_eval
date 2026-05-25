from __future__ import annotations

import re
from typing import Any

from ..utils import paragraph_to_text, paragraphs_to_text, section_text, extract_json, is_generic_heading, normalize_heading
from ..utility.evidence_check import EvidenceCheck
from ..utility.llmclient import AsyncChat
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from ..prompts import MISSING_TOPIC_CLAIM


def cosine_similarity_matrix(left, right):
    import numpy as np

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


class MissingTopicLLMClient(AsyncChat):
    PROMPT = MISSING_TOPIC_CLAIM

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        data = extract_json(response)
        if data['has_claim']:
            verified, _ = self.check.verify(data['evidence'], context["text"])
            assert verified
        return data

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(topic=inputs["topic"], text=inputs["text"]), {"text": inputs["text"]}


class TopicCoverageCritic:
    def __init__(self, config: ToolConfig):
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.sim_threshold = config.topic_sim_threshold
        self.weak_threshold = config.topic_weak_sim_threshold
        self.reason_llm = MissingTopicLLMClient(config)

    def _heading_text_for_embedding(self, title: str) -> str:
        return re.sub(r"\s+", " ", normalize_heading(title)).strip()

    def _semantic_targets(self, review_paper: dict[str, Any]) -> list[dict[str, Any]]:
        targets = []

        def walk(section: dict[str, Any]):
            section_id = str(section.get("section_id") or "")
            title = (section.get("title") or "").strip()
            if not is_generic_heading(title):
                targets.append(
                    {
                        "kind": "heading",
                        "section_id": section_id,
                        "section_title": title,
                        "text": self._heading_text_for_embedding(title),
                        "label": f"{section_id} {title}".strip(),
                    }
                )
                for paragraph in section["paragraphs"]:
                    text = paragraph_to_text(paragraph)
                    if text:
                        targets.append(
                            {
                                "kind": "paragraph",
                                "section_id": section_id,
                                "section_title": title,
                                "text": text,
                                "label": f"{section_id} {title} paragraph".strip(),
                            }
                        )
            for child in section["sections"]:
                walk(child)

        for section in review_paper["sections"]:
            walk(section)
        return targets

    def _self_claims(self, self_evidence: dict[str, Any]) -> list[dict[str, Any]]:
        claims = []
        for section_id, description in self_evidence.get("section_map", {}).items():
            claims.append({"type": "dict", "section_id": str(section_id), "text": description})
        for aspect in self_evidence.get("aspect_list", []):
            claims.append({"type": "list", "section_id": "", "text": aspect})
        return claims

    def _section_target_indexes(self, targets: list[dict[str, Any]], section_id: str) -> list[int]:
        prefix = f"{section_id}."
        return [
            idx
            for idx, target in enumerate(targets)
            if target["section_id"] == section_id or target["section_id"].startswith(prefix)
        ]

    def _best_match(self, similarities, targets: list[dict[str, Any]]) -> tuple[int, float]:
        best_idx = int(similarities.argmax().item())
        return best_idx, float(similarities.max().item())

    async def _missing_reason(self, topic_name: str, review_paper: dict[str, Any]) -> dict[str, Any]:
        blocks = []
        abstract = review_paper.get("abstract") or {}
        if isinstance(abstract, dict) and abstract.get("paragraphs"):
            blocks.append(paragraphs_to_text(abstract.get("paragraphs", [])))
        for section in review_paper.get("sections", []) or []:
            text = section_text(section, include_children=True)
            if text: blocks.append(text)
        for block in blocks:
            try:
                result = await self.reason_llm.call(inputs={"topic": topic_name, "text": block})
                if result["has_claim"]: return result
            except Exception as e:
                print(f"TopicCoverage {e}")
        return {"has_claim": False, "evidence": ""}

    async def _topic_coverage(
        self,
        topics: list[dict[str, Any]],
        review_paper: dict[str, Any],
        anchor_count: int,
        targets: list[dict[str, Any]],
        similarities,
    ) -> list[dict[str, Any]]:
        """是否覆盖了reference survey提到的topics。检查对象为文章内容。"""
        results = []
        for idx, topic in enumerate(topics):
            topic_name = topic["topic_name"]
            best_idx, best_sim = self._best_match(similarities[idx], targets)
            status = "covered" if best_sim >= self.sim_threshold else "weakly covered" if best_sim >= self.weak_threshold else "missing"
            item = {
                "topic": topic_name,
                "status": status,
                "best_match": targets[best_idx]["label"],
                "best_match_type": targets[best_idx]["kind"],
                "similarity": best_sim,
                "severity": None,
                "reason_claim": {"has_claim": False, "evidence": []},
                "sources": topic.get("sources", []),
                "source_type": topic.get("source", ""),
            }
            if status == "missing":
                item["reason_claim"] = await self._missing_reason(topic_name, review_paper)
                item["severity"] = "weakness" if self._is_consensus_topic(topic, anchor_count) else "comment"
            results.append(item)
        
        return results

    def _is_consensus_topic(self, topic: dict[str, Any], anchor_count: int) -> bool:
        if anchor_count <= 0: return False
        survey_ids = {source.get("survey_id") for source in topic.get("sources", []) if source.get("survey_id")}
        if len(survey_ids) == anchor_count: return True
        return len(survey_ids) / anchor_count > 0.5

    async def _self_consistency(
        self,
        self_claims: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        similarities,
    ) -> list[dict[str, Any]]:
        results = []
        for idx, claim in enumerate(self_claims):
            if claim["type"] == "dict":
                candidate_indexes = self._section_target_indexes(targets, claim["section_id"])
                best_sim = max(float(similarities[idx][j]) for j in candidate_indexes)
                if best_sim < self.sim_threshold:
                    results.append(
                        {
                            "type": "self_inconsistent",
                            "declared_key": claim["section_id"],
                            "declared_topic": claim["text"],
                            "severity": "weakness",
                            "similarity": best_sim,
                        }
                    )
            else:
                best_idx, best_sim = self._best_match(similarities[idx], targets)
                if best_sim < self.sim_threshold:
                    results.append(
                        {
                            "type": "self_inconsistent",
                            "declared_topic": claim["text"],
                            "severity": "weakness",
                            "similarity": best_sim,
                            "best_match": targets[best_idx]["label"],
                            "best_match_type": targets[best_idx]["kind"],
                        }
                    )
        return results

    async def __call__(self, topics: dict[str, Any], review_paper: dict[str, Any]) -> dict[str, Any]:
        """
            topics = {
                "reference_data": {
                    "reference_papers": 被reference_surveys引用的文献list，含详细信息及reference_surveys引用计数, 
                    "reference_surveys": reference_surveys list
                },
                "reference_topics": 从reference_surveys中总结出来的topics，格式为[{"topic": "topic名称", "sources": [来源]}],
                "self_topics": {
                    "section_map": 指定了具体章节的scope声明 <dict>, 
                    "aspect_list": 未指定具体章节的scope声明<list>
                },
            }
        """
        reference_topics = topics["reference_topics"]
        self_topics = topics["self_topics"]
        targets = self._semantic_targets(review_paper)
        self_claims = self._self_claims(self_topics)
        query_texts = [topic["topic_name"] for topic in reference_topics] + [claim["text"] for claim in self_claims]
        embeddings = self.sbert.embed(query_texts + [target["text"] for target in targets])
        query_emb = embeddings[: len(query_texts)]
        target_emb = embeddings[len(query_texts) :]
        sim = cosine_similarity_matrix(query_emb, target_emb)
        anchor_count = len(topics["reference_data"].get("reference_surveys") or {})
        topic_results = await self._topic_coverage(
            reference_topics,
            review_paper,
            anchor_count,
            targets,
            sim[: len(reference_topics)],
        )
        self_consistency = await self._self_consistency(
            self_claims,
            targets,
            sim[len(reference_topics) :],
        )
        covered_topics = [item["topic"] for item in topic_results if item["status"] != "missing"]
        return {
            "topic_evals": {
                "topic_coverage": topic_results,
                "self_consistency": self_consistency,
                "covered_topics": covered_topics,
                "missing_topics": [item for item in topic_results if item["status"] == "missing"],
            }
        }
