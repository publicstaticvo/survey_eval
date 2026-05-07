from __future__ import annotations

import json
import re
from typing import Any

from .topic_utils import paragraphs_to_text, section_text
from .utility.evidence_check import EvidenceCheck
from .utility.llmclient import AsyncChat
from .utility.sbert_client import SentenceTransformerClient
from .utility.tool_config import ToolConfig


def cosine_similarity_matrix(left, right):
    import numpy as np

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


MISSING_TOPIC_REASON_PROMPT = """You are checking whether a survey explicitly explains why one topic is not covered.

Topic:
{topic}

Text:
{text}

If the text explicitly states a limitation, scope restriction, omission, or future-work note related to this topic, extract verbatim evidence.
If not, return empty evidence.

Return JSON only:
```json
{{
  "has_claim": true,
  "evidence": ["verbatim sentence"]
}}
```
"""


class MissingTopicLLMClient(AsyncChat):
    PROMPT = MISSING_TOPIC_REASON_PROMPT

    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        try:
            data = json.loads(response)
        except Exception:
            match = re.search(r"\{.*\}", response or "", re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        evidence = data.get("evidence")
        evidence = evidence if isinstance(evidence, list) else ([evidence] if evidence else [])
        verified, confidence = self.check.verify(evidence, context["text"]) if evidence else (False, 0.0)
        return {"has_claim": verified, "evidence": evidence if verified else [], "confidence": confidence}

    def _organize_inputs(self, inputs):
        return self.PROMPT.format(topic=inputs["topic"], text=inputs["text"]), {"text": inputs["text"]}


class TopicCoverageCritic:
    def __init__(self, config: ToolConfig):
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.sim_threshold = config.topic_sim_threshold
        self.weak_threshold = config.topic_weak_sim_threshold
        self.reason_llm = MissingTopicLLMClient(config)

    def _topic_names(self, topic_bundle: dict[str, Any]) -> list[dict[str, Any]]:
        topics = []
        for topic in topic_bundle.get("golden_topics", []) or []:
            name = (topic.get("topic_name") or topic.get("topic") or "").strip()
            if name:
                topics.append({**topic, "topic_name": name})
        return topics

    def _section_documents(self, review_paper: dict[str, Any]) -> list[dict[str, str]]:
        docs = []
        for section in review_paper.get("sections", []) or []:
            docs.append(
                {
                    "section_id": str(section.get("section_id") or ""),
                    "title": (section.get("title") or "").strip(),
                    "text": section_text(section, include_children=True),
                }
            )
        return [doc for doc in docs if doc["text"]]

    async def _missing_reason(self, topic_name: str, review_paper: dict[str, Any]) -> dict[str, Any]:
        blocks = []
        abstract = review_paper.get("abstract") or {}
        if isinstance(abstract, dict) and abstract.get("paragraphs"):
            blocks.append(paragraphs_to_text(abstract.get("paragraphs", [])))
        for section in review_paper.get("sections", []) or []:
            text = section_text(section, include_children=True)
            if text:
                blocks.append(text)
        for block in blocks:
            try:
                result = await self.reason_llm.call(inputs={"topic": topic_name, "text": block})
            except Exception:
                result = {"has_claim": False, "evidence": []}
            if result.get("has_claim"):
                return result
        return {"has_claim": False, "evidence": [], "confidence": 0.0}

    async def _topic_coverage(self, topics: list[dict[str, Any]], review_paper: dict[str, Any], anchor_count: int) -> list[dict[str, Any]]:
        docs = self._section_documents(review_paper)
        if not topics:
            return []
        if not docs:
            results = []
            for topic in topics:
                results.append(
                    {
                        "topic": topic["topic_name"],
                        "status": "missing",
                        "best_match": None,
                        "similarity": 0.0,
                        "severity": "weakness" if self._is_consensus_topic(topic, anchor_count) else "comment",
                        "reason_claim": {"has_claim": False, "evidence": []},
                    }
                )
            return results

        topic_names = [topic["topic_name"] for topic in topics]
        doc_titles = [f'{doc["section_id"]} {doc["title"]}'.strip() for doc in docs]
        embeddings = self.sbert.embed(topic_names + doc_titles)
        topic_emb = embeddings[: len(topic_names)]
        doc_emb = embeddings[len(topic_names) :]
        sim = cosine_similarity_matrix(topic_emb, doc_emb)

        results = []
        for idx, topic in enumerate(topics):
            best_idx = int(sim[idx].argmax().item())
            best_sim = float(sim[idx].max().item())
            status = "covered" if best_sim >= self.sim_threshold else "weakly covered" if best_sim >= self.weak_threshold else "missing"
            item = {
                "topic": topic["topic_name"],
                "status": status,
                "best_match": doc_titles[best_idx],
                "similarity": best_sim,
                "severity": None,
                "reason_claim": {"has_claim": False, "evidence": []},
                "sources": topic.get("sources", []),
                "source_type": topic.get("source", ""),
            }
            if status == "missing":
                item["reason_claim"] = await self._missing_reason(topic["topic_name"], review_paper)
                item["severity"] = "weakness" if self._is_consensus_topic(topic, anchor_count) else "comment"
            results.append(item)
        return results

    def _is_consensus_topic(self, topic: dict[str, Any], anchor_count: int) -> bool:
        if anchor_count <= 0:
            return False
        survey_ids = {source.get("survey_id") for source in topic.get("sources", []) if source.get("survey_id")}
        if len(survey_ids) == anchor_count:
            return True
        return len(survey_ids) / anchor_count >= 0.75

    async def _self_consistency(self, self_evidence: dict[str, Any], review_paper: dict[str, Any]) -> list[dict[str, Any]]:
        docs = self._section_documents(review_paper)
        if not docs:
            return []
        doc_names = [f'{doc["section_id"]} {doc["title"]}'.strip() for doc in docs]
        doc_embeddings = self.sbert.embed(doc_names)
        results = []
        if self_evidence.get("type") == "dict":
            section_map = self_evidence.get("section_map") or {}
            texts = list(section_map.values())
            if texts:
                embeddings = self.sbert.embed(texts)
                sim = cosine_similarity_matrix(embeddings, doc_embeddings)
                for idx, (section_id, description) in enumerate(section_map.items()):
                    candidates = [j for j, doc in enumerate(docs) if doc["section_id"] == section_id]
                    best_sim = max((float(sim[idx][j]) for j in candidates), default=0.0)
                    if best_sim < self.sim_threshold:
                        results.append(
                            {
                                "type": "self_inconsistent",
                                "declared_key": section_id,
                                "declared_topic": description,
                                "severity": "weakness",
                                "similarity": best_sim,
                            }
                        )
        elif self_evidence.get("type") == "list":
            aspects = self_evidence.get("aspect_list") or []
            if aspects:
                embeddings = self.sbert.embed(aspects)
                sim = cosine_similarity_matrix(embeddings, doc_embeddings)
                for idx, aspect in enumerate(aspects):
                    best_sim = float(sim[idx].max().item())
                    if best_sim < self.sim_threshold:
                        results.append(
                            {
                                "type": "self_inconsistent",
                                "declared_topic": aspect,
                                "severity": "weakness",
                                "similarity": best_sim,
                            }
                        )
        return results

    async def __call__(self, topic_bundle: dict[str, Any], review_paper: dict[str, Any]) -> dict[str, Any]:
        topics = self._topic_names(topic_bundle)
        anchor_count = int(topic_bundle.get("metadata", {}).get("anchor_surveys_count", 0) or 0)
        topic_results = await self._topic_coverage(topics, review_paper, anchor_count)
        self_consistency = await self._self_consistency(topic_bundle.get("self_evidence", {}) or {}, review_paper)
        covered_topics = [item["topic"] for item in topic_results if item["status"] != "missing"]
        return {
            "topic_evals": {
                "topic_coverage": topic_results,
                "self_consistency": self_consistency,
                "covered_topics": covered_topics,
                "missing_topics": [item for item in topic_results if item["status"] == "missing"],
            }
        }
