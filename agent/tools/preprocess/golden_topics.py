from __future__ import annotations

import asyncio
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from ..prompts import TOPIC_LABEL_FROM_CLUSTER_PROMPT
from ..utility.llmclient import AsyncChat
from ..utility.sbert_client import SentenceTransformerClient
from ..utility.tool_config import ToolConfig
from .utils import cosine_similarity_matrix, extract_json, normalize_text


GENERIC_SECTION_KEYWORDS = [
    "introduction",
    "background",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "evaluation",
    "result",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "challenge",
    "challenges",
    "outlook",
    "prospect",
    "prospects",
    "frontier",
    "frontiers",
    "unsolved",
    "future work",
    "future works",
    "future direction",
    "future directions",
    "limitation",
    "limitations",
    "open problem",
    "open problems",
    "open question",
    "open questions",
]


@dataclass
class TopicRecord:
    topic_name: str
    source: str
    representative_papers: List[str]
    evidence_titles: List[str]
    metadata: Dict[str, Any]


class TopicLabelLLMClient(AsyncChat):
    PROMPT = TOPIC_LABEL_FROM_CLUSTER_PROMPT

    def _availability(self, response, context):
        data = extract_json(response)
        topic_name = (data.get("topic_name") or "").strip()
        if not topic_name:
            topic_name = context["fallback"]
        return {"topic_name": topic_name, "reason": data.get("reason", "")}

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(
            query=inputs["query"],
            keywords=", ".join(inputs["keywords"]),
            titles="\n".join(f"- {title}" for title in inputs["titles"]),
        )
        return prompt, {"fallback": inputs["fallback"]}


class GoldenTopicGenerator:
    """Build auditable survey topics from anchor-survey headings and oracle-paper clusters."""

    def __init__(self, config: ToolConfig):
        """Initialize topic generation utilities and thresholds."""
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.topic_label_llm = TopicLabelLLMClient(config.llm_server_info, config.sampling_params)
        self.anchor_merge_threshold = 0.85
        self.base_overlap_threshold = 0.5
        self.bertopic_min_topic_size = 10
        self.library_keep_ratio = 0.3

    def _normalize_title_for_keyword_match(self, title: str) -> str:
        """Normalize a heading for keyword-based generic-title filtering."""
        lowered = (title or "").strip().lower()
        lowered = re.sub(r"^[0-9.ivx]+\s*[.)-]?\s*", "", lowered)
        lowered = lowered.replace("&", " and ").replace("/", " ").replace("-", " ")
        return " ".join(lowered.split())

    def _is_generic_anchor_title(self, title_path: str) -> bool:
        """Return whether the current heading is a generic survey-structure title."""
        leaf_title = (title_path or "").split(">")[-1].strip()
        normalized_leaf = self._normalize_title_for_keyword_match(leaf_title)
        if not normalized_leaf:
            return True
        return any(keyword in normalized_leaf for keyword in GENERIC_SECTION_KEYWORDS)

    def _flatten_anchor_titles(self, anchor_surveys: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten anchor-survey title paths while excluding generic section headings."""
        records = []
        seen = set()
        raw_title_count = 0
        generic_filtered = 0
        duplicate_filtered = 0
        kept_examples = []
        filtered_examples = []
        for survey_title, survey_info in anchor_surveys.items():
            paper_title = survey_info.get("paper", {}).get("title", survey_title)
            for title_path in survey_info.get("titles", []):
                raw_title_count += 1
                if self._is_generic_anchor_title(title_path):
                    generic_filtered += 1
                    if len(filtered_examples) < 3:
                        filtered_examples.append(title_path)
                    continue
                normalized = normalize_text(title_path)
                if not normalized or normalized in seen:
                    duplicate_filtered += 1
                    continue
                seen.add(normalized)
                if len(kept_examples) < 3:
                    kept_examples.append(title_path)
                records.append(
                    {
                        "title_path": title_path,
                        "survey_title": survey_title,
                        "paper_title": paper_title,
                    }
                )
        print(
            "GoldenTopicGenerator::Anchor titles "
            f"raw={raw_title_count} kept={len(records)} "
            f"generic_filtered={generic_filtered} duplicate_filtered={duplicate_filtered}"
        )
        if kept_examples:
            print(f"GoldenTopicGenerator::Anchor kept examples: {kept_examples}")
        elif filtered_examples:
            print(f"GoldenTopicGenerator::Anchor filtered examples: {filtered_examples}")
        return records

    def _merge_similar_anchor_titles(self, title_records: List[Dict[str, Any]]) -> List[TopicRecord]:
        """Merge semantically similar anchor titles into auditable base topics."""
        if not title_records:
            return []

        title_texts = [record["title_path"] for record in title_records]
        embeddings = self.sbert.embed(title_texts)
        similarity = cosine_similarity_matrix(embeddings, embeddings)
        parent = list(range(len(title_records)))

        def _find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def _union(left: int, right: int):
            left_root = _find(left)
            right_root = _find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left in range(len(title_records)):
            for right in range(left + 1, len(title_records)):
                if float(similarity[left, right]) >= self.anchor_merge_threshold:
                    _union(left, right)

        groups = defaultdict(list)
        for index, record in enumerate(title_records):
            groups[_find(index)].append(record)

        topics = []
        for group_records in groups.values():
            sorted_titles = sorted(
                {record["title_path"] for record in group_records},
                key=lambda value: (value.count(">"), len(value), value.lower()),
            )
            representative_papers = sorted({record["paper_title"] for record in group_records})
            topics.append(
                TopicRecord(
                    topic_name=sorted_titles[0],
                    source="anchor-derived",
                    representative_papers=representative_papers,
                    evidence_titles=sorted_titles,
                    metadata={
                        "anchor_surveys": sorted({record["survey_title"] for record in group_records}),
                        "merge_size": len(group_records),
                    },
                )
            )
        topics.sort(
            key=lambda item: (len(item.evidence_titles), len(item.representative_papers), item.topic_name),
            reverse=True,
        )
        return topics

    def _filter_library_by_query_similarity(self, query: str, library: Dict[str, Dict[str, Any]]):
        """Keep the most query-relevant library papers and log all similarity scores."""
        if not library:
            return {}, []

        paper_items = list(library.items())
        documents = []
        for _, paper in paper_items:
            title = (paper.get("title") or "").strip()
            abstract = (paper.get("abstract") or "").strip()
            documents.append(f"{title}. {abstract}".strip())

        embeddings = self.sbert.embed([query, *documents])
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0].tolist()

        scored_papers = []
        for (paper_id, paper), score in zip(paper_items, scores):
            paper["query_similarity"] = float(score)
            scored_papers.append((paper_id, paper, float(score)))

        scored_papers.sort(key=lambda item: item[2], reverse=True)
        keep_num = max(1, math.ceil(len(scored_papers) * self.library_keep_ratio))
        kept = {paper_id: paper for paper_id, paper, _ in scored_papers[:keep_num]}

        print(f"GoldenTopicGenerator::Keep {keep_num}/{len(scored_papers)} papers for clustering")
        print(
            "GoldenTopicGenerator::Top kept papers: "
            f"{[(paper.get('title', ''), round(score, 4)) for _, paper, score in scored_papers[:5]]}"
        )

        metadata = [
            {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "query_similarity": score,
                "kept_for_topic_modeling": paper_id in kept,
            }
            for paper_id, paper, score in scored_papers
        ]
        return kept, metadata

    def _build_seed_topic_list(self, base_topics: List[TopicRecord]) -> List[List[str]]:
        """Convert base topics into BERTopic seed tokens for weak-anchor mode."""
        seed_topic_list = []
        for topic in base_topics:
            tokens = [
                token
                for token in topic.topic_name.replace(">", " ").replace("/", " ").split()
                if token
            ]
            if tokens:
                seed_topic_list.append(tokens[:8])
        return seed_topic_list

    def _run_bertopic(self, documents: List[str], seed_topic_list: List[List[str]] | None):
        """Run BERTopic and keep each cluster as a keyword list before LLM naming."""
        if not documents:
            return [], "bertopic_unavailable"

        embeddings = self.sbert.embed(documents)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.bertopic_min_topic_size,
            metric="euclidean",
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            seed_topic_list=seed_topic_list or None,
            calculate_probabilities=False,
            verbose=False,
        )
        topic_ids, _ = topic_model.fit_transform(documents, embeddings=embeddings)
        noise_count = sum(1 for topic_id in topic_ids if topic_id == -1)

        topic_groups = defaultdict(list)
        for index, topic_id in enumerate(topic_ids):
            if topic_id == -1:
                continue
            topic_groups[int(topic_id)].append(index)

        clustered_topics = []
        for topic_id, indices in topic_groups.items():
            keyword_weights = topic_model.get_topic(topic_id) or []
            if not keyword_weights:
                continue
            clustered_topics.append(
                {
                    "cluster_id": topic_id,
                    "keywords": [word for word, _ in keyword_weights[:8]],
                    "indices": indices,
                }
            )
        cluster_sizes = sorted((len(item["indices"]) for item in clustered_topics), reverse=True)
        print(
            "GoldenTopicGenerator::BERTopic "
            f"documents={len(documents)} noise={noise_count} "
            f"clusters={len(clustered_topics)} cluster_sizes={cluster_sizes}"
        )
        return clustered_topics, "ok"

    async def _name_cluster_topics(self, query: str, clustered_topics: List[Dict[str, Any]], paper_titles: List[str]):
        """Name each BERTopic cluster asynchronously with one LLM call per cluster."""

        async def _single_cluster(cluster: Dict[str, Any]):
            representative_titles = [
                paper_titles[index]
                for index in cluster["indices"][:8]
                if index < len(paper_titles)
            ]
            fallback = ", ".join(cluster["keywords"][:3]) if cluster["keywords"] else f"cluster_{cluster['cluster_id']}"
            try:
                label_info = await self.topic_label_llm.call(
                    inputs={
                        "query": query,
                        "keywords": cluster["keywords"],
                        "titles": representative_titles,
                        "fallback": fallback,
                    }
                )
            except Exception as exc:
                print(f"TopicLabelLLM {cluster['cluster_id']} {exc}")
                label_info = {"topic_name": fallback, "reason": "fallback"}
            return cluster, representative_titles, label_info

        tasks = [asyncio.create_task(_single_cluster(cluster)) for cluster in clustered_topics]
        named_clusters = []
        for task in asyncio.as_completed(tasks):
            cluster, representative_titles, label_info = await task
            named_clusters.append(
                {
                    **cluster,
                    "topic_name": label_info["topic_name"],
                    "reason": label_info.get("reason", ""),
                    "representative_titles": representative_titles,
                }
            )
        named_clusters.sort(key=lambda item: item["cluster_id"])
        return named_clusters

    def _similarity_to_any(self, topic_name: str, existing_names: List[str]) -> float:
        """Compute the maximum SBERT similarity between one topic and existing topics."""
        if not existing_names:
            return 0.0
        texts = [topic_name, *existing_names]
        embeddings = self.sbert.embed(texts)
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0]
        return float(scores.max())

    def _collect_supplementary_topics(self, base_topics: List[TopicRecord], cluster_topics: List[TopicRecord]) -> List[TopicRecord]:
        """Collect non-overlapping cluster topics as supplementary topics without gap filling."""
        base_names = [topic.topic_name for topic in base_topics]
        supplementary_topics = []
        for topic in cluster_topics:
            if self._similarity_to_any(topic.topic_name, base_names) < self.base_overlap_threshold:
                supplementary_topics.append(topic)
        return supplementary_topics

    def _gap_fill_topics(self, base_topics: List[TopicRecord], cluster_topics: List[TopicRecord]) -> List[TopicRecord]:
        """Add dissimilar cluster topics into base topics only in the 1-2 anchor-survey case."""
        accepted = []
        current_names = [topic.topic_name for topic in base_topics]
        remaining = sorted(
            cluster_topics,
            key=lambda item: item.metadata.get("cluster_size", 0),
            reverse=True,
        )

        changed = True
        while changed:
            changed = False
            next_round = []
            for topic in remaining:
                if self._similarity_to_any(topic.topic_name, current_names) < self.base_overlap_threshold:
                    accepted.append(topic)
                    current_names.append(topic.topic_name)
                    changed = True
                else:
                    next_round.append(topic)
            remaining = next_round
        return accepted

    def _serialize_topics(self, topics: List[TopicRecord]) -> List[Dict[str, Any]]:
        """Convert topic records into plain dictionaries for downstream modules."""
        return [
            {
                "topic_name": topic.topic_name,
                "source": topic.source,
                "representative_papers": topic.representative_papers,
                "evidence_titles": topic.evidence_titles,
                "metadata": topic.metadata,
            }
            for topic in topics
        ]

    async def __call__(self, query: str, anchor_surveys: Dict[str, Any], library: Dict[str, Dict[str, Any]]):
        """Build golden topics from anchor-survey headings and BERTopic-derived clusters."""
        anchor_surveys_count = len(anchor_surveys)
        print(
            "GoldenTopicGenerator::Input "
            f"anchor_surveys={anchor_surveys_count} library={len(library)} "
            f"keep_ratio={self.library_keep_ratio}"
        )
        if anchor_surveys_count:
            print(f"GoldenTopicGenerator::Anchor survey titles: {list(anchor_surveys)[:3]}")

        anchor_title_records = self._flatten_anchor_titles(anchor_surveys)
        base_topics = self._merge_similar_anchor_titles(anchor_title_records)
        print(f"GoldenTopicGenerator::Base topics count={len(base_topics)}")

        filtered_library, library_similarity = self._filter_library_by_query_similarity(query, library)
        filtered_papers = list(filtered_library.values())
        documents, paper_titles = [], []
        for paper in filtered_papers:
            title = (paper.get("title") or "").strip()
            if not title: continue
            abstract = (paper.get("abstract") or "").strip()
            documents.append(f"{title}. {abstract}".strip())
            paper_titles.append(title)
        print(f"GoldenTopicGenerator::Documents prepared={len(documents)}")

        seed_topic_list = self._build_seed_topic_list(base_topics) if anchor_surveys_count in {1, 2} else None
        if seed_topic_list is not None:
            print(f"GoldenTopicGenerator::Seed topics count={len(seed_topic_list)}")
        clustered_topics, bertopic_status = self._run_bertopic(documents, seed_topic_list)
        named_clusters = await self._name_cluster_topics(query, clustered_topics, paper_titles) if bertopic_status == "ok" else []

        raw_cluster_topics = [
            TopicRecord(
                topic_name=cluster["topic_name"],
                source="oracle-derived",
                representative_papers=cluster["representative_titles"][:5],
                evidence_titles=cluster["keywords"],
                metadata={
                    "cluster_id": cluster["cluster_id"],
                    "cluster_size": len(cluster["indices"]),
                    "keywords": cluster["keywords"],
                    "llm_reason": cluster["reason"],
                },
            )
            for cluster in named_clusters
        ]

        supplementary_topics = []
        if anchor_surveys_count >= 3:
            supplementary_topics = self._collect_supplementary_topics(base_topics, raw_cluster_topics)
            final_topics = [*base_topics, *supplementary_topics]
            confidence = "high"
        elif anchor_surveys_count in {1, 2}:
            supplementary_topics = self._gap_fill_topics(base_topics, raw_cluster_topics)
            final_topics = [*base_topics, *supplementary_topics]
            confidence = "medium"
        else:
            supplementary_topics = raw_cluster_topics
            final_topics = supplementary_topics
            confidence = "low"
        print(
            "GoldenTopicGenerator::Output "
            f"confidence={confidence} base={len(base_topics)} "
            f"supplementary={len(supplementary_topics)} final={len(final_topics)}"
        )

        return {
            "golden_topics": self._serialize_topics(final_topics),
            "base_topics": self._serialize_topics(base_topics),
            "supplementary_topics": self._serialize_topics(supplementary_topics),
            "metadata": {
                "confidence": confidence,
                "anchor_surveys_count": anchor_surveys_count,
                "base_topic_count": len(base_topics),
                "supplementary_topic_count": len(supplementary_topics),
                "library_keep_ratio": self.library_keep_ratio,
                "library_similarity": library_similarity,
                "bertopic": {
                    "status": bertopic_status,
                    "cluster_count": len(clustered_topics),
                    "used_seed_topics": bool(seed_topic_list),
                },
            },
        }
