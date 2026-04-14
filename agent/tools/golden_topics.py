from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .sbert_client import SentenceTransformerClient
from .tool_config import ToolConfig
from .utils import cosine_similarity_matrix, normalize_text

try:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
except Exception:
    BERTopic = None
    HDBSCAN = None
    CountVectorizer = None
    UMAP = None


@dataclass
class TopicRecord:
    topic_name: str
    source: str
    representative_papers: List[str]
    evidence_titles: List[str]
    metadata: Dict[str, Any]


class GoldenTopicGenerator:
    def __init__(self, config: ToolConfig):
        self.sbert = SentenceTransformerClient(config.sbert_server_url)
        self.anchor_merge_threshold = 0.85
        self.base_overlap_threshold = 0.5
        self.bertopic_min_topic_size = 5

    def _topic_label(self, topic: Any) -> str:
        if isinstance(topic, dict):
            return topic.get("topic_name") or topic.get("label") or topic.get("topic") or ""
        return str(topic or "")

    def _flatten_anchor_titles(self, anchor_surveys: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 中文注释：阶段一的输入整理，把每篇 anchor survey 的层级标题全部展平，并保留来源信息。
        records = []
        seen = set()
        for survey_title, survey_info in anchor_surveys.items():
            for title_path in survey_info.get("titles", []):
                normalized = normalize_text(title_path)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                records.append(
                    {
                        "title_path": title_path,
                        "survey_title": survey_title,
                        "paper_title": survey_info.get("paper", {}).get("title", survey_title),
                    }
                )
        return records

    def _merge_similar_anchor_titles(self, title_records: List[Dict[str, Any]]) -> List[TopicRecord]:
        # 中文注释：阶段一的核心步骤，用 SentenceBERT 合并语义相近的 section 标题，保证结果可审计。
        if not title_records:
            return []

        title_texts = [record["title_path"] for record in title_records]
        embeddings = self.sbert.embed(title_texts)
        similarity = cosine_similarity_matrix(embeddings, embeddings)
        parent = list(range(len(title_records)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int):
            root_x, root_y = _find(x), _find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        for i in range(len(title_records)):
            for j in range(i + 1, len(title_records)):
                if float(similarity[i, j]) >= self.anchor_merge_threshold:
                    _union(i, j)

        groups = defaultdict(list)
        for idx, record in enumerate(title_records):
            groups[_find(idx)].append(record)

        topics = []
        for group_records in groups.values():
            sorted_titles = sorted(
                {record["title_path"] for record in group_records},
                key=lambda value: (value.count(">"), len(value), value.lower()),
            )
            topic_name = sorted_titles[0]
            representative_papers = sorted({record["paper_title"] for record in group_records})
            evidence_titles = sorted_titles
            topics.append(
                TopicRecord(
                    topic_name=topic_name,
                    source="anchor-derived",
                    representative_papers=representative_papers,
                    evidence_titles=evidence_titles,
                    metadata={
                        "anchor_surveys": sorted({record["survey_title"] for record in group_records}),
                        "merge_size": len(group_records),
                    },
                )
            )
        topics.sort(key=lambda item: (len(item.evidence_titles), len(item.representative_papers), item.topic_name), reverse=True)
        return topics

    def _build_seed_topic_list(self, base_topics: List[TopicRecord]) -> List[List[str]]:
        seed_topic_list = []
        for topic in base_topics:
            tokens = [token for token in topic.topic_name.replace(">", " ").split() if token]
            if tokens:
                seed_topic_list.append(tokens[:8])
        return seed_topic_list

    def _align_cluster_label(self, cluster_label: str, base_topics: List[TopicRecord]) -> tuple[str, str]:
        # 中文注释：在 anchor survey 数量不足时，用 base topics 对聚类结果做标签校正。
        if not base_topics:
            return cluster_label, "oracle-derived"
        texts = [cluster_label, *[topic.topic_name for topic in base_topics]]
        embeddings = self.sbert.embed(texts)
        scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= self.base_overlap_threshold:
            return base_topics[best_idx].topic_name, "oracle-derived(seed-aligned)"
        return cluster_label, "oracle-derived"

    def _run_bertopic(self, documents: List[str], seed_topic_list: List[List[str]] | None):
        # 中文注释：阶段二的主流程，先降维，再用 HDBSCAN 聚类，最后用 BERTopic 产出可追溯主题。
        if not documents or BERTopic is None or UMAP is None or HDBSCAN is None or CountVectorizer is None:
            return [], "bertopic_unavailable"

        embeddings = self.sbert.embed(documents)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=self.bertopic_min_topic_size, metric="euclidean", prediction_data=True)
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
        topic_groups = defaultdict(list)
        for index, topic_id in enumerate(topic_ids):
            if topic_id == -1:
                continue
            topic_groups[int(topic_id)].append(index)

        clustered_topics = []
        for topic_id, indices in topic_groups.items():
            keywords = topic_model.get_topic(topic_id) or []
            if not keywords:
                continue
            label = ", ".join(word for word, _ in keywords[:4])
            clustered_topics.append(
                {
                    "cluster_id": topic_id,
                    "label": label,
                    "keywords": [word for word, _ in keywords[:8]],
                    "indices": indices,
                }
            )
        return clustered_topics, "ok"

    def _build_supplementary_topics(
        self,
        library: Dict[str, Dict[str, Any]],
        base_topics: List[TopicRecord],
        anchor_surveys_count: int,
    ) -> tuple[List[TopicRecord], Dict[str, Any]]:
        # 中文注释：阶段二输入使用 QueryExpand 的 200 篇 library papers，标题+摘要都作为可审计材料。
        papers = list(library.values())
        documents = []
        paper_titles = []
        for paper in papers:
            title = paper.get("title", "").strip()
            abstract = (paper.get("abstract") or "").strip()
            text = f"{title}. {abstract}".strip()
            if not text:
                continue
            documents.append(text)
            paper_titles.append(title or paper.get("id", ""))

        seed_topic_list = self._build_seed_topic_list(base_topics) if anchor_surveys_count in {1, 2} else None
        clustered_topics, status = self._run_bertopic(documents, seed_topic_list)
        if status != "ok":
            return [], {"status": status, "used_seed_topics": bool(seed_topic_list)}

        base_topic_names = [topic.topic_name for topic in base_topics]
        supplementary_topics = []
        used_base_topic_names = set()

        for cluster in clustered_topics:
            cluster_label = cluster["label"]
            source = "oracle-derived"

            # 中文注释：anchor surveys 足够时，base topic 为主，过滤掉已经被 anchor surveys 覆盖的聚类。
            if anchor_surveys_count >= 3 and base_topic_names:
                texts = [cluster_label, *base_topic_names]
                embeddings = self.sbert.embed(texts)
                scores = cosine_similarity_matrix(embeddings[:1], embeddings[1:])[0]
                if float(scores.max()) > self.base_overlap_threshold:
                    continue

            # 中文注释：anchor surveys 较少时，BERTopic 结果为主，但允许用 base topics 修正标签。
            if anchor_surveys_count in {1, 2}:
                aligned_label, source = self._align_cluster_label(cluster_label, base_topics)
                cluster_label = aligned_label
                if source == "oracle-derived(seed-aligned)":
                    used_base_topic_names.add(cluster_label)

            representative_titles = [paper_titles[idx] for idx in cluster["indices"][:5] if idx < len(paper_titles)]
            supplementary_topics.append(
                TopicRecord(
                    topic_name=cluster_label,
                    source=source,
                    representative_papers=representative_titles,
                    evidence_titles=cluster["keywords"],
                    metadata={
                        "cluster_id": cluster["cluster_id"],
                        "cluster_size": len(cluster["indices"]),
                        "keywords": cluster["keywords"],
                    },
                )
            )

        if anchor_surveys_count in {1, 2}:
            for topic in base_topics:
                if topic.topic_name not in used_base_topic_names:
                    supplementary_topics.append(
                        TopicRecord(
                            topic_name=topic.topic_name,
                            source="anchor-derived(soft-seed)",
                            representative_papers=topic.representative_papers,
                            evidence_titles=topic.evidence_titles,
                            metadata=topic.metadata | {"note": "anchor survey 数量不足时作为软约束保留"},
                        )
                    )

        return supplementary_topics, {"status": "ok", "used_seed_topics": bool(seed_topic_list)}

    def _serialize_topics(self, topics: List[TopicRecord]) -> List[Dict[str, Any]]:
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

    async def __call__(self, query: str, anchor_data: Dict[str, Any], library: Dict[str, Dict[str, Any]]):
        del query

        # 中文注释：阶段一，从 anchor surveys 的层级标题直接抽取 base topics。
        anchor_surveys = anchor_data.get("downloaded", {})
        anchor_title_records = self._flatten_anchor_titles(anchor_surveys)
        base_topics = self._merge_similar_anchor_titles(anchor_title_records)

        # 中文注释：阶段二，用 BERTopic 从 library papers 中寻找补充 topics。
        anchor_surveys_count = len(anchor_surveys)
        supplementary_topics, supplementary_meta = self._build_supplementary_topics(library, base_topics, anchor_surveys_count)

        # 中文注释：根据 anchor survey 数量决定最终 topics 的组织方式和可信度说明。
        if anchor_surveys_count >= 3:
            mode = "anchor_consensus_primary"
            confidence = "high"
            final_topics = [*base_topics, *supplementary_topics]
            note = "base topics 来自主流 anchor surveys，共识性较强；supplementary topics 仅作为补充。"
        elif anchor_surveys_count in {1, 2}:
            mode = "weak_anchor_guided"
            confidence = "medium"
            final_topics = supplementary_topics or base_topics
            note = "anchor surveys 数量不足，BERTopic 结果为主，base topics 仅用于软约束和标签校正。"
        else:
            mode = "cluster_only"
            confidence = "low"
            final_topics = supplementary_topics
            note = "未找到 anchor surveys，topic list 完全来自文献聚类，topic coverage 结果应视为低可信度。"

        return {
            "golden_topics": self._serialize_topics(final_topics),
            "base_topics": self._serialize_topics(base_topics),
            "supplementary_topics": self._serialize_topics(supplementary_topics),
            "metadata": {
                "mode": mode,
                "confidence": confidence,
                "note": note,
                "anchor_surveys_count": anchor_surveys_count,
                "base_topic_count": len(base_topics),
                "supplementary_topic_count": len(supplementary_topics),
                "bertopic": supplementary_meta,
            },
        }
