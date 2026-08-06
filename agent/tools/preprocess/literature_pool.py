from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from typing import Any


from ..utility.academic_engine import get_academic_engine
from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.paper_elements import Paper, Section
from ..utility.tool_config import ToolConfig


OPENALEX_LITERATURE_POOL_SELECT = "id,title,abstract_inverted_index,cited_by_count,counts_by_year,publication_date,referenced_works"
S2_LITERATURE_POOL_SELECT = "paperId,title,abstract,year,publicationDate,citationCount,referenceCount,externalIds,venue"
TARGET_SECTION_TYPES = {"CONTENT", ""}
TARGET_SENTENCE_LABELS = {"SUMMARY", "COMPARISON", "SYNTHESIS", ""}


class BuildLiteraturePool:
    """从已引用论文、邻居扩展结果和局部引用图构建文献池。"""

    def __init__(self, config: ToolConfig):
        """初始化当前组件及其配置。"""
        self.config = config
        self.eval_date = config.evaluation_date
        self.engine = get_academic_engine(config)
        self.engine_name = (config.default_academic_search_engine or "openalex").strip().lower()

    def _source_name(self) -> str:
        """返回当前学术检索引擎的规范名称。"""
        if self.engine_name in {"semantic_scholar", "semanticscholar", "semantic scholar", "s2"}: return "semantic scholar"
        return self.engine_name or "openalex"

    def _uses_semantic_scholar(self) -> bool:
        """判断当前是否使用 Semantic Scholar 作为检索后端。"""
        return self._source_name() == "semantic scholar"

    def _select_fields(self) -> str:
        """返回当前检索后端所需的论文字段选择字符串。"""
        return S2_LITERATURE_POOL_SELECT if self._uses_semantic_scholar() else OPENALEX_LITERATURE_POOL_SELECT

    def _paper_ids(self, paper: dict[str, Any]) -> set[str]:
        """提取论文对象中可用于检索请求的所有候选标识符。"""
        ids = set()
        for key in ("id", "paperId", "corpusId"):
            if paper.get(key):
                ids.add(str(paper[key]).replace("https://openalex.org/", ""))
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids.values() if value)
        elif isinstance(raw_ids, (list, tuple, set)):
            ids.update(str(value).replace("https://openalex.org/", "") for value in raw_ids if value)
        return {item for item in ids if item}

    def _paper_doi(self, paper: dict[str, Any]) -> str:
        """从论文元数据中规范化提取 DOI。"""
        candidates = [paper.get("doi")]
        raw_ids = paper.get("ids")
        if isinstance(raw_ids, dict):
            candidates.append(raw_ids.get("doi"))
        external_ids = paper.get("external_ids") or paper.get("externalIds") or {}
        for key, value in external_ids.items():
            if str(key).lower() == "doi": candidates.append(value)
        for value in candidates:
            if not value: continue
            doi = str(value).strip().lower()
            doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            doi = re.sub(r"^doi:", "", doi)
            if doi: return doi
        return ""

    def _paper_key(self, paper: dict[str, Any]) -> str:
        """根据 DOI、外部标识符或标题生成稳定的论文去重键。"""
        doi = self._paper_doi(paper)
        if doi: return f"doi:{doi}"
        ids = sorted(self._paper_ids(paper))
        if ids: return f"id:{ids[0]}"
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        return f"title:{title}" if title else ""

    def _paper_aliases(self, paper: dict[str, Any]) -> set[str]:
        """生成论文的 DOI、外部标识符和标题别名集合。"""
        aliases = set()
        doi = self._paper_doi(paper)
        if doi: aliases.add(f"doi:{doi}")
        aliases.update(f"id:{paper_id}" for paper_id in self._paper_ids(paper))
        title = re.sub(r"\s+", " ", paper.get("title", "") or "").strip().lower()
        if title: aliases.add(f"title:{title}")
        return aliases

    def _deduplicate_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按照论文别名去重，同时保留首次出现的论文记录。"""
        seen_aliases = set()
        unique = []
        for paper in papers:
            aliases = self._paper_aliases(paper)
            if not aliases or seen_aliases & aliases:
                continue
            seen_aliases.update(aliases)
            unique.append(paper)
        return unique

    def _metadata_sources(self, info: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """从信息条目中提取按来源组织的元数据。"""
        metadata = info.get("metadata") or {}
        if not isinstance(metadata, dict): return {}
        if "openalex" in metadata or "semantic scholar" in metadata:
            return {source: paper for source, paper in metadata.items() if isinstance(paper, dict)}
        return {self._source_name(): metadata} if metadata else {}

    def _default_engine_paper(self, info: dict[str, Any]) -> dict[str, Any] | None:
        """选择当前检索引擎优先使用的论文元数据，并提供来源回退。"""
        sources = self._metadata_sources(info)
        if not sources: return
        preferred = sources.get(self._source_name())
        if preferred: return preferred
        for source in ("openalex", "semantic scholar"):
            if sources.get(source): return sources[source]
        return next(iter(sources.values()))

    def _record_retrieval_source(
        self,
        pool: dict[str, dict[str, Any]],
        pool_key: str,
        source: dict[str, Any],
    ) -> None:
        """记录论文的一次检索来源，并避免重复记录相同来源。"""
        clean_source = {
            key: value
            for key, value in source.items()
            if value is not None and value != ""
        }
        sources = pool[pool_key].setdefault("retrieval_sources", [])
        if clean_source not in sources:
            sources.append(clean_source)

    def _merge_retrieval_summary(
        self,
        pool: dict[str, dict[str, Any]],
        pool_key: str,
        label: str,
        retrieval: dict[str, Any],
    ) -> None:
        """维护兼容旧代码的检索摘要，同时保留可多次命中的来源信息。"""
        record = pool[pool_key].setdefault("retrieval", {})
        labels = pool[pool_key].setdefault("labels", [])
        if label and label not in labels:
            labels.append(label)
        for name, value in retrieval.items():
            if value is None or value == "":
                continue
            if name in {"expanded_from", "citation_key", "query_rank", "neighbor_rank"}:
                record.setdefault(name, [])
                if value not in record[name]:
                    record[name].append(value)
            else:
                record.setdefault(name, value)

    def _add_to_pool(
        self,
        pool: dict[str, dict[str, Any]],
        pool_index: dict[str, str],
        paper: dict[str, Any],
        label: str,
        citation_key: str = "",
        citation_rank: int | None = None,
    ) -> str:
        """将论文加入文献池，并维护别名索引和引用关系元数据。"""
        if not paper or not paper.get("title"): return ""
        key = self._paper_key(paper)
        if not key: return ""
        aliases = self._paper_aliases(paper)
        existing = next((pool_index[alias] for alias in aliases if alias in pool_index), "")
        if existing:
            if citation_key:
                pool[existing].setdefault("citation_keys", [])
                if citation_key not in pool[existing]["citation_keys"]:
                    pool[existing]["citation_keys"].append(citation_key)
                pool_index[f"citation:{citation_key}"] = existing
                self._merge_retrieval_summary(
                    pool,
                    existing,
                    label,
                    {
                        "citation_key": str(citation_key),
                        "citation_rank": citation_rank,
                    },
                )
                self._record_retrieval_source(
                    pool,
                    existing,
                    {
                        "type": "cited_paper",
                        "citation_key": str(citation_key),
                        "citation_rank": citation_rank,
                    },
                )
            return existing
        pool[key] = {"paper": paper, "label": label, "labels": [label] if label else []}
        if citation_key:
            pool[key]["citation_keys"] = [citation_key]
            aliases.add(f"citation:{citation_key}")
            self._merge_retrieval_summary(
                pool,
                key,
                label,
                {
                    "citation_key": str(citation_key),
                    "citation_rank": citation_rank,
                },
            )
            self._record_retrieval_source(
                pool,
                key,
                {
                    "type": "cited_paper",
                    "citation_key": str(citation_key),
                    "citation_rank": citation_rank,
                },
            )
        for alias in aliases:
            pool_index[alias] = key
        return key

    def _pool_key_for(self, pool_index: dict[str, str], paper: dict[str, Any]) -> str:
        """根据论文别名查找其在文献池中的内部键。"""
        return next((pool_index[alias] for alias in self._paper_aliases(paper) if alias in pool_index), "")

    def _target_sections(self, paper: Paper) -> list[Section]:
        """递归收集参与核心引用分析的目标章节。"""
        sections = []

        def walk(section: Section):
            """递归遍历章节树或句子结构。"""
            # if section.functional_type in TARGET_SECTION_TYPES:
            # 暂定为收集所有文献，到4.2再过滤种子文献。为了消融实验。
            sections.append(section)
            for child in section.children: walk(child)

        for section in paper.children:
            walk(section)
        return sections

    def _section_core_citation_keys(self, sections: list[Section]) -> list[str]:
        """从目标章节中特定句子标签的引用中提取核心引用键。"""
        keys = []

        def walk(section: Section):
            """递归遍历章节树或句子结构。"""
            for paragraph in section.paragraphs:
                for sentence in paragraph.sentences:
                    if sentence.label in TARGET_SENTENCE_LABELS:
                        keys.extend(normalize_citation_keys(sentence.citations))
            for child in section.children:
                walk(child)

        for section in sections: walk(section)
        return list(dict.fromkeys(keys))

    def _query_ids(self, paper: dict[str, Any]) -> list[str]:
        """提取可传递给引用关系查询接口的论文标识符。"""
        ids = []
        for key in ("id", "paperId"):
            if paper.get(key):
                ids.append(str(paper[key]).replace("https://openalex.org/", ""))
        ids.extend(self._paper_ids(paper))
        return list(dict.fromkeys(item for item in ids if item))

    async def _expand_one(self, paper: dict[str, Any], direction: str, filter: dict, offset: int, limit: int) -> list[dict[str, Any]]:
        """扩展一篇种子论文的被引用或引用关系。"""
        method = self.engine.get_citations if direction == "cited_by" else self.engine.get_references
        request_kwargs = {"sort": "cited_by_count:desc"} if not self._uses_semantic_scholar() else {}
        for paper_id in self._query_ids(paper):
            try:
                if self._uses_semantic_scholar() and direction == "cited_by":
                    result = await method(paper_id, offset=offset, limit=limit, select=self._select_fields(), **filter)
                else:
                    result = await method(paper_id, offset=offset, limit=limit, select=self._select_fields(), filter=filter, **request_kwargs)
            except TypeError:
                result = await method(paper_id, offset=offset, limit=limit, fields=self._select_fields(), filter=filter, **request_kwargs)
            except Exception:
                continue
            papers = result.get("results", []) or []
            if papers:
                return papers
        return []

    def _query_terms(self, query: list[str] | str, paper: Paper) -> list[str]:
        """生成需要独立检索的查询词，并保留其来源顺序。"""
        values = query if isinstance(query, list) else [query]
        terms = []
        for value in values:
            term = re.sub(r"\s+", " ", str(value or "")).strip()
            if term and term not in terms:
                terms.append(term)
        if not terms and paper.title:
            terms.append(re.sub(r"\s+", " ", paper.title).strip())
        limit = max(1, int(self.config.literature_pool_max_query_keywords))
        return terms[:limit]

    async def _search_one(
        self,
        query_term: str,
        search_filter: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """执行一个查询词的文献检索并返回原始排序结果。"""
        payload = await self.engine.search_works(
            search=query_term,
            filter=search_filter,
            per_page=int(self.config.literature_pool_query_search_limit),
            select=self._select_fields(),
        )
        return payload.get("results", []) or []

    def _query_text(self, query: list[str] | str, paper: Paper) -> str:
        """将查询词列表合并为文献池检索文本，并在缺失时回退到论文标题。"""
        values = query if isinstance(query, list) else [query]
        text = " ".join(str(item or "").strip() for item in values if str(item or "").strip())
        return re.sub(r"\s+", " ", text or paper.title or "").strip()

    def _referenced_work_aliases(self, work_id: str) -> list[str]:
        """将 referenced_works 标识符转换为池内查找别名。"""
        value = str(work_id or "").replace("https://openalex.org/", "").strip()
        return [f"id:{value}"] if value else []

    def _graph_dict(self, pool: dict[str, dict[str, Any]], edges: set[tuple[str, str]]) -> dict[str, Any]:
        """根据文献池节点和有向引用边生成图结构，并更新局部被引计数。"""
        nodes = sorted(pool)
        edges = {(source, target) for source, target in edges if source in pool and target in pool and source != target}
        out_counts = {node: 0 for node in nodes}
        for source, _target in edges:
            out_counts[source] += 1
        for key, count in out_counts.items():
            pool[key]["local_cited_by_count"] = count
            pool[key].setdefault("paper", {})["local_cited_by_count"] = count
        return {
            "nodes": nodes,
            "edges": [
                {"source": source, "target": target}
                for source, target in sorted(edges)
            ],
        }

    async def _build_unfiltered_graph(self, query: list[str], paper: Paper, paper_content_map: dict[str, Any]) -> dict[str, Any]:
        """不执行语义过滤，构建完整的保留文献池及其引用关系图。"""
        to_publication_date = (self.eval_date - timedelta(days=30)).strftime("%Y-%m-%d")
        batch_size = max(1, int(self.config.literature_pool_neighbor_batch_size))
        expansion_filter = {"to_publication_date": to_publication_date}
        pool: dict[str, dict[str, Any]] = {}
        pool_index: dict[str, str] = {}
        edges: set[tuple[str, str]] = set()

        def add(paper_data: dict[str, Any], label: str, **retrieval: Any) -> str:
            """将论文加入全图，并追加本次命中的完整来源。"""
            key = self._add_to_pool(pool, pool_index, paper_data, label)
            if not key:
                return ""
            self._merge_retrieval_summary(pool, key, label, retrieval)
            self._record_retrieval_source(pool, key, {"type": label, **retrieval})
            return key

        # 将已引用文章加入文献池
        for citation_rank, (citation_key, info) in enumerate((paper_content_map or {}).items(), 1):
            cited_paper = self._default_engine_paper(info if isinstance(info, dict) else {})
            if cited_paper:
                key = self._add_to_pool(
                    pool,
                    pool_index,
                    cited_paper,
                    "cited_paper",
                    citation_key=str(citation_key),
                    citation_rank=citation_rank,
                )

        # 种子文献：定义为CONTENT类型中的所有引用文献。
        sections = self._target_sections(paper)
        seed_papers = []
        for key in self._section_core_citation_keys(sections):
            info = (paper_content_map or {}).get(key)
            if isinstance(info, dict):
                cited_paper = self._default_engine_paper(info)
                if cited_paper:
                    seed_papers.append(cited_paper)
        seed_papers = self._deduplicate_papers(seed_papers) or [
            self._default_engine_paper(info) for info in (paper_content_map or {}).values()
            if isinstance(info, dict) and self._default_engine_paper(info)
        ]
        seed_papers = self._deduplicate_papers([item for item in seed_papers if item])
        logging.info("Unfiltered literature graph starts with %d seeds and %d cited papers", len(seed_papers), len(pool))

        query_terms = self._query_terms(query, paper)
        search_results = await asyncio.gather(
            *(self._search_one(term, dict(expansion_filter)) for term in query_terms),
            return_exceptions=True,
        )
        for query_index, (query_term, result) in enumerate(
            zip(query_terms, search_results),
            1,
        ):
            if isinstance(result, Exception):
                logging.error("Unfiltered query search failed for %r: %s", query_term, result)
                continue
            for rank, candidate in enumerate(result, 1):
                add(
                    candidate,
                    "query_search",
                    query=query_term,
                    query_index=query_index,
                    rank=rank,
                )
        logging.info(
            "Unfiltered query search complete: %d queries, graph=%d nodes/%d edges",
            len(query_terms),
            len(pool),
            len(edges),
        )

        active = [(index, direction) for index in range(len(seed_papers)) for direction in ("cited_by", "cites")]
        offset = 0
        round_index = 0
        while active:
            round_index += 1
            results = await asyncio.gather(
                *(self._expand_one(seed_papers[index], direction, dict(expansion_filter), offset, batch_size)
                  for index, direction in active),
                return_exceptions=True,
            )
            next_active = []
            fetched = 0
            for (seed_index, direction), result in zip(active, results):
                if isinstance(result, Exception):
                    logging.error("Unfiltered neighbor expansion failed: %s", result)
                    continue
                seed = seed_papers[seed_index]
                seed_key = self._pool_key_for(pool_index, seed)
                papers = result or []
                fetched += len(papers)
                for rank, candidate in enumerate(papers, 1):
                    key = add(
                        candidate,
                        "neighbor_expansion",
                        expanded_from=seed_key,
                        expanded_from_title=seed.get("title", ""),
                        expanded_from_ids=self._query_ids(seed),
                        direction=direction,
                        offset=offset,
                        rank=offset + rank,
                        round=round_index,
                    )
                    if not key or not seed_key:
                        continue
                    if direction == "cited_by":
                        edges.add((seed_key, key))
                    else:
                        edges.add((key, seed_key))
                if len(papers) >= batch_size:
                    next_active.append((seed_index, direction))
            logging.info("Unfiltered neighbor round %d offset=%d: fetched=%d, graph=%d nodes/%d edges", round_index, offset, fetched, len(pool), len(edges))
            active = next_active
            offset += batch_size

        for source_key, item in list(pool.items()):
            for work_id in item.get("paper", {}).get("referenced_works", []) or []:
                target_key = next((pool_index[alias] for alias in self._referenced_work_aliases(work_id) if alias in pool_index), "")
                if target_key and target_key != source_key:
                    edges.add((target_key, source_key))
        graph = self._graph_dict(pool, edges)
        logging.info("Unfiltered literature graph complete: %d papers, %d edges", len(pool), len(graph["edges"]))
        return {"literature_pool": pool, "citation_graph": graph}

    async def __call__(self, query: list[str], paper: Paper, paper_content_map: dict[str, Any] | None = None):
        """构建论文的未过滤文献池及其引用关系图。"""
        return await self._build_unfiltered_graph(query, paper, paper_content_map or paper.references or {})
