import asyncio
from collections import defaultdict

from tools.preprocess.get_reference_surveys import GetReferenceSurveys
from tools.utility.openalex import get_openalex_client
from tools.utility.request_utils import SessionManager
from tools.utility.s2 import get_semantic_scholar_client
from tools.utility.tool_config import ToolConfig
from tools.utility.llmclient import AsyncChat
from tools.utils import extract_json


DEFAULT_QUERY = "Transformers Natural Language Processing"
S2_FIELDS_SELECT = (
    "paperId,title,abstract,year,publicationDate,citationCount,referenceCount,"
    "authors,externalIds,openAccessPdf,url,venue,fieldsOfStudy,s2FieldsOfStudy,publicationTypes"
)
OPENALEX_TOPIC_SELECT = "id,title,concepts"
PROMPT = """You are a professional academic researcher. Judge whether a candidate paper qualifies as a reference survey for evaluating a target survey on: "{query}".

### What is a Reference Survey?
A reference survey is a paper that an expert would consult BEFORE reviewing another survey on this topic — not because it covers the same narrow question, but because it maps the broader field: what sub-areas exist, what the key papers are, and what a complete treatment of the topic should look like.

### Qualify (ALL must be true)
1. The query topic is the central organizing principle of this paper — not a tool applied within medicine, law, finance, robotics, or any other domain.
2. The paper covers at least THREE distinct sub-topics within the query field (e.g., different tasks, architectures, methods, or application categories that together represent the field's breadth).
3. The paper synthesizes existing literature; it does not primarily report new experimental results.

### Disqualify (ANY is sufficient)
- Covers only ONE task or sub-area (e.g., a survey solely on sentiment analysis, image segmentation, or speech recognition is too narrow if the query covers the full field).
- The query topic appears as a method applied to a specific downstream domain as the paper's primary subject (e.g., "X for Medical Diagnosis", "X in Legal Documents").
- Not a survey: benchmark paper, position paper, tutorial, or original research paper.

### Negative Example
Query: "graph neural networks"
Candidate: "Graph Neural Networks for Drug Discovery: A Survey"
Judgment: FALSE
Reason: Primary subject is drug discovery. Graph neural networks appear as the method, not the organizing focus. Disqualified by downstream domain rule.

### Input
Title: {title}
Abstract: {abstract}

### Output (JSON only)
```json
{{
  "is_reference_survey": true,
  "reason": "One sentence on the decisive factor."
}}
```
or
```json
{{
  "is_reference_survey": false,
  "reason": "One sentence on which disqualification criterion applies."
}}
```"""
PROMPT2 = """You are a professional academic researcher. Your task is to determine whether a candidate paper qualifies as a **reference survey** for evaluating a target survey on the topic: "{query}".

### Definition
A reference survey must satisfy ONE condition:

> The candidate survey's primary subject contains ALL scientific entities present 
> in the query topic, and NO scientific entities beyond those in the query topic.

A "scientific entity" is a distinct technical concept, method, or research area named in the query (e.g., "Graph Neural Networks" and "Knowledge Graphs" are two separate entities in the query "Graph Neural Networks for Knowledge Graphs").

**True cases** — the primary subject matches the query entities exactly:

Query: "Recurrent Neural Networks Sequence Modeling"
Entities: {{Recurrent Neural Networks, Sequence Modeling}}
- "A Survey of RNN Architectures and Their Role in Sequence Modeling" → TRUE
  Primary subject = {{Recurrent Neural Networks, Sequence Modeling}} — exact match.

**False cases — extra entity** (primary subject contains entities beyond the query):

Query: "Recurrent Neural Networks Sequence Modeling"
- "Recurrent Neural Networks for Speech Recognition: A Survey" → FALSE
  Primary subject = {{Recurrent Neural Networks, Speech Recognition}} — "Speech Recognition" is an extra entity not in the query.

**False cases — missing entity** (primary subject does not cover all query entities):

Query: "Recurrent Neural Networks Sequence Modeling"
- "Deep Neural Networks: A Comprehensive Survey" → FALSE
  Primary subject = {{Deep Neural Networks}} — neither "Recurrent" nor "Sequence Modeling" is the organizing focus.

**False cases — query topic used as method only**:

Query: "Knowledge Graph Embedding"
Entities: {{Knowledge Graph, Embedding}}
- "Knowledge Graph Embedding for Drug Interaction Prediction: A Survey" → FALSE
  Primary subject = {{Drug Interaction Prediction}} — Knowledge Graph Embedding is the method, not the organizing focus.

---

### Candidate Survey
Title: {title}
Abstract: {abstract}

---

### Output Format
Return JSON only:
```json
{{
  "is_reference_survey": true,
  "entities_in_query": ["entity1", "entity2"],
  "entities_in_primary_subject": ["entity1", "entity2"],
  "reason": "One sentence explaining the match or mismatch."
}}
```

If false, set "result" to false and identify which entities are missing or extra."""


class ReferenceSurveySelect(AsyncChat):
    PROMPT: str = PROMPT2

    def _availability(self, response: str, context: dict):
        results = extract_json(response)
        assert isinstance(results['is_reference_survey'], bool)
        return {**context['paper'], **results}

    def _organize_inputs(self, inputs):
        prompt = self.PROMPT.format(query=inputs["query"], title=inputs['paper']['title'], abstract=inputs['paper']['abstract'])
        return prompt, {"paper": inputs["paper"]}


def _s2_field_categories(paper: dict) -> set[str]:
    categories = set()
    for item in paper.get("s2FieldsOfStudy") or []:
        if isinstance(item, dict) and item.get("category"):
            categories.add(str(item["category"]).strip())
    return {category for category in categories if category}


def _openalex_topic_weights(papers: list[dict]) -> dict[str, float]:
    if not papers:
        return {}
    weights = defaultdict(float)
    for paper in papers:
        for item in paper.get("concepts") or []:
            name = (item.get("display_name") or "").strip()
            if not name:
                continue
            score = item.get("score", 0.0) or 0.0
            weights[name] += float(score)
    denominator = max(1, len(papers))
    weights = {name: value / denominator for name, value in weights.items()}
    return {k: v for k, v in weights.items() if v >= 0.1}


async def semantic_scholar_query_fields(config: ToolConfig, query: str = DEFAULT_QUERY) -> list[str]:
    engine = get_semantic_scholar_client(config)
    payload = await engine.search_works(search=query, per_page=100, select=S2_FIELDS_SELECT)
    categories = set()
    for paper in payload.get("results", []) or []:
        categories.update(_s2_field_categories(paper))
    return sorted(categories)


async def semantic_scholar_reference_survey_field_overlap(query: str = DEFAULT_QUERY) -> list[dict]:
    config = ToolConfig(default_academic_search_engine='semantic_scholar')
    query_categories = set(await semantic_scholar_query_fields(config, query))
    print(f"Query categories: {query_categories}")
    source = GetReferenceSurveys(config)
    surveys = await source._search_surveys(query, limit=50)
    review_like = [paper for paper in surveys if source._is_review_like(paper)]
    print(f"referenceSurveySource: {len(review_like)} rule-filtered surveys")

    async def _single(survey: dict):
        semantic_meta = await source._resolve_semantic_scholar(survey)
        if not semantic_meta:
            return None
        refs = await source.semantic_scholar.get_references(
            semantic_meta["id"],
            limit=9999,
            select=S2_FIELDS_SELECT,
        )
        categories = set()
        for paper in refs.get("results", []) or []:
            categories.update(_s2_field_categories(paper))
        overlap = sorted(categories & query_categories)
        return {
            "title": semantic_meta.get("title") or survey.get("title", ""),
            "overlap_count": len(overlap),
            "overlap": overlap,
        }

    tasks = [asyncio.create_task(_single(survey)) for survey in review_like]
    rows = []
    for task in asyncio.as_completed(tasks):
        item = await task
        if item: rows.append(item)

    rows.sort(key=lambda item: item["overlap_count"], reverse=True)
    for item in rows:
        if item['overlap_count'] > 0:
            print(f"- {item['title']}\t{item['overlap_count']}\t{', '.join(item['overlap'])}")
    return rows


async def openalex_query_topics(config: ToolConfig, query: str = DEFAULT_QUERY) -> dict[str, float]:
    engine = get_openalex_client(config)
    payload = await engine.search_works(search=query, per_page=100, select=OPENALEX_TOPIC_SELECT)
    return _openalex_topic_weights(payload.get("results", []) or [])


def _weighted_topic_overlap(left: dict[str, float], right: dict[str, float]) -> tuple[float, dict[str, float]]:
    common = set(left) & set(right)
    contributions = {name: left[name] * right[name] for name in common}
    return sum(contributions.values()), dict(sorted(contributions.items(), key=lambda item: item[1], reverse=True))


async def openalex_reference_survey_topic_overlap(query: str = DEFAULT_QUERY) -> list[dict]:
    config = ToolConfig(default_academic_search_engine="openalex")
    query_topics = await openalex_query_topics(config, query)
    print(", ".join(f"{name}:{value:.4f}" for name, value in query_topics.items()))
    source = GetReferenceSurveys(config)
    surveys = await source._search_surveys(query, limit=50)
    review_like = [paper for paper in surveys if source._is_review_like(paper)]
    print(f"referenceSurveySource: {len(review_like)} rule-filtered surveys")

    async def _single(survey: dict):
        openalex_meta = await source._resolve_openalex(survey)
        if not openalex_meta:
            return None
        refs = await source.openalex.get_references(
            openalex_meta["id"],
            limit=9999,
            fields=OPENALEX_TOPIC_SELECT,
        )
        ref_topics = _openalex_topic_weights(refs.get("results", []) or [])
        score, overlap = _weighted_topic_overlap(query_topics, ref_topics)
        return {
            "title": openalex_meta.get("title") or survey.get("title", ""),
            "weighted_overlap": score,
            "overlap": overlap,
        }

    tasks = [asyncio.create_task(_single(survey)) for survey in review_like]
    rows = []
    for task in asyncio.as_completed(tasks):
        item = await task
        if item:
            rows.append(item)

    rows.sort(key=lambda item: item["weighted_overlap"], reverse=True)
    for item in rows:
        terms = ", ".join(f"{name}:{value:.4f}" for name, value in item["overlap"].items())
        if item['weighted_overlap'] > 0.1:
            print(f"{item['title']}\t{item['weighted_overlap']:.4f}\t{terms}")
    return rows


def _write_jsonl(name: str, rows):
    import json
    with open(name, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def llm_survey_topic_overlap(survey_engine: str = 'openalex', query: str = DEFAULT_QUERY) -> list[dict]:
    config = ToolConfig(default_academic_search_engine=survey_engine)
    source = GetReferenceSurveys(config)
    surveys = await source._search_surveys(query, limit=50)
    print(f"referenceSurveySource: {len(surveys)} surveys")
    review_like = [paper for paper in surveys if source._is_review_like(paper)]
    print(f"referenceSurveySource: {len(review_like)} rule-filtered surveys")
    llm = ReferenceSurveySelect(config.llm_server_info, config.sampling_params)

    tasks = [asyncio.create_task(llm.call(inputs={"query": query, "paper": survey})) for survey in review_like]
    rows = []
    for task in asyncio.as_completed(tasks):
        try:
            item = await task
            if item['is_reference_survey']: rows.append(item)
        except Exception as e:
            print(f"ReferenceSurveySelect {e}")

    for item in rows: print(f"- {item['title']}")
    _write_jsonl(f"{survey_engine}.jsonl", rows)


async def main():
    await SessionManager.init()
    try:
        # await semantic_scholar_reference_survey_field_overlap()
        # await openalex_reference_survey_topic_overlap()
        await llm_survey_topic_overlap('semantic_scholar')
        await llm_survey_topic_overlap('openalex')
    finally:
        await SessionManager.close()


if __name__ == "__main__":
    asyncio.run(main())
