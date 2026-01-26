QUERY_EXPANSION_PROMPT = '''You are a Senior Research Librarian specializing in Systematic Literature Reviews. 
Your goal is to generate 4 distinct, high-recall search queries for the topic: "{query}"

### CONTEXT
- **Goal**: We need to build a candidate pool of ~4000 papers to identify foundational "hubs" and specific research papers.
- **Previous Attempts**: {prev_query}
- **Constraint**: Do not repeat previous queries. If previous queries yielded low results, broaden the terminology (e.g., use "Deep Learning" instead of "ResNet50").
- **Constraint**: Do not use wildcards ('*' or '?') in query, your search engine does not support them.

### SEARCH STRATEGY: "CORE + BRIDGES"
Generate exactly 4 search queries to cover the topic's core evidence and its disconnected theoretical foundations.

1. **The Core Anchor (2 Queries)**
   - Target the intersection of the **Subject** AND the **Population/Context**.
   - Use standard synonyms for both.
   - **Goal:** Find the dense cluster of applied research papers.
   - *Constraint:* MUST include both Subject and Population terms.

2. **The Theoretical Bridge (1 Query)**
   - Target the **Parent Discipline** or **Mechanism** that explains *why* the intervention works.
   - **Goal:** Find broad theoretical papers (e.g., "Health Promotion," "Behavioral Theory") that may not mention the specific population.
   - *Constraint:* You MAY drop the "Population" term. You MUST keep the "Subject" or "Parent Field" term.

3. **The Methodological Bridge (1 Query)**
   - Target the **Tools**, **Designs**, or **Evaluation Standards** used.
   - **Goal:** Find protocols, validation studies, or general tools (e.g., "MMAT", "Consolidated Framework").
   - *Constraint:* You MAY drop the "Subject" term if focusing on a tool used in this Population.

### PROHIBITED:
- Do NOT generate "Ghost Queries" that have NO anchor (e.g., just "Policy" AND "Evaluation").
- Do NOT use specific publication years or "Recent".

### OUTPUT FORMAT
Provide a JSON object with a brief strategy for the expansion and the queries.

```json
{{
  "strategy": "Analysis of why previous queries failed and how these queries will broaden the search.",
  "queries": ["query 1", "query 2", "query 3", "query 4"]
}}
```
'''

CLAIM_SEGMENTATION_PROMPT = '''
You are a rigorous scientific information extractor. Your task is to extract verifiable units (also called claims) from a paragraph. Each verifiable unit must be suitable for independent factual verification against the cited reference(s).

========================
INPUT
========================

Full Paragraph:
\"\"\"{range}\"\"\"

Target Sentence (contains citation markers):
\"\"\"{text}\"\"\"

Citation markers in this sentence:
{keys}

========================
CORE DEFINITIONS
========================

A *verifiable unit* (claim) is the minimal statement that:
- Is directly supported by the cited reference(s), and
- Can be independently checked against a single paper.

Claim Types:
- background: definition, scope, or topic description (what something is about)
- action: what a paper does (proposes, evaluates, studies, analyzes, shows)
- result: capability assertions, comparisons, performance, limitations, conclusions

========================
STRICT EXTRACTION RULES
========================

1. Context Window Constraint:
   - You may ONLY use information from a local window of at most 3 consecutive sentences:
     the anchor sentence and its immediately adjacent sentences.
   - You MUST NOT use any information outside this 3-sentence window.

2. Anchor-Centered Constraint:
   - Every extracted claim MUST be anchored to the target sentence.
   - The claim must represent what the citation in the target sentence is used to support.
   - Do NOT extract claims unrelated to the citation purpose.

3. Minimality Constraint:
   - Extract the minimal verifiable statement required for factual checking.
   - Do NOT include background exposition, motivation, or elaboration
     unless strictly necessary to understand the anchored claim.

4. Coreference Resolution Constraint:
   - You SHOULD attempt to resolve pronouns or implicit references
     using only the allowed 3-sentence context window.
   - Replace pronouns (e.g., it, they, this model, such task) with explicit entities or concepts.
   - If a pronoun cannot be clearly resolved within the window,
     you MUST omit that information rather than guessing.

5. Token Legitimacy Constraint:
   - Every word in the extracted claim MUST appear verbatim in the allowed context window,
     EXCEPT for the optional phrase "this paper" at the beginning of a claim.
   - You MUST NOT introduce new words, paraphrases, or inferred terminology.

6. Multiple Citations Handling:
   - If multiple citations in the target sentence refer to the SAME claim,
     output ONE claim with multiple citation keys.
   - If multiple citations correspond to DISTINCT contributions,
     output separate claims, each with exactly one citation key.

7. Paragraph-Level Writing Note:
   - Even if the paragraph describes one method or concept across multiple sentences,
     you MUST NOT treat the entire paragraph as a single claim.
   - Claims must remain citation-centered, minimal, and independently verifiable.

========================
OUTPUT FORMAT
========================

Return a JSON object in the following format:

```json
{{
  "claims": [
    {{
      "claim": "<verbatim reconstructed statement>",
      "claim_type": "background" | "action" | "result",
      "citation_markers": ["<citation_marker>", "..."]
    }}
  ]
}}
```

========================
FINAL CHECK BEFORE OUTPUT
========================

Before producing the final answer, ensure that:
- All claims satisfy the 3-sentence window constraint.
- All tokens are traceable to the allowed context (except optional 'this paper').
- No unresolved pronouns remain.
- Each claim can be independently verified using its citation(s).
'''

CLAIM_SCHEMA = {
    "type": "object", 
    "required": ["claims"],
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object", 
            "required": ["claim", "claim_type", "citation_markers"],
            "properties": {
                "claim": {"type": "string", "minLength": 1},
                "claim_type": {"type": "string", "enum": ["background", "action", "result"]},
                "citation_markers": {"type": "array", "items": {"type": "string"}}
            }
        }}
    }
}

FACTUAL_CORRECTNESS_PROMPT = '''
You are a factual correctness verifier for academic surveys. Given:

- A claim extracted from a survey, and
- The paper that it cites (including {content_type})

Determine whether the claim is supported by the cited paper. Your judgment should be one of the following:

- SUPPORTED: the claim is clearly supported by the evidence.
- REFUTED: the claim is clearly contradicted by the evidence.
- NEUTRAL: the claim is not mentioned in the evidence, or there's no sufficient information to verify if the claim is supported or refuted.

**Important:** If your judgment is "SUPPORTED" or "REFUTED", you MUST provide verbatim evidence from the content of the cited paper to support that.

Your output should be a single JSON object only:

```json
{{
  "judgment": "SUPPORTED" | "REFUTED" | "NEUTRAL",
  "evidence": "verbatim evidence from the cited paper, if judgment == SUPPORTED or REFUTED" | "" (if judgment == NEUTRAL)
}}
```

### Claim
{claim}

### Evidence
{text}
'''

ANCHOR_SURVEY_SELECT = """You are selecting anchor surveys for evaluating a target survey titled "{query}".

From the candidate survey list, select 1 to 5 surveys that can serve as STRUCTURAL and CONCEPTUAL ANCHORS.

Strict inclusion criteria:
1. The primary research object of the survey MUST be Transformer models or Transformer-based architectures themselves.
2. The survey MUST organize content around model architecture, pretraining strategies, architectural variants, or scaling laws.
3. Transformers must be the central organizing principle, not one method among many.

Strict exclusion criteria:
- Exclude non-survey papers.
- Exclude task-centric surveys (e.g., sentiment analysis, stance detection, text generation, ASR, QA).
- Exclude application- or domain-specific surveys (e.g., healthcare, education, customer service).
- Exclude surveys where Transformers are discussed only as a technique applied to another main topic.

For each selected survey:
- Provide the title
- Provide a 1–2 sentence justification explicitly explaining WHY it is architecture-centric and Transformer-first.

If no candidate satisfies these criteria, return an empty list and explain why.

My survey list are:

{titles}

Your output format should be a JSON object containing anchor surveys you selected, as follows:

```json
{{
  "surveys": [
    {{
      "title": "Survey 1",
      "reason": "Reason",
    }},
    ...
  ]
}}
```

or an empty list with reasons if no anchor surveys are found:

```json
{{
  "surveys": [],
  "reason": "Reason"
}}
```
"""

TOPIC_AGGREGATION_PROMPT = """
You are an expert researcher tasked with synthesizing a survey-level topic structure from a collection of academic papers. Your goal is to identify high-level research topics that would reasonably appear as major sections in a well-written survey on the given query.

### Target query: {query}

The query defines the conceptual scope of the survey. Topics must fall within this scope, not merely mention it.

### Paper Collection

You are given:
* High-priority evidences: a list of section and subsection names extracted from anchor surveys for this query.
* Low-priority evidences: a list of other relevant paper titles retrieved for this query.
Note that not all sources are equally reliable.

[High-priority evidence]
Survey section names extracted from anchor surveys:
{anchors}

[Low-priority evidence]
Titles and abstracts of other relevant papers:
{surveys}

### Your Task

From the paper title collection, induce 5–12 high-level survey topics. Each topic should:

1. Represent a recurring research direction or theme, not a single paper.
2. Be appropriate as a top-level section in a survey.
3. Be clearly within the semantic scope of the query.
4. Be methodological or conceptual, not a downstream application domain unless the application itself is central to the query.

### Important Constraints

A topic must be:
* short (≤ 8 words)
* noun-phrase like
* suitable as a section header
* checkable by surface semantic similarity

Do NOT create topics that are purely:
- Irrelevant to the query.
- Application domains (e.g., radiology, neurosurgery, clinical decision making)
- Neighboring fields outside the query scope (e.g., speech recognition if the query is about NLP)
- If a paper applies the queried method to another field, treat it as evidence, not a standalone topic.
- If multiple papers treat a specific model family (e.g., ChatGPT as a representative LLM) as a recurring focus within the query scope, it may form a topic.

### Output Format

Return a JSON object:
```json
{{
  "topics": [
    {{
      "topic_name": "...",
      "representative_papers": ["title1", "title2", "..."]
    }}
  ]
}}
```

### Sanity Checks (must satisfy internally)

- Every topic should be supported by multiple papers.
- The union of topics should cover most papers, but not necessarily all.
- Topics should be distinct and non-overlapping at a high level.
"""

FINAL_AGGREGATION_PROMPT = ''''''
