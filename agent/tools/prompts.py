QUERY_EXPANSION_PROMPT = '''You are a Senior Research Librarian specializing in Systematic Literature Reviews. 
Your goal is to generate 4 distinct, high-recall search queries for the topic or literature review: "{query}"

### CONTEXT
- **Goal**: Build a sufficiently broad candidate pool of papers to support downstream identification of anchor papers, oracle papers, and research topics.
- **Search Strategy**: Prioritize recall over precision.
- **Constraint**: Queries should be semantically distinct and cover complementary perspectives of the same research area.
- **Constraint**: Do NOT use wildcards ('*' or '?'). The search engine does not support them.

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
  "strategy": "Brief explanation of how the queries jointly maximize coverage of the research landscape.",
  "core_anchor": ["query 1", "query 2"],
  "theoretical_bridge": "query 3",
  "methodological_bridge": "query 4"
}}
```
'''

SURVEY_SPECIFIED_QUERY_EXPANSION = """You are constructing a HIGH-PRECISION academic search query.

### Context
You are about to evaluate a survey titled:
"{query}"

To do so, you want to retrieve only the most canonical, field-defining papers or surveys that a knowledgeable researcher would EXPECT to see referenced.

### Your Goal
Generate EXACTLY ONE search query that prioritizes PRECISION over RECALL.

### Design Rules (VERY IMPORTANT)
1. The query must reflect the STANDARD name of the core method or concept.
2. Use AND to constrain the scope if necessary.
3. Avoid OR unless the terms are near-identical synonyms used interchangeably by experts.
4. Do NOT include:
   - Specific tasks
   - Benchmarks or datasets
   - Applications or domains
   - Model variants or product names
5. The query should be understandable and reasonable if read by a domain expert.
6. Survey-specific keywords (e.g., "survey", "review", "overview") will be ADDED MANUALLY later — DO NOT include them.

### Failure Is Acceptable
- It is acceptable if this query retrieves very few or zero results.
- Do NOT broaden the query to guarantee results.

### Output Format
Return a JSON object:

```json
{{
  "query": "..."
}}
```
"""

QUERY_SCHEMA = {
    "type": "object", 
    "required": ["strategy", "core_anchor", "theoretical_bridge", "methodological_bridge"],
    "properties": {
        "strategy": {"type": "string"},
        "core_anchor": {"type": "array", "items": {"type": "string"}},
        "theoretical_bridge": {"type": "string"},
        "methodological_bridge": {"type": "string"}
    }
}

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
   - Do NOT include background exposition, motivation, or elaboration unless strictly necessary to understand the anchored claim.

4. Coreference Resolution Constraint:
   - You SHOULD attempt to resolve pronouns or implicit references using only the allowed 3-sentence context window.
   - Replace pronouns (e.g., it, they, this model, such task) with explicit entities or concepts.
   - If a pronoun cannot be clearly resolved within the window, you MUST omit that information rather than guessing.

5. Token Legitimacy Constraint:
   - Every word in the extracted claim MUST appear verbatim in the allowed context window, EXCEPT for the optional phrase "this paper" at the beginning of a claim.
   - You MUST NOT introduce new words, paraphrases, or inferred terminology.

6. Multiple Citations Handling:
   - If multiple citations in the target sentence refer to the SAME claim, output ONE claim with multiple citation keys.
   - If multiple citations correspond to DISTINCT contributions, output separate claims, each with exactly one citation key.

7. Paragraph-Level Writing Note:
   - Even if the paragraph describes one method or concept across multiple sentences, you MUST NOT treat the entire paragraph as a single claim.
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
EXAMPLES
========================

### Example 1

Full Paragraph:
\"\"\"In contrast, more complex tasks have become the mainstream benchmarks for assessing the capabilities of LLMs. 
These include tasks such as mathematical reasoning [225, 236, 243] and structured data inference [86, 151]. 
Overall, LLMs show great potential in reasoning and show a continuous improvement trend, but still face many challenges and limitations, requiring more in-depth research and optimization.\"\"\"

Target Sentence (contains citation markers):
\"\"\"These include tasks such as mathematical reasoning [225, 236, 243] and structured data inference [86, 151].\"\"\"

Citation markers in this sentence:
225, 236, 243, 86, 151

Output:
```json
{{
  "claims": [ 
    {{ 
      "claim": "Mathematical reasoning benchmarks for assessing the capabilities of LLMs", 
      "claim_type": "background", 
      "citation_keys": ["225", "236", "243"] 
    }},
    {{
      "claim": "Structured data inference benchmarks for assessing the capabilities of LLMs",
      "claim_type": "background",
      "citation_keys": ["86", "151"]
    }}
  ] 
}}
```

### Example 2

Full Paragraph:
\"\"\"Sentiment analysis is a task that analyzes and interprets the text to determine the emotional inclination. 
It is typically a binary (positive and negative) or triple (positive, neutral, and negative) class classification problem. Evaluating sentiment analysis tasks is a popular direction. 
Liang et al. [114] and Zeng et al. [242] showed that the performance of the models on this task is usually high. 
ChatGPT’s sentiment analysis prediction performance is superior to traditional sentiment analysis methods [129] and comes close to that of GPT-3.5 [159]. 
In fine-grained sentiment and emotion cause analysis, ChatGPT also exhibits exceptional performance [218].\"\"\"

Target Sentence (contains citation markers):
\"\"\"Liang et al. [114] and Zeng et al. [242] showed that the performance of the models on this task is usually high.\"\"\"

Citation markers in this sentence:
114, 242

Output:
```json
{{
  "claims": [ 
    {{
      "claim": "Liang et al. showed that the performance of the models on this task is usually high.",  # Alternative: "This paper showed that the performance of the models on this task is usually high."
      "claim_type": "action", 
      "citation_keys": ["114"] 
    }},
    {{
      "claim": "Zeng et al. showed that the performance of the models on this task is usually high.",
      "claim_type": "action",
      "citation_keys": ["242"] 
    }}
  ] 
}}
```

### Example 3

Full Paragraph:
\"\"\"ChatGPT exhibits a strong capability for arithmetic reasoning by outperforming GPT-3.5 in the majority of tasks [159]. 
However, its proficiency in mathematical reasoning still requires improvement [6, 45, 263]. 
On symbolic reasoning tasks, ChatGPT is mostly worse than GPT-3.5, which may be because ChatGPT is prone to uncertain responses, leading to poor performance [6].\"\"\"

Target Sentence (contains citation markers):
\"\"\"However, its proficiency in mathematical reasoning still requires improvement [6, 45, 263].\"\"\"

Citation markers in this sentence:
6, 45, 263

Output:
```json
{{
  "claims": [ 
    {{ 
      "claim": "ChatGPT's proficiency in mathematical reasoning still requires improvement.", 
      "claim_type": "result", 
      "citation_keys": ["6", "45", "263"] 
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

CLAIMS_SCHEMA = {
    "type": "object", 
    "required": ["claims"],
    "properties": CLAIM_SCHEMA
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

ANCHOR_PAPER_SELECT = """You are an expert researcher preparing to evaluate a survey paper titled:

"{query}"

Before evaluating this survey, you need to understand the CORE LITERATURE of the field it claims to cover.

You are given a list of academic papers retrieved using the survey's topic query. Your task is to filter this list and select a subset of papers that together represent the CORE RESEARCH LANDSCAPE of the field.

These selected papers will later be used to:
- identify foundational and widely-accepted works,
- infer which papers are considered important by the field,
- derive anchor papers and anchor topics.

You are NOT selecting papers to cite in the survey.
You are selecting papers that someone evaluating this survey MUST be aware of.

---

### Input

My paper list are:

{titles}

---

### Selection Goal

Select approximately **30–80 papers** that collectively serve as **"consensus carriers"** for the field described by the survey title.

---

### Inclusion Criteria (strong signals)

A paper SHOULD be selected if one or more of the following is true:

1. The paper studies the **primary research object** implied by the survey title as its central focus (not as a secondary application or example).
2. The paper is **methodological, theoretical, or conceptual**, contributing models, frameworks, principles, or systematic analyses relevant to the field.
3. The paper is likely to be **commonly recognized or discussed** by researchers working in this area (e.g., foundational work, influential model, or representative approach).
4. The paper represents a **major sub-direction or paradigm** within the field.

---

### Exclusion Criteria (strong signals)

A paper SHOULD NOT be selected if:

1. The primary focus is an **application domain** where the survey topic is only a tool or one component among many.
2. The paper applies the survey topic to a **specific task, dataset, or niche setting** without contributing to the general understanding of the field.
3. The paper belongs to a **neighboring or overlapping field**, but the survey topic is not its main research subject.
4. The paper is a **broad overview of an unrelated area**, even if it briefly mentions concepts related to the survey topic.

---

### Important Clarifications

- Do NOT filter based on whether a paper is a survey or not.
- Do NOT filter based solely on citation count.
- Do NOT aim for diversity for its own sake.
- Focus on whether a paper helps define "what this field is about".

It is acceptable if:
- Some selected papers are surveys, and others are not.
- Some selected papers are older or newer.
- Some selected papers are very influential, and others are representative but less cited.

---

### Output Format

Return a JSON object with the following structure:

```json
{{
  "selected_papers": [
    {{
      "title": "...",
      "reason": "Brief explanation of why this paper helps define the core literature of the field."
    }}
  ]
}}
```

### Sanity Check (must satisfy internally)

- The selected papers, taken together, should give a reviewer enough context to judge whether the survey is missing important work.
- Removing several selected papers would noticeably reduce understanding of the field.
- The set should not be dominated by application-specific or peripheral works.

"""

ANCHOR_SURVEY_SELECT = """You are a professional academic researcher. You are selecting anchor surveys for evaluating a target survey titled "{query}".

### Candidate Surveys
You are given a list of candidate papers that are likely to be surveys or survey-like works.

Your task is to select 1–5 surveys that can serve as STRUCTURAL and CONCEPTUAL ANCHORS for evaluating another survey written for this query.

### Definition: Anchor Survey
An anchor survey is a paper that:
- Treats the query topic as its PRIMARY organizing focus (not as a side example).
- Organizes the literature into coherent conceptual or methodological dimensions.
- Provides a structural view of the field (e.g., taxonomies, categorizations, evolution, or design space).
- Would reasonably be consulted by an expert before writing or reviewing a survey on this topic.

### Strict Inclusion Criteria
A selected survey MUST:
1. Take the query topic as the main research object.
2. Use the query topic as the organizing principle of the survey.
3. Discuss multiple sub-dimensions, variants, or perspectives of the topic.

### Strict Exclusion Criteria
Exclude papers that:
- Are not surveys or survey-like syntheses.
- Focus primarily on downstream applications or domains unless the domain itself is the query.
- Mention the query topic only as one method among many.
- Are narrow task-specific summaries rather than field-level overviews.

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

### Important Note

Be conservative. It is acceptable to select fewer surveys or none if the candidates do not clearly qualify.
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
