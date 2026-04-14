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

CLAIM_CLASSIFICATION_PROMPT = '''You are a precise scientific text classifier. Your task is to determine whether a given sentence describes a concrete experimental result or performance claim that can be verified against its cited reference(s).

========================
INPUT
========================

Context Window (up to 3 consecutive sentences):
\"\"\"{context}\"\"\"

Target Sentence (contains exactly one citation marker):
\"\"\"{sentence}\"\"\"

Citation marker:
{citation_key}

========================
DEFINITION
========================

A *verifiable performance claim* is a sentence that:
- Asserts a specific, measurable result, capability, or limitation of a method, model, or system, AND
- The assertion can in principle be confirmed or refuted by reading the cited reference.

Typical verifiable performance claims include:
- Quantitative results: accuracy, scores, rankings, comparisons with numbers
- Qualitative capability assertions: "Model X outperforms Y on task Z"
- Explicitly stated limitations: "Model X fails to generalize to domain Y"
- Direct benchmark results: "This model achieves state-of-the-art on dataset X"

========================
EXCLUSION CRITERIA
========================

Answer NO if the sentence is any of the following:

1. Background definition: defines a concept, task, or field
   Example: "Language models are computational models that understand human language [36]."

2. Existence citation: cites a paper merely as an example or representative of a category
   Example: "Tasks such as mathematical reasoning [225] and structured data inference [86]."

3. Motivation or scope statement: explains why something matters or what a paper covers
   Example: "Evaluating LLMs on complex tasks has become an active research direction [12]."

4. Indirect attribution: the cited paper is not the primary source of the claimed result
   Example: "As noted by recent surveys [5], performance has improved significantly."

5. Unresolvable reference: the subject of the claim cannot be determined from the 3-sentence window
   Example: "Its proficiency still requires improvement [6]." (when "its" cannot be resolved)

========================
OUTPUT FORMAT
========================

Return a JSON object in the following format:

```json
{{
  "is_verifiable_performance_claim": true | false,
  "reason": "<one sentence explaining the decision>"
}}
```

Do NOT return anything outside the JSON object.

========================
EXAMPLES
========================

### Example 1

Context Window:
\"\"\"ChatGPT exhibits a strong capability for arithmetic reasoning by outperforming GPT-3.5 in the majority of tasks [159].
However, its proficiency in mathematical reasoning still requires improvement [6].
On symbolic reasoning tasks, ChatGPT is mostly worse than GPT-3.5 [6].\"\"\"

Target Sentence:
\"\"\"However, its proficiency in mathematical reasoning still requires improvement [6].\"\"\"

Citation marker: 6

Output:
```json
{{
  "is_verifiable_performance_claim": false,
  "reason": "The subject 'its' refers to ChatGPT based on context, but 'requires improvement' is the author's interpretive judgment rather than a specific measurable result reported in the cited paper."
}}
```

### Example 2

Context Window:
\"\"\"We evaluate our model on the SQuAD 2.0 reading comprehension benchmark.
Our approach achieves an F1 score of 87.4, surpassing the previous best result of 85.1 reported by [43].
This represents a significant improvement in extractive question answering performance.\"\"\"

Target Sentence:
\"\"\"Our approach achieves an F1 score of 87.4, surpassing the previous best result of 85.1 reported by [43].\"\"\"

Citation marker: 43

Output:
```json
{{
  "is_verifiable_performance_claim": true,
  "reason": "The sentence makes a specific quantitative comparison against a result (85.1 F1) attributed to the cited reference, which can be verified by reading that paper."
}}
```

### Example 3

Context Window:
\"\"\"Sentiment analysis is a task that analyzes text to determine emotional inclination.
It is typically a binary or triple classification problem.
Evaluating sentiment analysis tasks is a popular direction [114].\"\"\"

Target Sentence:
\"\"\"Evaluating sentiment analysis tasks is a popular direction [114].\"\"\"

Citation marker: 114

Output:
```json
{{
  "is_verifiable_performance_claim": false,
  "reason": "The sentence describes the general research landscape rather than asserting a specific experimental result or performance measurement attributable to the cited paper."
}}
```
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

FACTUAL_CORRECTNESS_PROMPT = '''You are a factual correctness verifier for academic surveys. Given:

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

TOPIC_AGGREGATION_PROMPT = """You are an expert researcher tasked with synthesizing a survey-level topic structure from a collection of academic papers. Your goal is to identify high-level research topics that would reasonably appear as major sections in a well-written survey on the given query.

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

INTERGRATION_INTENT = """You are a strict survey reviewer. You are assessing whether a given text segment explicitly shows an **intent to integrate, synthesize, or organize prior literature**, as expected in a survey paper.

### Definition

"Integration intent" means the text **explicitly states** one or more of the following:

* organizing prior work into categories, themes, or dimensions
* summarizing trends, comparisons, or common findings across multiple works
* positioning multiple studies relative to each other

It does **NOT** include:

* merely describing one paper or one method
* background definitions
* narrative mentions without synthesis language

### Instructions

1. Read the text carefully.
2. Decide whether **explicit integration intent** is present.
3. If yes, copy **the exact sentence(s)** that express this intent.
4. If no, return `false` and leave evidence empty.

### Constraints

* Do **not** infer intent if it is not explicitly stated.
* If you are unsure, choose `false`.

### Output Format

```json
{
  "integration_intent": true | false,
  "evidence": "verbatim sentence(s) from the text, or empty string if false"
}
```"""

STRUCTURE_AND_DISCUSSION = """You are a strict survey reviewer tasked with evaluating whether a given survey meets the minimal requirements of a survey. You are given the hierarchical section titles of the survey. Determine:

1. Whether the structure is **topic-driven**, as expected for a survey.
2. Whether the survey includes **explicit discussion sections** about open problems, limitations, or future directions.

### Definitions

* **Topic-driven structure**:

  * Sections are organized by themes, categories, tasks, approaches, or dimensions.
  * NOT organized around proposing a single method, model, or algorithm.
* **Discussion sections** include titles containing ideas such as:

  * future directions, open problems, challenges, limitations, outlook, discussion

### Instructions

1. Judge whether the overall structure resembles a survey.
2. Identify section titles that clearly indicate discussion of open questions or future work.
3. Copy section titles exactly as provided.

### Constraints

* Base your decision **only on section titles**, not assumptions.
* If uncertain, choose `false`.

### Output Format

```json
{
  "topic_driven": true | false,
  "discussion_sections": ["Section Title 1", "Section Title 2"]
}
```"""

IS_LANDMARK = """Given a reference and the paragraph where it appears, determine the **role** this reference plays in the narrative.

### Role Definitions

* **foundational**：introduced as a landmark, origin, or basis of the field
* **representative**：used as a typical or canonical example of a category
* **incremental**：presented as a minor improvement or extension over prior work
* **background**：mentioned for context, definition, or historical background only

### Instructions

1. Focus on how the cited work is **positioned**, not its actual importance.
2. Choose exactly one role from the defined categories.
3. Base your decision only on the provided paragraph.

### Constraints

* Do not assume importance beyond what is stated.
* If multiple roles appear, choose the **dominant** one.

### Output Format

```json
{
  "role": "foundational" | "representative" | "incremental" | "background"
  "explanation": "brief justification of the chosen role"
}
```"""

EXTRACT_METHODS = """Identify methods that are **substantively introduced** in this section.

### Task Definition

A method is considered “introduced” **only if**:

* It is discussed across **at least two consecutive sentences**
* The sentences **describe, explain, or elaborate** the method
* Mere mentions, examples, or name drops do NOT count

### Instructions

1. Examine sentence order carefully.
2. For each citation key, check whether it appears in **two or more consecutive sentences** that discuss the same method.
3. Record the sentence index range as `[start, end]`.
4. Skip any method discussed in only one sentence.

### Constraints

* Only output spans that truly correspond to method discussion.
* If no such methods exist, return an empty list.

### Output Format

```json
{
  "introduce_spans": [
    { "ref_key": "key1", "span": [3, 4] },
    { "ref_key": "key2", "span": [7, 9] }
  ]
}
```"""

SECTION_ORGANIZE = """Determine how the methods in this section are organized.

### Inputs

* Information about all methods introduced in this paper, each method containing:
  - method id (formatted as "Mi")
  - reference key
  - related text introducing this method

### Organization Types

* **grouping by criteria**: methods grouped by shared properties, categories, or dimensions
* **chronological or technical progression**: methods ordered by time or technical evolution
* **explicit comparison**: methods directly contrasted or compared
* **no clear structure**: no discernible organizing principle

### Instructions

1. Consider only the provided methods and their original order.
2. Select methods that clearly participate in an organizing pattern.
3. If selected methods are **less than half** of all methods, choose `no clear structure`.
4. Explain your reasoning briefly.

### Constraints

* Do not invent structure.
* If uncertain, choose `no clear structure`.

### Output Format

```json
{
  "organization_type": "grouping_by_criteria" | "chronological_or_technical_progression" | "explicit_comparison" | "no_clear_structure",
  "selected_methods": ["M1", "M2", "M4"],
  "justification": "..."
}
```"""

PAPER_ORGANIZE = """Infer the **overall organization principle** of the paper based on its sections.

### Inputs

* Information about all sections of the paper, each item containing:
  - section id (formatted as "Si")
  - section title 
  - number of methods introduced in this section
  - methods organization type

### Organization Types

* **grouping by criteria**: methods grouped by shared properties, categories, or dimensions
* **chronological or technical progression**: methods ordered by time or technical evolution
* **explicit comparison**: methods directly contrasted or compared
* **no clear structure**: no discernible organizing principle

### Instructions

1. Consider section titles and their organization types.
2. Identify sections that clearly follow the same organizing principle.
3. If selected sections are **less than half**, choose `no clear structure`.
4. Base justification on **section-level evidence**, not speculation.

### Constraints

* Ignore sections with very few methods.
* Do not assume global structure if evidence is weak.

### Output Format

```json
{
  "organization_type": "grouping_by_criteria" | "chronological_or_technical_progression" | "explicit_comparison" | "no_clear_structure",
  "selected_sections": ["S1", "S2", "S4"],
  "justification": "..."
}
```"""

REFUTE_ORGANIZE = """Check whether the claimed overall organization principle is **contradicted by the text**.

### Instructions

1. Look for passages that clearly violate or contradict the claimed structure.
2. Only mark contradiction if **explicit evidence exists**.
3. If no clear refutation is found, return false.

### Constraints

* Absence of support ≠ refutation.
* Cite verbatim text if refuted.

### Output Format

```json
{
  "refute": true | false,
  "evidence": "verbatim text if refuted, otherwise empty"
}
```"""

MISSING_TOPIC_CLAIM = """Determine whether the paper **explicitly states** that a given topic is excluded, and why.

### Instructions

1. Search for explicit scope limitation statements.
2. Only accept **clear declarative claims** (e.g., “we do not cover…”).
3. Do not infer justification.

### Constraints

* If justification is implicit or vague, return false.

### Output Format

```json
{
  "has_claim": true | false,
  "evidence": "verbatim text if present, otherwise empty"
}
```"""

FINAL_AGGREGATION_PROMPT = '''You are a professional research assistant. You are writing an official review report for a survey paper.

### Task

Your task is to organize the provided results into a clear and professional review report. Follow the structure below exactly:

1. Summary
2. Strengths
3. Weaknesses
4. Comments and Suggestions
5. Evidence from Automatic Evaluation
6. Overall Score

### Guidelines

- Summary: Briefly describe what the survey attempts to do based only on the provided information.
- Strengths: List strengths provided in the evaluation results.
- Weaknesses: List all weaknesses.
- Comments: Include improvement suggestions or non-critical observations.
- Evidence from Automatic Evaluation: Summarize key findings from:
  * citation correctness
  * missing papers
  * factual claim verification
  * topic coverage
  * anchor paper roles
  * section organization
  * global organization
- Overall Score: Report the score exactly as given. Should be a integer from 1-5.

Do not add any information beyond the provided evaluation results.

### Important rules

1. You MUST NOT perform new analysis or introduce new judgments.
2. You MUST ONLY use the information provided in the structured evaluation results.
3. Do NOT invent missing weaknesses, strengths, or comments.
4. Do NOT reinterpret evidence.

### Output Format

Your output format should be a JSON object only:

```json
{
  "summary": "...",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "comments": ["..."],
  "evidence": ["..."]
  "overall_score": <integer from 1-5>
}
```
'''

FINAL_AGGREGATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "strengths": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "weaknesses": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "comments": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "evidence": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 5}
    },
    "required": ["summary", "strengths", "weaknesses", "comments", "evidence", "overall_score"],
    "additionalProperties": False
}


CLAIM_SEGMENTATION_PROMPT = """You are a careful scientific reviewer. Determine whether the target sentence is a claim that should be fact-checked against its single citation.

Input paragraph:
{range}

Target sentence:
{text}

Citation keys in this sentence:
{keys}

Return a JSON object only:
```json
{{
  "is_verifiable_performance_claim": true | false,
  "reason": "short explanation"
}}
```

Mark `true` only if all of the following hold:
- the sentence has exactly one citation,
- the cited work is the direct source of the sentence's meaning,
- the sentence states a factual claim that could be checked from the cited paper.

Mark `false` for background definitions, loose motivation, author opinions, or cases where the citation is just an example.
"""


WEBSEARCH_FILTER_PROMPT = """You are filtering academic web search results for a cited paper lookup.

Target paper title:
{title}

Candidates:
{candidates}

Return a JSON object only:
```json
{{
  "matched_indices": [1, 3],
  "reason": "brief explanation"
}}
```

Rules:
- keep only candidates that are very likely to refer to the same paper,
- prefer publisher, DOI, arXiv, OpenReview, ACL Anthology, Semantic Scholar, DBLP, or author pages,
- it is acceptable to return an empty list if the evidence is weak.
"""


ARGUMENT_EXTRACTION_COMMON_RULES = """
General rules:
- Extract only what is explicitly stated in the source text.
- Failure is acceptable. If the paper does not state the target item clearly, return `null`.
- Do not infer unstated claims.
- Evidence must be copied verbatim from the source text.
- Leave `few_shot_examples` empty if no examples are provided.
"""


CORE_ARGUMENT_PROMPT = """You are extracting the author's core argument from a survey paper.

Source text:
{text}

Few-shot examples:
{few_shot_examples}

""" + ARGUMENT_EXTRACTION_COMMON_RULES + """

Return a JSON object only:
```json
{{
  "item": {{
    "statement": "one-sentence core viewpoint",
    "evidence": ["verbatim quote 1", "verbatim quote 2"]
  }} | null
}}
```

The statement should capture the central viewpoint or thesis of the survey, not just its topic.
"""


MAIN_CONTRIBUTION_PROMPT = """You are extracting the main claimed contributions of a survey paper.

Source text:
{text}

Few-shot examples:
{few_shot_examples}

""" + ARGUMENT_EXTRACTION_COMMON_RULES + """

Return a JSON object only:
```json
{{
  "items": [
    {{
      "statement": "contribution statement",
      "evidence": ["verbatim quote 1"]
    }}
  ]
}}
```

Return an empty list if no explicit contribution claims are stated.
Prefer 1 to 3 major contributions, not minor details.
"""


RESEARCH_GAP_PROMPT = """You are extracting research gaps and future directions from a survey paper.

Source text:
{text}

Few-shot examples:
{few_shot_examples}

""" + ARGUMENT_EXTRACTION_COMMON_RULES + """

Return a JSON object only:
```json
{{
  "research_gaps": [
    {{
      "statement": "research gap",
      "evidence": ["verbatim quote 1"]
    }}
  ],
  "future_directions": [
    {{
      "statement": "future direction",
      "evidence": ["verbatim quote 1"]
    }}
  ]
}}
```

Either list may be empty.
"""
