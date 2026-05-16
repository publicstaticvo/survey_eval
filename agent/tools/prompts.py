# get_reference_surveys.py
REFERENCE_SURVEY_SELECT = """You are a professional academic researcher selecting reference surveys to evaluate a target survey titled "{query}".

### Goal
Select surveys that serve as FIELD-LEVEL structural references — surveys that cover the same broad scope as the query, organized around the same primary subject.

### Input Format
Each candidate is provided with a pre-extracted scope declaration containing four fields:
- title: the title of the candidate
- section_map: maps section numbers to their topics
- aspect_list: dimensions or aspects the survey explicitly covers
- evidence_records: verbatim sentences from the paper describing its scope

### Candidate Scope Declarations
{candidates}

Use these fields as the sole basis for judging topic coverage. Do not infer topics beyond what is stated in these fields.

### Inclusion Criteria
A selected survey must satisfy ALL of the following:
- Its PRIMARY SUBJECT matches the query topic directly, not as a subordinate method or tool applied within a different domain.
- At least 3 items from its aspect_list or section_map values fall within the query field.
- It synthesizes and organizes the literature rather than reporting original experimental results, as indicated by its evidence.

### Exclusion Criteria
Exclude a paper if ANY of the following apply:
- Its primary subject is a specific downstream domain, and the query topic appears only as the method used within that domain.
- Fewer than three items in its aspect_list or section_map values belong to the query field, regardless of total item count.
- The query topic is mentioned as background or one method among many, but is not the organizing principle of the survey.
- It is not a survey: excludes tutorials, position papers, benchmarks, or original research papers.

### Required Self-Check (apply to each candidate before deciding)
Q1. What is the primary subject of this survey, based on its evidence    sentences? State it in one sentence.
Q2. Does that primary subject directly match the query topic? Or is the query topic a tool or method applied within a different primary subject?
Q3. From the provided aspect_list and section_map, list only the items that belong to the query field. Count them.

If Q2 = "tool within different subject" → EXCLUDE.
If Q3 count < 3 → EXCLUDE.

### Output Format
Return JSON only, no extra text:
```json
{{
  "surveys": [
    {{
      "title": "Exact title from candidate list",
      "primary_subject": "One phrase: what this survey is fundamentally about",
      "subtopics_covered": [
        "all items from aspect_list or section_map that belong to the query field"
      ]
    }}
  ]
}}
```

Return `{{"surveys": []}}` if no candidate clearly qualifies.
"""

REFERENCE_SURVEY_SCHEMA = {
    "type": "object",
    "required": ["surveys"],
    "properties": {
        "surveys": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "primary_subject", "subtopics_covered"],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "primary_subject": {"type": "string", "minLength": 1},
                    "subtopics_covered": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

# golden_topics.py
TOPIC_CLUSTER_PROMPT = """You are a Senior Research Librarian specializing in Systematic Literature Reviews. You are given a set of reference surveys, each with an ID, title, and a filtered list of section headings. Your task is to identify the core research topics covered across these surveys by clustering semantically related headings.

### Input format
[
  {
    "survey_id": "<id>",
    "survey_title": "<title>",
    "section_headings": ["<heading1>", "<heading2>", ...]
  },
  ...
]

### Instructions

- Group semantically equivalent or closely related non-generic headings into topics.
- A heading is generic if it does not name a specific research area, method, task, or concept (e.g. "Overview", "Summary", "Preliminaries", "Notation", "Conclusion", "Future Work", "Discussion", "Limitations", "Appendix"). Exclude such headings even if they passed the pre-filter.
- Each topic must represent a coherent, specific research concept, task, method family, or application area.
- Every topic must include one or more sources as evidence.
- Each source must copy the original section title CHARACTER FOR CHARACTER exactly as given in the input. Do not paraphrase, normalize, or correct the original wording.
- Do not invent section titles that do not appear in the input.
- Do not split a single heading across multiple topics unless it clearly names two distinct concepts.
- A heading may appear in at most one topic. Choose the most specific topic it belongs to.
- Prefer concise, human-readable topic names in English noun phrase form (e.g. "Knowledge Distillation", "Low-Resource Machine Translation", "Evaluation Benchmarks").
- Topics that appear in only one survey and have no semantically related counterpart in any other survey may be included if the heading is specific and substantive.
- Return an empty topics list if the headings are insufficient to form any meaningful topic.

### Output Format

Output a single JSON object with no additional commentary, explanation, or markdown formatting:

```json
{{
  "topics": [
    {{
      "topic": "topic_name",
      "sources": [
        {{"survey_id": "S1", "section_title": "exact original heading"}},
        ...
      ]
    }}
  ]
}}
```
"""

TOPIC_CLUSTER_SCHEMA = {
    "type": "object",
    "required": ['topics'],
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ['topic', 'sources'],
                "properties": {
                    "topic": {"type": "string", "minLength": 1},
                    "sources": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["survey_id", 'section_title'],
                            "properties": {
                                "survey_id": {"type": "string", "pattern": r"^S\d+$"},
                                "section_title": {"type": "string", "minLength": 1}
                            },
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}

# self_evidence.py
SCOPE_CLAIM_EXTRACT = """You are a Senior Research Librarian specializing in Systematic Literature Reviews. You are given some paragraphs from a literature review.

### Input
{text}

### Task
Extract sentences that explicitly describe the paper's section-by-section organization or covered aspects. Follow these rules:

1. section_map: For each sentence that maps a section number to a topic (e.g., "Section 2 discusses X"), add an entry where:
   - key: the full section number as a string, preserving all levels (e.g., "2", "3.1", "4.2.1")
   - value: the topic or aspect described for that section

2. aspect_list: If the text describes covered aspects or topics WITHOUT section numbers (e.g., "this survey covers A, B, and C"), list each aspect as a separate string.

3. Do NOT extract generic paper-organization sections as topics. Exclude items whose value is only or mainly: introduction, background, preliminary/preliminaries, related work, methods/methodology, experiments/evaluation/results, discussion, conclusion, future work/future directions, limitations, open problems/open questions, appendix, references, acknowledgments.

4. evidence: Copy the source sentences that support the above extractions. Use the exact original wording; escape any internal quotation marks with a backslash.

5. If the text contains no organizational or scope statements, return empty objects.

### Positive examples
- "Section 3 reviews parameter-efficient fine-tuning methods." -> {{"3": "parameter-efficient fine-tuning methods"}}
- "We cover retrieval-augmented generation, tool use, and agent evaluation." -> ["retrieval-augmented generation", "tool use", "agent evaluation"]

### Negative examples
- "Section 2 introduces background, Section 7 discusses open challenges, and Section 8 concludes the paper." -> do not extract Section 2, Section 7, or Section 8.
- "The paper is organized as follows: Section 1 is the introduction and Section 6 is the conclusion." -> return empty section_map and aspect_list.

### Output Format
Return valid JSON only, no other text:

```json
{{
  "section_map": {{"2": "topic", "3.1": "topic"}},
  "aspect_list": ["aspect"],
  "evidence": ["verbatim sentence"]
}}
```
"""

SCOPE_CLAIM_SCHEMA = {
    "type": "object",
    "required": ["section_map", "aspect_list", "evidence"],
    "properties": {
        "section_map": {
            "type": "object",
            "additionalProperties": {"type": "string", "minLength": 1},
        },
        "aspect_list": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
    "additionalProperties": False,
}

# claim_segmentation.py
CLAIM_CLASSIFICATION_PROMPT = """You are a careful scientific reviewer. Determine whether the target sentence is a claim that should be fact-checked against its single citation.

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

# websearch.py
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

# fact_check.py
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

# structure_eval.py
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

# argument_eval.py
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

# aggregate_review.py
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
