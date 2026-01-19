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
{{
  "strategy": "Analysis of why previous queries failed and how these queries will broaden the search.",
  "queries": ["query 1", "query 2", "query 3", "query 4"]
}}'''

CLAIM_SEGMENTATION_PROMPT = '''
You are a rigorous scientific fact-checker. Your task is to extract a specific claim from a text segment and classify its requirements for verification.

**Input Context:**

Full Paragraph: """{range}"""

Target Sentence (containing citations): """{text}"""

**Definitions & Examples:**

1. **Claim Type**:
   - **SINGLE_FACTUAL**: A statement regarding a single entity or source.
     *Example:* "BERT utilizes a masked language modeling objective [1]."
   - **SERIAL_FACTUAL**: A list of independent facts or examples, each with its own citation.
     *Example:* "Recent works have applied LLMs to biology [1], chemistry [2], and physics [3]."
   - **SYNTHESIS**: A statement claiming a relationship (comparison, contrast, evolution) between multiple sources.
     *Example:* "While method A [1] focuses on speed, method B [2] improves accuracy."

2. **Verification Requirement (per citation)**:
   - **TITLE_ONLY**: The claim only asserts the existence or general topic of the paper.
     *Example:* "Several surveys have discussed this topic [1][2]." (Checking the title is enough to prove relevance).
   - **TITLE_AND_ABSTRACT**: The claim summarizes the main contribution, high-level method, or primary conclusion.
     *Example:* "Author X proposed a new graph neural network [1]." (Abstract usually states the proposal).
   - **FULL_TEXT**: The claim cites specific experimental results, hyperparameters, mathematical proofs, or minor details not found in an abstract.
     *Example:* "The model achieved 92.5% accuracy on ImageNet [1]." or "They used a learning rate of 1e-4 [1]."

**Task:**
1. Identify the full semantic span of the claim associated with the citations in the Target Sentence (this may encompass parts of the surrounding paragraph).
2. Classify the Claim Type.
3. For EACH citation key found in the claim, determine the Verification Requirement.

**Output:**
Return strictly a JSON object. Do not output markdown code blocks.

{{
    "claim": "The full, self-contained text of the extracted claim",
    "claim_type": "SINGLE_FACTUAL | SERIAL_FACTUAL | SYNTHESIS",
    "requires": {{
        "citation_key_1": "FULL_TEXT | TITLE_AND_ABSTRACT | TITLE_ONLY",
        "citation_key_2": "FULL_TEXT | TITLE_AND_ABSTRACT | TITLE_ONLY"
    }}
}}
'''

FACTUAL_CORRECTNESS_PROMPT = '''
You are a factual correctness verifier for academic surveys. Given:

- A claim extracted from a survey,
- Evidence from a cited paper (title, abstract, retrieved related sentences),

Determine whether the claim is supported by the cited paper. 

Please first explain your judgment in 1–2 sentences, citing what is missing or incorrect. Finally, select and output your judgment from one of $\\boxed{{SUPPORTED}}$, $\\boxed{{REFUTED}}$ or $\\boxed{{NEUTRAL}}$, in a boxed format, where:

- SUPPORTED: the claim is clearly supported by the evidence.
- REFUTED: the claim is clearly contradicted by the evidence.
- NEUTRAL: the claim is not mentioned in the evidence.

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

SYNTHESIS_CORRECTNESS_PROMPT = ''''''

CLARITY_EVAL_PROMPT = '''
You are a senior editor at a top-tier scientific journal (e.g., Nature, NeurIPS). Your task is to score the writing quality of the following target paper segment, specifically focusing on logical flow and clarity.

**Context (Previous Paragraph):**
"""{pre_text}"""

**Input Target (Current Paragraph):**
"""{text}"""

**Scoring Rubric (1-5):**
1. **Unreadable**: Grammatically broken, incoherent, or nonsensical.
2. **Poor**: Major logic gaps. The paragraph does not follow logically from the context. Confusing phrasing.
3. **Average**: Understandable but dry or repetitive. Transitions between sentences or from the previous paragraph are weak or abrupt.
4. **Good**: Clear, logical flow. Good use of transition words. Terminology is correct and professional.
5. **Excellent**: Compelling narrative. The transition from the context is seamless. The argument builds perfectly within the paragraph. Concise and persuasive.

**Instructions:**
1. **Check Transitions:** If the "Context" is NOT "This is the first paragraph of the paper", evaluate how well the "Input Target" flows from it. Does the topic shift abruptly?
2. **Check Internal Flow:** Within the "Input Target", do the sentences progress logically?
3. **Check Style:** Is the tone scientific? Is there unnecessary redundancy?

**Output:**
{{
    "score": <Integer 1-5,
    "reason": ""<Specific critique. Mention if the transition from the previous paragraph was smooth or abrupt."
}}
'''

FINAL_AGGREGATION_PROMPT = ''''''
