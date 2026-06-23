# preprocess/get_reference_surveys.py
REFERENCE_SURVEY_SELECT = """You are a professional academic researcher selecting reference surveys to evaluate a target survey titled "{query}".

### Input
You are given a list of candidate papers with their titles and abstracts.

### Definition: Reference Survey (Two Tiers)
A **strict reference survey** is a field-level survey that:
- Treats the query topic as its PRIMARY organizing focus, not as a tool or method applied within a different domain.
- Organizes the literature into coherent conceptual or methodological dimensions (e.g., taxonomies, categorizations, design spaces).
- Covers multiple distinct sub-topics within the query field.
- Would be consulted by an expert to judge whether another survey on this topic has missed important topics or references.

A **partial reference survey** is a survey that:
- Treats the query topic as one of its main research objects, even if the organizing principle is narrower or domain-specific.
- Covers at least TWO distinct sub-topics within the query field.
- Synthesizes existing literature on the query topic within its scope, rather than merely using the query topic as a method.
- Would be consulted by an expert for the sub-area it covers, but not necessarily as a comprehensive reference for the entire field.

### Inclusion Criteria for Strict Reference Surveys (ALL must be satisfied)
1. The query topic is the main research object AND the primary organizing principle of the paper.
2. At least THREE distinct sub-topics within the query field are covered.
3. The paper synthesizes existing literature rather than reporting original experimental results.

### Inclusion Criteria for Partial Reference Surveys (ALL must be satisfied)
1. The query topic is one of the main research objects, even if the scope is narrower or the organizing principle is domain-specific.
2. At least TWO distinct sub-topics within the query field are covered.
3. The paper synthesizes existing literature on the query topic within its scope.

### Exclusion Criteria (ANY triggers exclusion from BOTH tiers)
- Not a survey: excludes benchmarks, position papers, tutorials, or original research papers.
- Primary subject is a downstream application domain, with the query topic appearing only as the method used (e.g., "Transformers for Medical Imaging" is excluded when evaluating a survey on Transformers).
- Covers only ONE task or sub-area within the query field.
- Mentions the query topic only as background or one method among many.

### Classification Instructions
- First apply exclusion criteria. If any exclusion criterion is met, discard the candidate entirely.
- For remaining candidates, determine tier:
  - If ALL strict inclusion criteria are satisfied → strict reference survey.
  - If strict criteria are not fully met but ALL partial inclusion criteria are satisfied → partial reference survey.
  - Otherwise → discard.
- Select at most 3 surveys in total. Prefer strict reference surveys over partial reference surveys.
- Be conservative: fewer is better than including a marginal candidate.
- If no candidate meets even partial reference survey criteria, return an empty list.

### Candidate Surveys
{candidates}

### Output Format
Return JSON only, no extra text.
{{
  "strict_reference_surveys": [
    {{
      "title": "...",
      "covered_subtopics": ["subtopic1", "subtopic2", "subtopic3"],
      "reason": "one sentence explaining why this qualifies as a strict reference survey"
    }}
  ],
  "partial_reference_surveys": [
    {{
      "title": "...",
      "covered_subtopics": ["subtopic1", "subtopic2"],
      "reason": "one sentence explaining why this only qualifies as a partial reference survey"
    }}
  ]
}}
"""

REFERENCE_SURVEY_SCHEMA = {
  "type": "object",
  "properties": {
    "strict_reference_surveys": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string", "minLength": 1},
          "covered_subtopics": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 3},
          "reason": {"type": "string", "minLength": 1}
        },
        "required": ["title", "covered_subtopics", "reason"],
        "additionalProperties": False
      },
      "maxItems": 3,
    },
    "partial_reference_surveys": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string", "minLength": 1},
          "covered_subtopics": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 2},
          "reason": {"type": "string", "minLength": 1}
        },
        "required": ["title", "covered_subtopics", "reason"],
        "additionalProperties": False
      },
      "maxItems": 3,
    }
  },
  "required": ["strict_reference_surveys", "partial_reference_surveys"],
  "additionalProperties": False
}

# query_expand
QUERY_EXPAND = """You are a professional academic researcher selecting reference surveys to evaluate a target survey titled "{survey_title}".

## Task
Generate ONE search query to find papers that are relevant to a specific section of a survey but may be missing from its references.

## Input
- Survey title: {survey_title}
- Section title: {section_title}
- Key sentences describing the main topics discussed in this section:
{summary_sentences}

## Requirements
1. The query MUST contain core domain terms from the survey title.
2. The query MUST contain specific technical concepts grounded in the key sentences. Do NOT invent concepts not present in the sentences.
3. Total query length: 3-5 terms.
4. Do NOT use question structures, conjunctions (AND/OR), quotation marks, or rhetorical phrases ("the issue of", "how to", "a study on").

## Output format (JSON only)
{{
  "query": "..."
}}

## Example
Survey title: A Survey on Causal Reinforcement Learning
Section title: The Issue of Generalizability in Reinforcement Learning
Key sentences:
- Domain randomization uniformly samples simulation parameters to reduce the sim-to-real gap.
- Invariant Risk Minimization learns representations that remain stable across training environments.
- Meta-learning approaches adapt policies to unseen environments using few-shot experience.

Output:
{{
  "query": "causal reinforcement learning out-of-distribution generalization"
}}
"""

# preprocess/sentences.py
RULES = """### Labels and definitions (apply Rule 1 first; then use the first applicable rule among 2–8)

1. CONTRIBUTION — The current paper is the subject: an explicit first-person or self-referential subject (we, our, this survey/paper) paired with an action verb (propose, define, categorize, present, introduce, show, demonstrate, categorize). If the subject is a named prior system, model, or method, classify as SUMMARY or EVALUATION instead. Applied before all other rules regardless of content.

2. LIMITATION - Declares what this survey does not cover and why: explicit scope exclusions, acknowledged gaps, or methodological constraints of the review itself. Does not summarize contributions (CONCLUSION) and does not identify field-level open problems (GAP).

3. GAP — Identifies an unresolved problem, open challenge, or missing capability in the research field, typically signaling a direction for future work. Does not include motivational framing that merely justifies the current survey.

4. EVALUATION — The author makes an explicit positive or negative judgment about a specific named prior work, using evaluative language (outperforms, suffers from, is limited by, fails to, effectively handles). The judgment must reflect the author's own stance, not a description of a result or consequence. The current survey must not be one of the evaluated works. Sentences that only report what a prior paper found, demonstrated, or showed — without the survey author adding an evaluative stance — are SUMMARY, not EVALUATION. Examples:
  - EVALUATION: "X is more effective than Y for Z tasks"  (author judgment) 
  - SUMMARY:    "X demonstrated 95% accuracy on dataset Y" (result report with numerical data)
  - SUMMARY:    "Experiments show that X outperforms Y"    (reporting prior work's finding)

5. COMPARISON — Explicitly contrasts two or more specific named prior works (methods, models, or systems) using explicit contrast markers (unlike, in contrast, whereas, compared to) or quantitative side-by-side metrics. The current survey must not be one of the contrasted works.

6. SYNTHESIS — Organizes multiple prior works into categories, trends, or abstractions. Subject is implicit or third-person (studies, methods, approaches, researchers); if the subject is first-person, apply Rule 1 instead.

7. SUMMARY — Describes one or more specific named prior works without organizing, judging, or contrasting them.

8. BACKGROUND — General field context, definitions, or facts. May mention concept names but does not refer to specific works by author or title."""

SENTENCE_CLASSIFICATION_SINGLE = """You are an expert annotator for rhetorical structure in academic literature reviews.

### Task
Classify the rhetorical function of the given sentence from the literature review. Choose exactly ONE label.

### Sentence
"{S}"

{RULE}

### Output format (JSON only)
{{
  "label": "...",
  "confidence": 0-1
}}"""

SENTENCE_CLASSIFICATION_PARAGRAPH = """You are an expert annotator for rhetorical structure in academic literature reviews.

### Task
Given a paragraph from the literature review, each line representing a sentence. Classify the rhetorical function of EACH sentence. Process sentences independently, but use paragraph context if needed.

### Input
{P}

{RULE}

### Instructions
- Each input line is exactly one sentence.
- Do NOT split or merge sentences.
- Assign exactly ONE label per sentence. For this paragraph, you should return {length} labels.
- The number of outputs must equal the number of input lines.

### Output format (JSON only)
{{
  "results": [
    {{
      "label": "...",
      "confidence": 0-1
    }},
    {{
      "label": "...",
      "confidence": 0-1
    }}
  ]
}}
"""

SENTENCE_LABELS = {"CONTRIBUTION", "LIMITATION", "GAP", "EVALUATION", "COMPARISON", "SYNTHESIS", "SUMMARY", "BACKGROUND"}

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": sorted(SENTENCE_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "confidence"],
    "additionalProperties": True,
}

PARAGRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": SINGLE_SCHEMA,
        },
    },
    "required": ["results"],
    "additionalProperties": True,
}

# preprocess/section.py
SECTION_CLASSIFICATION = """You are an expert annotator for the structure of academic literature review papers.

### Task
Given a section or subsection from a literature review, assign:
  (1) Exactly ONE functional type describing its rhetorical role.
  (2) One or more content tags describing the topics it covers.

### Input
Document title   : "{DOCUMENT_TITLE}"
Parent section   : "{PARENT_TITLE}"   (empty if top-level)
Section title    : "{SECTION_TITLE}"
Opening text     : "{PREAMBLE}"       (first 1–3 sentences; may be empty)

### Functional Type Definitions (choose exactly ONE)

- SCOPE
  * Definition: The section states the survey's objectives, contributions, scope, or organization. Typically the first section. May contain an outline of the rest of the paper.
  * Typical titles: Introduction, Overview, Motivation, Contribution, Scope of this survey.

- BACKGROUND
  * Definition: The section provides prerequisite knowledge required to understand the rest of the survey: formal definitions, notation, core concepts, theory, or historical development that are not the survey's primary contribution.
  * Typical titles: Background, Preliminaries, Foundations, Notation.

- POSITIONING
  * Definition: The section situates this survey relative to existing surveys or closely related work. Its primary purpose is to explain what distinguishes this survey from prior surveys on the same or adjacent topics.
  * Typical titles: Related Surveys, Related Work, Comparison with Existing Surveys.

- TAXONOMY
  * Definition: The section proposes, explains, or justifies a classification framework, categorization scheme, or organizing criteria that structures the rest of the survey. Look for explicit statements such as "we categorize … into", "we organize … according to", or a diagram/table that defines the taxonomy.
  * Typical titles: Taxonomy, Categorization, Problem Formulation, A Taxonomy of X.

- CONTENT
  * Definition: The section reviews specific methods, approaches, systems, or sub-areas following the survey's organizational scheme. This is the default type for body sections that do not match any other type.
  * Typical titles: domain- or method-specific titles.

- EVALUATION
  * Definition: The section systematically compares methods or systems using benchmarks, quantitative metrics, experimental results, or performance tables. The primary purpose is comparative analysis rather than description.
  * Typical titles: Evaluation, Experiments, Benchmark Comparison, Performance Analysis.

- LIMITATION
  * Definition: Declares what this survey does not cover and why: explicit scope exclusions, acknowledged gaps, or methodological constraints of the review itself. Does not summarize contributions (CONCLUSION) and does not identify field-level open problems (FUTURE_WORK).
  * Typical titles: Limitations, Limitations of This Survey, Out of Scope, Exclusion Criteria, What This Survey Does Not Cover, Threats to Validity.

- FUTURE_WORK
  * Definition: Identifies open problems, unsolved challenges, or directions for the research community. Includes perspective statements ("we believe", "we envision") when they appear as section-level content rather than isolated sentences.
  * Typical titles: Future Work, Open Problems, Challenges, Open Issues, Outlook, Key Design Issues, Research Challenges, Opportunities.

- CONCLUSION
  * Definition: The section summarizes the survey's main findings and contributions. Does not introduce new content or open questions (those belong to PROSPECTIVE).
  * Typical titles: Conclusion, Summary, Concluding Remarks.

### Content Tag Definitions (choose ALL that apply)

METHOD — Covers specific algorithms, architectures, models, or technical approaches.
DATASET — Describes datasets, corpora, or data collection/annotation procedures.
BENCHMARK — Discusses evaluation benchmarks, leaderboards, standard test sets, or evaluation metrics/protocols.
ETHICS_AND_SAFETY  — Addresses ethical considerations, fairness, bias, discrimination, privacy, safety, robustness, reliability, or adversarial vulnerabilities.
TOOLKIT — Covers software libraries, open-source frameworks, toolkits, or code repositories.
APPLICATION - Covers real-world deployment, industrial use cases, or scenario-based selection guidance for reviewed systems.
GENERAL — FALLBACK ONLY. Assign this tag if and only if none of the tags above (METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION) clearly applies to this section. NEVER assign GENERAL together with any other tag. If you are uncertain whether a specific tag fits but it is the best available option, assign that specific tag without GENERAL.

### Decision Notes
- If the opening text is empty, base your decision on the section title and document title alone.
- Assign all content tags that clearly apply. If you assign any tag other than GENERAL, do not add GENERAL. GENERAL is only valid as the sole tag when no other tag fits at all.
- When the section title is ambiguous (e.g., "Discussion"), use the opening text to decide between FUTURE_WORK and CONCLUSION.
- A subsection inherits no constraints from its parent section type; classify it solely on its own title and opening text.
- "Discussion" titles: inspect opening text — field-level open problems → FUTURE_WORK; summary of findings → CONCLUSION; scope exclusions → LIMITATION. 
- LIMITATION is often a short subsection inside Introduction or Conclusion; position does not override content. Decisive signal: "we do not discuss", "out of scope", "we exclude". 
- Perspective sentences ("we believe", "in our view") within any section do not change that section's functional type; they are sentence-level phenomena.

### Conflict resolution: preamble background vs. section title 
Many survey body sections open with 1–3 sentences of motivating background before surveying specific works. If the section title names a substantive technical area (methods, systems, tools, hardware), classify as CONTENT even when the opening sentences are background in nature. The preamble background belongs to the section but does not determine its functional type. 

Apply BACKGROUND only when the entire purpose of the section is to provide prerequisite knowledge — not when it is a survey body section that happens to start with motivation.

### Hard constraints on output
- content_tags must contain either GENERAL alone, or one or more tags from {{METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION}}. Any output combining GENERAL with another tag is invalid.

### Output format (JSON only)
{{
  "functional_type": "...",
  "content_tags": ["...", "..."],
  "confidence": 0.0
}}"""

SECTION_LABELS = {'SCOPE', 'BACKGROUND', 'POSITIONING', 'TAXONOMY', 'CONTENT', 'EVALUATION', 'FUTURE_WORK', 'LIMITATION', 'CONCLUSION'}

CONTENT_TAGS = {'METHOD', 'DATASET', 'BENCHMARK', 'ETHICS_AND_SAFETY', 'TOOLKIT', 'APPLICATION', 'GENERAL'}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "functional_type": {"enum": sorted(SECTION_LABELS)},
        "content_tags": {"type": "array", "items": {"enum": sorted(CONTENT_TAGS)}, 'minItems': 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["functional_type", 'content_tags', "confidence"],
    "additionalProperties": True,
}

# preprocess/contribution_classify.py
CONTRIBUTION_CLASSIFICATION = """You are an expert annotator for academic literature review papers.

### Task
Given a sentence from the Abstract or Introduction of a survey paper, first determine if it should be excluded, then extract all verifiable *self-limiting claims* it contains. A self-limiting claim is a statement where the survey promises to cover a topic, perform a type of analysis, or contain a specific type of content. One sentence may yield multiple claims.

### Input
Sentence : "{S}"
Context  : "{CONTEXT}"

### Step 1 — Exclusion check

- Return {"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE"} if: The grammatical subject is a named prior system, paper, model, dataset, or method — not the current survey. Example triggers: "[Named System] is proposed / introduced / presented / designed / consists of"

- Return {"excluded": true, "reason": "NON_VERIFIABLE"} if any of:
  * Pure intent or hope with no content claim: "We hope this work will inspire..."
  * Bare list header: "Our contributions are:", "This paper:"
  * Generic quality claim with no named topic: "a thorough and comprehensive survey" (alone, no topic)
  * Document-level organization without content: "This paper is organized as follows."

  Note: "Section 2 introduces X" is NOT non-verifiable. It claims Section 2 covers topic X and must be extracted.

If neither exclusion applies, proceed to Step 2.

### Step 2 — Claim extraction

For each claim, produce one entry with three fields: section, type, target.

─────────────────────────────────────────────────────────────
section (string)
  Scope of this specific claim.
  "document"       — claim applies to the whole paper
  "Section 2"      — use exact number if stated
  "Figure 1"       — claim applies to a specific figure
  "Table 1"       — claim applies to a specific table
  "Related Work"   — use section title if named but no number
  "this section"   — if text says "in this section" without specifying
─────────────────────────────────────────────────────────────
type (string, choose exactly one from the list below)
─────────────────────────────────────────────────────────────
target (string)
  Concise description of what to look for when verifying.
  Always required; provide even when type already names the label.
─────────────────────────────────────────────────────────────

### Type definitions

- sentence:COMPARISON
  * Definition: The claim is verified by finding COMPARISON sentences (explicit cross-work contrasts using "unlike", "in contrast", "compared to") or multi-dimensional comparison tables in the body.
  * Use when: "comparative study", "we compare X and Y along dimensions", "Table N contrasts systems across criteria".

- sentence:SYNTHESIS
  * Definition: Verified by finding SYNTHESIS sentences that organize multiple works into categories, trends, or unified abstractions (e.g., "these methods fall into three families...").
  * Use when: "we synthesize findings across approaches", "methods are unified under a common framework", "we propose a taxonomy of X".

- section:EVALUATION
  * Definition: Verified by finding a section with functional type EVALUATION — a dedicated section for systematic comparison of systems using benchmarks, metrics, or tables.
  * Use when: "a dedicated evaluation section is provided", "Section N presents a benchmark comparison of systems", "performance results are summarized in Table N".

- section:FUTURE_WORK
  * Definition: Verified by finding a section with functional type FUTURE_WORK - a section that identify unresolved problems, missing work, or open questions.
  * Use when: "open challenges are discussed", "future research directions are identified", "open problems are studied", "we highlight unsolved issues in Section X".

- tag:METHOD
  * Definition: Verified by finding a section with content tag METHOD, covering algorithm classes, model architectures, or technical approaches as a topic in themselves.
  * Use when: "we survey optimization methods", "deep learning approaches are reviewed", "we cover model compression techniques". For highly specific named domains → prefer coverage.

- tag:DATASET
  * Definition: Verified by finding a section with content_tag DATASET.
  * Use when: "datasets are surveyed", "existing corpora are reviewed", "data collection and annotation methods are discussed".

- tag:BENCHMARK
  * Definition: Verified by finding a section with content_tag BENCHMARK. This type is for reviewing benchmark protocols and evaluation metrics as a topic, not for comparing systems on benchmarks.
  * Use when: "evaluation benchmarks are reviewed", "we survey existing metrics and their limitations", "benchmark datasets are categorized". Do NOT use when the claim is about comparing systems → section:EVALUATION.

- tag:ETHICS_AND_SAFETY
  * Definition: Verified by finding a section with content_tag ETHICS_AND_SAFETY.
  * Use when: "ethical considerations are discussed", "fairness and bias are addressed", "privacy implications are analyzed", "societal impacts are examined", "safety of systems is discussed", "robustness to adversarial inputs is reviewed", "security vulnerabilities are analyzed", "reliability is addressed".

- tag:TOOLKIT
  * Definition: Verified by finding a section with content_tag TOOLKIT.
  * Use when: "open-source tools are surveyed", "software libraries are reviewed", "available frameworks are compared", "code repositories are discussed".

- tag:APPLICATION
  * Definition: Verified by finding a section with content_tag APPLICATION.
  * Use when: "real-world applications are discussed", "industrial deployment scenarios are covered", "use cases are analyzed", "practical guidance is provided".

- coverage
  * Definition: Verified by finding a section whose title is semantically similar to the stated topic. Use for domain-specific topics that do not map to any tag:* type above.
  * Use when: "we discuss graph neural networks for drug discovery", "federated learning over heterogeneous networks is reviewed". Do NOT use when a tag:* type fits.

### Disambiguation rules

- tag:BENCHMARK vs section:EVALUATION
  * Reviewing what benchmarks exist and their properties → tag:BENCHMARK
  * Using benchmarks to compare systems against each other → section:EVALUATION

- tag:METHOD vs coverage
  * General method class without named domain → tag:METHOD
  * Specific named sub-domain or cross-domain application → coverage

### Multi-claim handling

Extract each claim as a separate entry. One sentence may yield multiple entries with different sections, types, or targets.

Example:
  Input: "Section 2 introduces edge computing systems, Section 3 benchmarks performance across systems, Section 4 discusses open research challenges."
  Output:
  ```
  {
    "claims": [
      {"section": "Section 2", "type": "coverage", "target": "edge computing systems"},
      {"section": "Section 3", "type": "section:EVALUATION", "target": "systematic benchmark comparison of edge systems"},
      {"section": "Section 4", "type": "section:FUTURE_WORK", "target": "open research challenges"}
    ]
  }
  ```

Example:
  Input: "We provide a comprehensive overview of privacy-preserving methods and a comparative analysis of their computational overhead."
  Output:
  {
    "claims": [
      {"section": "document", "type": "tag:ETHICS", "target": "privacy-preserving methods"},
      {"section": "document", "type": "sentence:COMPARISON", "target": "computational overhead comparison across privacy-preserving approaches"}
    ]
  }

### Output format (JSON only)

If excluded:
{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE | NON_VERIFIABLE"}

If not excluded:
{
  "excluded": false,
  "claims": [
    {
      "section": "...",
      "type": "...",
      "target": "..."
    }
  ]
}
"""

CONTRIBUTION_LABELS = {"sentence:COMPARISON", "sentence:SYNTHESIS", "section:EVALUATION", "section:FUTURE_WORK", "tag:METHOD", "tag:DATASET", "tag:BENCHMARK", "tag:ETHICS_AND_SAFETY", "tag:TOOLKIT", "tag:APPLICATION", "coverage"}

# preprocess/claim_segmentation.py
CLAIM_SEGMENTATION = """"""

CLAIM_CLASSIFICATION = """You are a careful scientific reviewer. Determine whether the target sentence is a claim that should be fact-checked against its single citation.

### Input paragraph
{range}

### Target sentence
{text}

### Citation keys in this sentence
{keys}

### Output format
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


def CONTRIBUTION_SCHEMA(sections):
    return {
      "type": "object",
      "oneOf": [
        {
          "properties": {
            "excluded": {"const": True},
            "reason": {"enum": ["PRIOR_WORK_FALSE_POSITIVE", "NON_VERIFIABLE"]}
          },
          "required": ["excluded", "reason"],
          "additionalProperties": False
        },
        {
          "properties": {
            "excluded": {"const": False},
            "claims": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "section": {"enum": sorted(sections)},
                  "type": {"enum": sorted(CONTRIBUTION_LABELS)},
                  "target": {"type": "string", "minLength": 1}
                },
                "required": ["section", "type", "target"],
                "additionalProperties": False
              }
            }
          },
          "required": ["excluded", "claims"],
          "additionalProperties": False
        }
      ]
    }


# preprocess/websearch.py
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

# preprocess/find_entities.py
FIND_ALL_ENTITIES = """### Task
Extract all tokens or short phrases in the sentence that may be names of specific technical artifacts. Base extraction on surface form only — do not rely on whether you recognize the term.

EXTRACT tokens matching any rule:
1. CamelCase: uppercase letter appears after a lowercase letter (GraphNarrator, RoBERTa, EcomScriptBench, SciBERT)
2. All-caps or alphanumeric artifact token ≥ 2 chars (BERT, BLEU, RST, GPT-4, T5, LLAMA-3, ELABORATION)
3. Mid-sentence word with unexpected initial uppercase — not a personal name, language, country, day, or month name, and not Figure / Table / Section / Equation / Appendix / Algorithm

SKIP sentence-initial words unless they also match rule 1 or 2.

### Output format (Return JSON only)
```json
{{"artifacts": [...]}}
```

### Examples

Input: "GraphNarrator produces graph-to-text outputs on WebNLG, outperforming BART and T5."
Output: {"artifacts": ["GraphNarrator", "WebNLG", "BART", "T5"]}

Input: "We fine-tune on EcomScriptBench using Adam and report ROUGE-L."
Output: {"artifacts": ["EcomScriptBench", "Adam", "ROUGE-L"]}

Input: "Modern language models perform well on English reading comprehension."
Output: {"artifacts": []}

### Input sentence
Input: \"{sentence}\"
Output: """

# fact/fact_check.py
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

# contribution/internal_consistent.py
INTERNAL_CONSISTENT = """You are an expert annotator for academic literature review papers.

### Task
You are given a survey section. Your task is to identify items whose topic is CLEARLY DIFFERENT from the topic promised by the section title and opening text. 

### Input

Section title : "{SECTION_TITLE}"

Content sentences (pre-classified; background sentences excluded):
{SENTENCE_LIST}
Format: S1: Sentence one. / S2: Sentence two. / ...

Subsection titles (if any):
{SUBSECTION_LIST}
Format: X.1 Subsection one / X.2 Subsection two / ...

### Step 1 — State the promised topic

In one sentence, state the specific topic this section promises to discuss, based solely on the title and opening text.
Be concrete: "privacy protection mechanisms in edge computing", not "privacy".

### Step 2 — Label content sentences

For each content sentence, decide: does this sentence introduce a topic that is CLEARLY DIFFERENT from the promised topic?

A sentence is DIFFERENT if:
  — It describes a specific system, method, or finding whose subject matter has no connection to the promised topic.
  — The sentence is a substantive claim about something else, not a transition, motivation, or framing statement.

A sentence is NOT DIFFERENT if:
  — It motivates, introduces, or provides context for the promised topic (even if it mentions other areas by contrast).
  — It is a comparison or contrast where the main subject is still the promised topic: "Unlike method X, our reviewed approach Y..."
  — It is a transition between sub-topics within the promised topic.

Output: list of sentence indices (e.g. S1, S3) that are DIFFERENT. If none, output an empty list.

### Step 3 — Label subsection titles

For each subsection title, decide: does this subsection title clearly indicate a DIFFERENT topic from the promised topic?

A subsection title is DIFFERENT if:
  — It names a subject that has no plausible connection to the promised topic, even as a sub-component or related aspect.
  — Example: promised topic is "privacy protection", subsection title is "Cache Optimization" → DIFFERENT.
  — Example: promised topic is "privacy protection", subsection title is "Differential Privacy Mechanisms" → SAME.
  — Example: promised topic is "inference optimization", subsection title is "Security Considerations" → DIFFERENT.

A subsection title is NOT DIFFERENT if:
  — It names a component, technique, or sub-area that naturally belongs to the promised topic.
  — It provides a finer-grained categorization of the promised topic.
  - It refers to a specific dataset, benchmark, method name or research entity. A specific name or research entity usually does not follow the spelling rules of English words, such as not a english word, or capitalizing non-initial letters.
  - It is a general section name (background, method, evaluation, limitations, future works, conclusions, ...)

Output: list of subsection IDs (e.g. X.2, X.5) that are DIFFERENT. If none, output an empty list.

### Output format (JSON only)
```json
{
  "promised_topic": "...",
  "different_sentences": ["S1", "S3"]
  "different_subsections": ["X.2", "X.5"],
}
```

Do not include any explanation outside the JSON.
"""

# topic_coverage.py
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
  * reference survey paper roles
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
