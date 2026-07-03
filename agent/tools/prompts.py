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
  - If ALL strict inclusion criteria are satisfied 鈫?strict reference survey.
  - If strict criteria are not fully met but ALL partial inclusion criteria are satisfied 鈫?partial reference survey.
  - Otherwise 鈫?discard.
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
RULES = """### Labels and definitions (apply Rule 1 first; then use the first applicable rule among 2鈥?)

1. CONTRIBUTION 鈥?The current paper is the subject: an explicit first-person or self-referential subject (we, our, this survey/paper) paired with an action verb (propose, define, categorize, present, introduce, show, demonstrate, categorize). If the subject is a named prior system, model, or method, classify as SUMMARY or EVALUATION instead. Applied before all other rules regardless of content.

2. LIMITATION - Declares what this survey does not cover and why: explicit scope exclusions, acknowledged gaps, or methodological constraints of the review itself. Does not summarize contributions (CONCLUSION) and does not identify field-level open problems (GAP).

3. GAP 鈥?Identifies an unresolved problem, open challenge, or missing capability in the research field, typically signaling a direction for future work. Does not include motivational framing that merely justifies the current survey.

4. EVALUATION 鈥?The author makes an explicit positive or negative judgment about a specific named prior work, using evaluative language (outperforms, suffers from, is limited by, fails to, effectively handles). The judgment must reflect the author's own stance, not a description of a result or consequence. The current survey must not be one of the evaluated works. Sentences that only report what a prior paper found, demonstrated, or showed 鈥?without the survey author adding an evaluative stance 鈥?are SUMMARY, not EVALUATION. Examples:
  - EVALUATION: "X is more effective than Y for Z tasks"  (author judgment) 
  - SUMMARY:    "X demonstrated 95% accuracy on dataset Y" (result report with numerical data)
  - SUMMARY:    "Experiments show that X outperforms Y"    (reporting prior work's finding)

5. COMPARISON 鈥?Explicitly contrasts two or more specific named prior works (methods, models, or systems) using explicit contrast markers (unlike, in contrast, whereas, compared to) or quantitative side-by-side metrics. The current survey must not be one of the contrasted works.

6. SYNTHESIS 鈥?Organizes multiple prior works into categories, trends, or abstractions. Subject is implicit or third-person (studies, methods, approaches, researchers); if the subject is first-person, apply Rule 1 instead.

7. SUMMARY 鈥?Describes one or more specific named prior works without organizing, judging, or contrasting them.

8. BACKGROUND 鈥?General field context, definitions, or facts. May mention concept names but does not refer to specific works by author or title."""

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
Opening text     : "{PREAMBLE}"       (first 1鈥? sentences; may be empty)

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
  * Definition: The section proposes, explains, or justifies a classification framework, categorization scheme, or organizing criteria that structures the rest of the survey. Look for explicit statements such as "we categorize 鈥?into", "we organize 鈥?according to", or a diagram/table that defines the taxonomy.
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

METHOD 鈥?Covers specific algorithms, architectures, models, or technical approaches.
DATASET 鈥?Describes datasets, corpora, or data collection/annotation procedures.
BENCHMARK 鈥?Discusses evaluation benchmarks, leaderboards, standard test sets, or evaluation metrics/protocols.
ETHICS_AND_SAFETY  鈥?Addresses ethical considerations, fairness, bias, discrimination, privacy, safety, robustness, reliability, or adversarial vulnerabilities.
TOOLKIT 鈥?Covers software libraries, open-source frameworks, toolkits, or code repositories.
APPLICATION - Covers real-world deployment, industrial use cases, or scenario-based selection guidance for reviewed systems.
GENERAL 鈥?FALLBACK ONLY. Assign this tag if and only if none of the tags above (METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION) clearly applies to this section. NEVER assign GENERAL together with any other tag. If you are uncertain whether a specific tag fits but it is the best available option, assign that specific tag without GENERAL.

### Decision Notes
- If the opening text is empty, base your decision on the section title and document title alone.
- Assign all content tags that clearly apply. If you assign any tag other than GENERAL, do not add GENERAL. GENERAL is only valid as the sole tag when no other tag fits at all.
- When the section title is ambiguous (e.g., "Discussion"), use the opening text to decide between FUTURE_WORK and CONCLUSION.
- A subsection inherits no constraints from its parent section type; classify it solely on its own title and opening text.
- "Discussion" titles: inspect opening text 鈥?field-level open problems 鈫?FUTURE_WORK; summary of findings 鈫?CONCLUSION; scope exclusions 鈫?LIMITATION. 
- LIMITATION is often a short subsection inside Introduction or Conclusion; position does not override content. Decisive signal: "we do not discuss", "out of scope", "we exclude". 
- Perspective sentences ("we believe", "in our view") within any section do not change that section's functional type; they are sentence-level phenomena.

### Conflict resolution: preamble background vs. section title 
Many survey body sections open with 1鈥? sentences of motivating background before surveying specific works. If the section title names a substantive technical area (methods, systems, tools, hardware), classify as CONTENT even when the opening sentences are background in nature. The preamble background belongs to the section but does not determine its functional type. 

Apply BACKGROUND only when the entire purpose of the section is to provide prerequisite knowledge 鈥?not when it is a survey body section that happens to start with motivation.

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

### Step 1 鈥?Exclusion check

- Return {{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE"}} if: The grammatical subject is a named prior system, paper, model, dataset, or method 鈥?not the current survey. Example triggers: "[Named System] is proposed / introduced / presented / designed / consists of"

- Return {{"excluded": true, "reason": "NON_VERIFIABLE"}} if any of:
  * Pure intent or hope with no content claim: "We hope this work will inspire..."
  * Bare list header: "Our contributions are:", "This paper:"
  * Generic quality claim with no named topic: "a thorough and comprehensive survey" (alone, no topic)
  * Document-level organization without content: "This paper is organized as follows."

  Note: "Section 2 introduces X" is NOT non-verifiable. It claims Section 2 covers topic X and must be extracted.

If neither exclusion applies, proceed to Step 2.

### Step 2 鈥?Claim extraction

For each claim, produce one entry with three fields: section, type, target.

鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
section (string)
  Scope of this specific claim.
  "document"       鈥?claim applies to the whole paper
  "Section 2"      鈥?use exact number if stated
  "Figure 1"       鈥?claim applies to a specific figure
  "Table 1"       鈥?claim applies to a specific table
  "Related Work"   鈥?use section title if named but no number
  "this section"   鈥?if text says "in this section" without specifying
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
type (string, choose exactly one from the list below)
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
target (string)
  Concise description of what to look for when verifying.
  Always required; provide even when type already names the label.
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

### Type definitions

- sentence:COMPARISON
  * Definition: The claim is verified by finding COMPARISON sentences (explicit cross-work contrasts using "unlike", "in contrast", "compared to") or multi-dimensional comparison tables in the body.
  * Use when: "comparative study", "we compare X and Y along dimensions", "Table N contrasts systems across criteria".

- sentence:SYNTHESIS
  * Definition: Verified by finding SYNTHESIS sentences that organize multiple works into categories, trends, or unified abstractions (e.g., "these methods fall into three families...").
  * Use when: "we synthesize findings across approaches", "methods are unified under a common framework", "we propose a taxonomy of X".

- section:EVALUATION
  * Definition: Verified by finding a section with functional type EVALUATION 鈥?a dedicated section for systematic comparison of systems using benchmarks, metrics, or tables.
  * Use when: "a dedicated evaluation section is provided", "Section N presents a benchmark comparison of systems", "performance results are summarized in Table N".

- section:FUTURE_WORK
  * Definition: Verified by finding a section with functional type FUTURE_WORK - a section that identify unresolved problems, missing work, or open questions.
  * Use when: "open challenges are discussed", "future research directions are identified", "open problems are studied", "we highlight unsolved issues in Section X".

- tag:METHOD
  * Definition: Verified by finding a section with content tag METHOD, covering algorithm classes, model architectures, or technical approaches as a topic in themselves.
  * Use when: "we survey optimization methods", "deep learning approaches are reviewed", "we cover model compression techniques". For highly specific named domains 鈫?prefer coverage.

- tag:DATASET
  * Definition: Verified by finding a section with content_tag DATASET.
  * Use when: "datasets are surveyed", "existing corpora are reviewed", "data collection and annotation methods are discussed".

- tag:BENCHMARK
  * Definition: Verified by finding a section with content_tag BENCHMARK. This type is for reviewing benchmark protocols and evaluation metrics as a topic, not for comparing systems on benchmarks.
  * Use when: "evaluation benchmarks are reviewed", "we survey existing metrics and their limitations", "benchmark datasets are categorized". Do NOT use when the claim is about comparing systems 鈫?section:EVALUATION.

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
  * Reviewing what benchmarks exist and their properties 鈫?tag:BENCHMARK
  * Using benchmarks to compare systems against each other 鈫?section:EVALUATION

- tag:METHOD vs coverage
  * General method class without named domain 鈫?tag:METHOD
  * Specific named sub-domain or cross-domain application 鈫?coverage

### Multi-claim handling

Extract each claim as a separate entry. One sentence may yield multiple entries with different sections, types, or targets.

Example:
  Input: "Section 2 introduces edge computing systems, Section 3 benchmarks performance across systems, Section 4 discusses open research challenges."
  Output:
  ```
  {{
    "claims": [
      {{"section": "Section 2", "type": "coverage", "target": "edge computing systems"}},
      {{"section": "Section 3", "type": "section:EVALUATION", "target": "systematic benchmark comparison of edge systems"}},
      {{"section": "Section 4", "type": "section:FUTURE_WORK", "target": "open research challenges"}}
    ]
  }}
  ```

Example:
  Input: "We provide a comprehensive overview of privacy-preserving methods and a comparative analysis of their computational overhead."
  Output:
  {{
    "claims": [
      {{"section": "document", "type": "tag:ETHICS", "target": "privacy-preserving methods"}},
      {{"section": "document", "type": "sentence:COMPARISON", "target": "computational overhead comparison across privacy-preserving approaches"}}
    ]
  }}

### Output format (JSON only)

If excluded:
{{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE | NON_VERIFIABLE"}}

If not excluded:
{{
  "excluded": false,
  "claims": [
    {{
      "section": "...",
      "type": "...",
      "target": "..."
    }}
  ]
}}
"""

CONTRIBUTION_LABELS = {"sentence:COMPARISON", "sentence:SYNTHESIS", "section:EVALUATION", "section:FUTURE_WORK", "tag:METHOD", "tag:DATASET", "tag:BENCHMARK", "tag:ETHICS_AND_SAFETY", "tag:TOOLKIT", "tag:APPLICATION", "coverage"}

# preprocess/claim_segmentation.py
CLAIM_SEGMENTATION = '''
You are a precise claim extractor for citation verification. Extract
minimal, independently verifiable claims from the paragraph below.

INPUT
Paragraph:
"""{paragraph}"""

RULES

1. Atomicity: Each claim must express exactly ONE checkable statement about ONE specific entity (a paper, method, model, system). Split coordinate predicates/objects sharing one citation (e.g., "does A and B") into separate claims, one per item 鈥?UNLESS the joint statement only makes sense together (e.g., "X and Y jointly demonstrated Z"), in which case keep them as one claim.

2. Citation attachment:
   - Multiple markers supporting the same statement stay in one claim.
   - A pronoun or implicit reference to an entity introduced earlier in THIS paragraph inherits that entity's citation key(s); replace the pronoun with the resolved entity name.
   - If no citation can be found or inherited, set citation_keys to [].

3. Verifiability label 鈥?mark a claim "unverifiable" if ANY apply:
   - It is not associated to any citations.
   - It only restates or continues a prior statement without adding a new checkable detail (e.g., "Following [10], we consider...").
   - It is a categorical/definitional statement about a class of things, backed by multiple citations, not specific to any single cited work (e.g., "LLMs are models with massive parameter sizes [a,b,c]").
   - It is a catch-all enumeration tail with no specific referent (e.g., "...and other restoration tasks").
   Otherwise mark "verifiable" 鈥?this includes short claims, as long as they assert a specific, checkable property of one named entity (e.g., "X is an inductive framework for learning node embeddings").

4. Skip pure transitions/hedges that make no checkable statement at all (e.g., "this remains an active research area") 鈥?do not output these as claims.

5. Verbatim constraint: every word must come from the paragraph, except resolved entity names substituted for pronouns.

OUTPUT (JSON only)
{{
  "claims": [
    {{"claim": "...", "verifiable": true|false, "citation_keys": ["..."]}}
  ]
}}

EXAMPLE 1 鈥?coordinate splitting, all verifiable

Paragraph:
"""[12] introduced GraphSAGE, an inductive framework for learning node embeddings on large graphs. It achieves strong results on node classification and link prediction."""

Output:
{{
  "claims": [
    {{"claim": "[12] introduced GraphSAGE, an inductive framework for learning node embeddings on large graphs.", "verifiable": true, "citation_keys": ["12"]}},
    {{"claim": "GraphSAGE achieves strong results on node classification.", "verifiable": true, "citation_keys": ["12"]}},
    {{"claim": "GraphSAGE achieves strong results on link prediction.", "verifiable": true, "citation_keys": ["12"]}}
  ]
}}

EXAMPLE 2 鈥?categorical statement (unverifiable) vs specific claim

Paragraph:
"""Large Language Models (LLMs) [19, 91, 255] are advanced language models with massive parameter sizes. GPT-3 [7] showed that scaling parameters enables few-shot learning without fine-tuning."""

Output:
{{
  "claims": [
    {{"claim": "Large Language Models [19, 91, 255] are advanced language models
      with massive parameter sizes.", "verifiable": false,
      "citation_keys": ["19", "91", "255"]}},
    {{"claim": "GPT-3 [7] showed that scaling parameters enables few-shot
      learning without fine-tuning.", "verifiable": true,
      "citation_keys": ["7"]}}
  ]
}}

EXAMPLE 3 鈥?continuation sentence (unverifiable) + enumeration tail

Paragraph:
"""Diffusion models excel at super-resolution, inpainting, and other restoration tasks [33]."""

Output:
{{
  "claims": [
    {{"claim": "Diffusion models excel at super-resolution.", "verifiable": true, "citation_keys": ["33"]}},
    {{"claim": "Diffusion models excel at inpainting.", "verifiable": true, "citation_keys": ["33"]}}
    {{"claim": "Diffusion models excel at other restoration tasks.", "verifiable": false, "citation_keys": ["33"]}}
  ]
}}

CHECK before output: every claim is atomic, every token is traceable to the paragraph (except resolved pronouns), unverifiable claims are still output (not silently dropped), and no pure transition/hedge sentence was extracted.
'''

CLAIM_SCHEMA = {
    "type": "object",
    "required": ["claim", "verifiable", "citation_keys"],
    "properties": {
        "claim": {"type": "string", "minLength": 1},
        "verifiable": {"type": "boolean"},
        "citation_keys": {"type": "array", "items": {"type": "string"}}
    }
}

CLAIMS_SCHEMA = {
    "type": "object", 
    "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": CLAIM_SCHEMA}}
}


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
You will be given a paragraph from a survey in the field of Computer Sciences. Identify every mention of a specific research object: a named method, model, architecture, algorithm, dataset, benchmark, evaluation metric, software framework, or named technical/methodological concept that the survey is discussing, introducing, comparing, or crediting as part of the literature it is reviewing.

Extraction is judged by MEANING, not by capitalization or spelling. A lowercase multi-word phrase naming a specific paradigm or method family (e.g. a category of approaches) is just as valid a target as an acronym or a capitalized model name.

### Extract a term when:
- It names a specific approach, model, dataset, benchmark, metric, or technical concept that is being surveyed, cited, or positioned as part of the related literature 鈥?something a reader would reasonably expect to trace back to a specific source.
- This includes general paradigm/family names when they are being surveyed as a category of prior work (e.g. "early approaches relied on X and Y", where X and Y are method families, not single papers).

### Do NOT extract a term when:
- It is a foundational mathematical, statistical, or physical concept borrowed into the field of {query} as a generic tool, rather than a specific named contribution of {query} itself.
- It is a certain type of task or application scenario. (e.g. text generation)
- It names a cluster of works or a method family. This usually appear at the first or the last sentence of the paragraph.
- It is mentioned only as an incidental implementation detail or generic engineering practice used to carry out the survey-external experiments being described (e.g. routine training/evaluation procedures mentioned in passing), rather than as a research object the survey is actually discussing.
- It is a bare common noun or generic descriptive word with no specific referent (e.g. "the model", "this approach", "the dataset" used anaphorically).

### For each extracted entity, also judge: locally_cited
- Set "locally_cited": true if, within this same sentence or an immediately adjacent sentence, there is a citation marker that is clearly providing the source FOR THIS SPECIFIC ENTITY 鈥?i.e. the citation is attached to this entity's introduction, proposal, or attribution, not attached to some other claim that merely happens to appear in the same sentence.
- Set "locally_cited": false if no such citation is present nearby, OR if a citation marker exists in the sentence but is clearly supporting a different claim (e.g. a result, comparison, or statistic) rather than naming the source of this entity itself.

### Additional task: alias pairs
If the paragraph contains a pattern where a full name and a short form/abbreviation appear together 鈥?in either order, with or without parentheses 鈥?record them as a pair, even if the short form does not look like a standard acronym of the full name.

### Output format (JSON only, no other text)
{{
  "entities": [
    {{"name": "...", "locally_cited": true/false}},
    ...
  ],
  "alias_pairs": [["full form", "short form"], ...]
}}

### Examples

Input:
Early work framed the task with rule-based pattern matching [9], while later systems adopted sequence labeling [10, 11]. More recently, a retrieval-augmented framework called QueryNet (QN) [22] reported strong gains, though follow-up analyses [23] showed QN underperforms on low-resource languages.
Output:
{{
  "entities": [
    {{"name": "rule-based pattern matching", "locally_cited": true}},
    {{"name": "sequence labeling", "locally_cited": true}},
    {{"name": "QueryNet", "locally_cited": true}},
    {{"name": "QN", "locally_cited": true}}
  ],
  "alias_pairs": [["QueryNet", "QN"]]
}}
(Note: [23] in the last clause is attached to a finding about QN's performance, not to introducing QN itself 鈥?QN's own citation is [22], already captured.)

Input:
All models are trained with standard cross-entropy loss and early stopping, using a held-out validation split, consistent with common practice in the broader literature.
Output:
{{
  "entities": [],
  "alias_pairs": []
}}
(Note: generic training/engineering procedures mentioned in passing, not research objects the survey is discussing.)

Input:
The underlying dynamics are often assumed to follow a random walk, whereas the proposed Trend-Aware Recurrent Estimator (TARE) explicitly models seasonal decomposition.
Output:
{{
  "entities": [
    {{"name": "Trend-Aware Recurrent Estimator", "locally_cited": false}},
    {{"name": "TARE", "locally_cited": false}}
  ],
  "alias_pairs": [["Trend-Aware Recurrent Estimator", "TARE"]]
}}
(Note: "random walk" is excluded as a foundational stochastic concept external to the field. TARE is marked locally_cited: false because no citation marker accompanies its introduction in this passage 鈥?even though it is described as "proposed", no source is attached here.)

Input (field: computer vision):
"Several benchmarks have been used to evaluate this task, including ObjectNet-200 [31] and a smaller diagnostic split, MiniBench [32], which was later shown to contain significant label noise [33]."
Output:
{{
  "entities": [
    {{"name": "ObjectNet-200", "locally_cited": true}},
    {{"name": "MiniBench", "locally_cited": true}}
  ],
  "alias_pairs": []
}}
(Note: [33] supports a separate claim about label noise, not the introduction of MiniBench, but MiniBench already has its own citation [32] 鈥?so it is still locally_cited: true overall.)

### Input paragraph
Input: 
"{paragraph}"
Output:"""

# scope/uncited_entities.py
EXTRACT_PROPOSED = """### Task
You will be given the title and abstract of a single paper. Identify every method, model, framework, or technique that this paper itself explicitly claims to propose, introduce, or present as its own contribution.

### Include only claims of original contribution
Extract a name only when the abstract states, in substance, "we propose/introduce/present X" 鈥?i.e. X is being introduced as new work by the authors of THIS paper.

### Do NOT include:
- Names of prior methods this paper merely uses, extends, compares against, or builds on (e.g. "we propose Y based on Z" 鈥?extract Y, not Z; "our approach combines A and B" where A, B are cited prior work 鈥?do not extract A or B)
- Generic descriptions of the paper's general topic or task without a specific named contribution

### Output format (JSON only, no other text)
{{
  "proposed": [
    {{"name": "...", "evidence_sentence": "exact sentence from the abstract"}},
    ...
  ]
}}
The evidence_sentence must be copied verbatim from the input abstract.

### Examples

Input:
Title: "Fast Sampling via Implicit Probability Flow"
Abstract: "Diffusion models require many sampling steps. We present Implicit Flow Sampler (IFS), a non-Markovian sampling procedure that reduces the number of steps needed while maintaining sample quality. IFS builds on the denoising framework introduced in prior diffusion work."
Output:
{{
  "proposed": [
    {{"name": "Implicit Flow Sampler", "evidence_sentence": "We present Implicit Flow Sampler (IFS), a non-Markovian sampling procedure that reduces the number of steps needed while maintaining sample quality."}},
    {{"name": "IFS", "evidence_sentence": "We present Implicit Flow Sampler (IFS), a non-Markovian sampling procedure that reduces the number of steps needed while maintaining sample quality."}}
  ]
}}

Input:
Title: "Improving Conditional Generation with Auxiliary Classifiers"
Abstract: "We build a new conditional generation pipeline based on score-based generative models, incorporating an auxiliary classifier to guide the sampling trajectory. Experiments on standard image benchmarks show improved fidelity."
Output:
{{
  "proposed": []
}}
(Note: "score-based generative models" is prior work this paper builds on, not a contribution of this paper. The paper does not name its own pipeline, so nothing qualifies for extraction.)

### Input
Title: "{title}"
Abstract: "{abstract}"
Output:"""

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

BACKGROUND_CHECK = """Given a claim about what a cited paper is/proposed, and that paper's title+abstract, judge whether the abstract supports this identification.

Claim: "{claim}"
Title: "{title}"
Abstract: "{abstract}"

Judge SUPPORTED if the abstract's content (possibly using different wording) is consistent with the claim's identification of what the paper is/does. Judge NOT SUPPORTED if the abstract does not provide enough information to confirm this, or describes something substantially different.

Output: {{"label": "SUPPORTED"|"NOT SUPPORTED"}}"""

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
  - It describes a specific system, method, or finding whose subject matter has no connection to the promised topic.
  - The sentence is a substantive claim about something else, not a transition, motivation, or framing statement.

A sentence is NOT DIFFERENT if:
  - It motivates, introduces, or provides context for the promised topic (even if it mentions other areas by contrast).
  - It is a comparison or contrast where the main subject is still the promised topic: "Unlike method X, our reviewed approach Y..."
  - It is a transition between sub-topics within the promised topic.

Output: list of sentence indices (e.g. S1, S3) that are DIFFERENT. If none, output an empty list.

### Step 3 — Label subsection titles

For each subsection title, decide: does this subsection title clearly indicate a DIFFERENT topic from the promised topic?

A subsection title is DIFFERENT if:
  - It names a subject that has no plausible connection to the promised topic, even as a sub-component or related aspect.
  - Example: promised topic is "privacy protection", subsection title is "Cache Optimization" — DIFFERENT.
  - Example: promised topic is "privacy protection", subsection title is "Differential Privacy Mechanisms" — SAME.
  - Example: promised topic is "inference optimization", subsection title is "Security Considerations" — DIFFERENT.

A subsection title is NOT DIFFERENT if:
  - It names a component, technique, or sub-area that naturally belongs to the promised topic.
  - It provides a finer-grained categorization of the promised topic.
  - It refers to a specific dataset, benchmark, method name or research entity. A specific name or research entity usually does not follow the spelling rules of English words, such as not a english word, or capitalizing non-initial letters.
  - It is a general section name (background, method, evaluation, limitations, future works, conclusions, ...)

Output: list of subsection IDs (e.g. X.2, X.5) that are DIFFERENT. If none, output an empty list.

### Output format (JSON only)
```json
{{
  "promised_topic": "...",
  "different_sentences": ["S1", "S3"]
  "different_subsections": ["X.2", "X.5"],
}}
```

Do not include any explanation outside the JSON.
"""

# topic_coverage.py
MISSING_TOPIC_CLAIM = """Determine whether the paper **explicitly states** that a given topic is excluded, and why.

### Instructions

1. Search for explicit scope limitation statements.
2. Only accept **clear declarative claims** (e.g., 鈥渨e do not cover鈥︹€?.
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
