# perprocess/citation_parser.py
EXTRACT_TITLE = """# Task
Extract the title of the cited work from the following reference string. Return only the title, enclosed in quotation marks. Return the following JSON object only:
{{
  "title": "..."
}}

# Example
Input: Tianhe Lin, Jian Xie, Siyu Yuan, et al. (2025). Implicit Reasoning in Transformers is Reasoning through Shortcuts. Annual Meeting of the Association for Computational Linguistics.
Output:
{{
  "title": "Implicit Reasoning in Transformers is Reasoning through Shortcuts"
}}

# Input
Input: {info}
Output: """

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
  - If ALL strict inclusion criteria are satisfied -- strict reference survey.
  - If strict criteria are not fully met but ALL partial inclusion criteria are satisfied -- partial reference survey.
  - Otherwise -- discard.
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

# preprocess/sentences.py
RULES = """### Labels and definitions (apply Rule 1 first; then use the first applicable rule among 2-7)

1. CONTRIBUTION - The current paper is the subject: an explicit first-person or self-referential subject (we, our, this survey/paper) paired with an action verb (propose, define, categorize, present, introduce, show, demonstrate, categorize). If the subject is a named prior system, model, or method, classify as SUMMARY or EVALUATION instead. Applied before all other rules regardless of content.

2. LIMITATION - Declares what this survey does not cover and why: explicit scope exclusions, acknowledged gaps, or methodological constraints of the review itself. Does not summarize contributions (CONCLUSION) and does not identify field-level open problems (GAP).

3. GAP - Identifies an unresolved problem, open challenge, or missing capability in the research field, typically signaling a direction for future work. Does not include motivational framing that merely justifies the current survey.

4. EVALUATION - The author makes an explicit positive or negative judgment about a specific named prior work, using evaluative language (outperforms, suffers from, is limited by, fails to, effectively handles). The judgment must reflect the author's own stance, not a description of a result or consequence. The current survey must not be one of the evaluated works. Sentences that only report what a prior paper found, demonstrated, or showed - without the survey author adding an evaluative stance - are SUMMARY, not EVALUATION. Examples:
  - EVALUATION: "X is more effective than Y for Z tasks"  (author judgment) 
  - SUMMARY:    "X demonstrated 95% accuracy on dataset Y" (result report with numerical data)
  - SUMMARY:    "Experiments show that X outperforms Y"    (reporting prior work's finding)

5. COMPARISON - Explicitly contrasts two or more specific named prior works (methods, models, or systems) using explicit contrast markers (unlike, in contrast, whereas, compared to) or quantitative side-by-side metrics. The current survey must not be one of the contrasted works.

6. SYNTHESIS - Organizes multiple prior works into categories, trends, or abstractions. Subject is implicit or third-person (studies, methods, approaches, researchers); if the subject is first-person, apply Rule 1 instead.

7. SUMMARY - Describes one or more specific named prior works without organizing, judging, or contrasting them.

8. BACKGROUND - General field context, definitions, or facts. May mention concept names but does not refer to specific works by author or title."""

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
Opening text     : "{PREAMBLE}"       (first 1-3 sentences; may be empty)

### Functional Type Definitions (choose exactly ONE)

- SCOPE
  * Definition: The section states the survey's objectives, contributions, scope, or organization. Typically the first section. May contain an outline of the rest of the paper.
  * Typical titles: Introduction, Overview, Motivation, Contribution, Scope of this survey.

- BACKGROUND
  * Definition: The section provides prerequisite knowledge required to understand the rest of the survey: formal definitions, notation, core concepts, theory, or historical development that are not the survey's primary contribution.
  * Typical titles: Background, Preliminaries, Foundations, Notation.

- TAXONOMY
  * Definition: The section proposes, explains, or justifies a classification framework, categorization scheme, or organizing criteria that structures the rest of the survey. Look for explicit statements such as "we categorize - into", "we organize - according to", or a diagram/table that defines the taxonomy.
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

METHOD - Covers specific algorithms, architectures, models, or technical approaches.
DATASET - Describes datasets, corpora, or data collection/annotation procedures.
BENCHMARK - Discusses evaluation benchmarks, leaderboards, standard test sets, or evaluation metrics/protocols.
ETHICS_AND_SAFETY  - Addresses ethical considerations, fairness, bias, discrimination, privacy, safety, robustness, reliability, or adversarial vulnerabilities.
TOOLKIT - Covers software libraries, open-source frameworks, toolkits, or code repositories.
APPLICATION - Covers real-world deployment, industrial use cases, or scenario-based selection guidance for reviewed systems.
GENERAL - FALLBACK ONLY. Assign this tag if and only if none of the tags above (METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION) clearly applies to this section. NEVER assign GENERAL together with any other tag. If you are uncertain whether a specific tag fits but it is the best available option, assign that specific tag without GENERAL.

### Decision Notes
- If the opening text is empty, base your decision on the section title and document title alone.
- Assign all content tags that clearly apply. If you assign any tag other than GENERAL, do not add GENERAL. GENERAL is only valid as the sole tag when no other tag fits at all.
- When the section title is ambiguous (e.g., "Discussion"), use the opening text to decide between FUTURE_WORK and CONCLUSION.
- A subsection inherits no constraints from its parent section type; classify it solely on its own title and opening text.
- "Discussion" titles: inspect opening text - field-level open problems -- FUTURE_WORK; summary of findings -- CONCLUSION; scope exclusions -- LIMITATION. 
- LIMITATION is often a short subsection inside Introduction or Conclusion; position does not override content. Decisive signal: "we do not discuss", "out of scope", "we exclude". 
- Perspective sentences ("we believe", "in our view") within any section do not change that section's functional type; they are sentence-level phenomena.

### Conflict resolution: preamble background vs. section title 
Many survey body sections open with 1-3 sentences of motivating background before surveying specific works. If the section title names a substantive technical area (methods, systems, tools, hardware), classify as CONTENT even when the opening sentences are background in nature. The preamble background belongs to the section but does not determine its functional type. 

Apply BACKGROUND only when the entire purpose of the section is to provide prerequisite knowledge - not when it is a survey body section that happens to start with motivation.

### Hard constraints on output
- content_tags must contain either GENERAL alone, or one or more tags from {{METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION}}. Any output combining GENERAL with another tag is invalid.

### Output format (JSON only)
{{
  "functional_type": "...",
  "content_tags": ["...", "..."],
  "confidence": 0.0
}}"""

SECTION_LABELS = {'SCOPE', 'BACKGROUND', 'TAXONOMY', 'CONTENT', 'EVALUATION', 'FUTURE_WORK', 'LIMITATION', 'CONCLUSION'}

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

### Step 1 - Exclusion check

- Return {{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE"}} if: The grammatical subject is a named prior system, paper, model, dataset, or method - not the current survey. Example triggers: "[Named System] is proposed / introduced / presented / designed / consists of"

- Return {{"excluded": true, "reason": "NON_VERIFIABLE"}} if any of:
  * Pure intent or hope with no content claim: "We hope this work will inspire..."
  * Bare list header: "Our contributions are:", "This paper:"
  * Generic quality claim with no named topic: "a thorough and comprehensive survey" (alone, no topic)
  * Document-level organization without content: "This paper is organized as follows."

  Note: "Section 2 introduces X" is NOT non-verifiable. It claims Section 2 covers topic X and must be extracted.

If neither exclusion applies, proceed to Step 2.

### Step 2 - Claim extraction

For each claim, produce one entry with three fields: section, type, target.

-------------------------------------------------------------
section (string)
  Scope of this specific claim.
  "document"       - claim applies to the whole paper
  "Section 2"      - use exact number if stated
  "Figure 1"       - claim applies to a specific figure. Figure claims may be extracted, but visual evidence will be skipped by downstream consistency checking.
  "Table 1"       - claim applies to a specific table. Use only integer table numbers from 1 upward; never output chapter-style numbers such as "Table 2.3".
  "Related Work"   - use section title if named but no number
  "this section"   - if text says "in this section / chapter / subsection" without specifying
-------------------------------------------------------------
type (string, choose exactly one from the list below)
-------------------------------------------------------------
target (string)
  Concise description of what to look for when verifying.
  Always required; provide even when type already names the label.
-------------------------------------------------------------

### Type definitions

- sentence:COMPARISON
  * Definition: The claim is verified by finding COMPARISON sentences (explicit cross-work contrasts using "unlike", "in contrast", "compared to") or multi-dimensional comparison tables in the body.
  * Use when: "comparative study", "we compare X and Y along dimensions", "Table N contrasts systems across criteria".

- sentence:SYNTHESIS
  * Definition: Verified by finding SYNTHESIS sentences that organize multiple works into categories, trends, or unified abstractions (e.g., "these methods fall into three families...").
  * Use when: "we synthesize findings across approaches", "methods are unified under a common framework", "we propose a taxonomy of X".

- section:EVALUATION
  * Definition: Verified by finding a section with functional type EVALUATION - a dedicated section for systematic comparison of systems using benchmarks, metrics, or tables.
  * Use when: "a dedicated evaluation section is provided", "Section N presents a benchmark comparison of systems", "performance results are summarized in Table N".

- section:FUTURE_WORK
  * Definition: Verified by finding a section with functional type FUTURE_WORK - a section that identify unresolved problems, missing work, or open questions.
  * Use when: "open challenges are discussed", "future research directions are identified", "open problems are studied", "we highlight unsolved issues in Section X".

- tag:METHOD
  * Definition: Verified by finding a section with content tag METHOD, covering algorithm classes, model architectures, or technical approaches as a topic in themselves.
  * Use when: "we survey optimization methods", "deep learning approaches are reviewed", "we cover model compression techniques". For highly specific named domains - prefer coverage.

- tag:DATASET
  * Definition: Verified by finding a section with content_tag DATASET.
  * Use when: "datasets are surveyed", "existing corpora are reviewed", "data collection and annotation methods are discussed".

- tag:BENCHMARK
  * Definition: Verified by finding a section with content_tag BENCHMARK. This type is for reviewing benchmark protocols and evaluation metrics as a topic, not for comparing systems on benchmarks.
  * Use when: "evaluation benchmarks are reviewed", "we survey existing metrics and their limitations", "benchmark datasets are categorized". Do NOT use when the claim is about comparing systems -- section:EVALUATION.

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
  * Reviewing what benchmarks exist and their properties -- tag:BENCHMARK
  * Using benchmarks to compare systems against each other -- section:EVALUATION

- tag:METHOD vs coverage
  * General method class without named domain -- tag:METHOD
  * Specific named sub-domain or cross-domain application -- coverage

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
CLAIM_SEGMENTATION = '''You are a precise claim extractor for citation verification. Extract minimal, independently verifiable claims from the paragraph below.

INPUT
Paragraph:
"""{range}"""

RULES

1. Atomicity: Each claim must express exactly ONE checkable statement about ONE specific entity (a paper, method, model, system). Split coordinate predicates/objects sharing one citation (e.g., "does A and B") into separate claims, one per item - UNLESS the joint statement only makes sense together (e.g., "X and Y jointly demonstrated Z"), in which case keep them as one claim.

2. Citation attachment:
   - Multiple markers supporting the same statement stay in one claim.
   - A pronoun or implicit reference to an entity introduced earlier in THIS paragraph inherits that entity's citation key(s); replace the pronoun with the resolved entity name.
   - If no citation can be found or inherited, set citation_keys to [].

3. Verifiability label - mark a claim "unverifiable" if ANY apply:
   - It is not associated to any citations.
   - It only restates or continues a prior statement without adding a new checkable detail (e.g., "Following [10], we consider...").
   - It is a categorical/definitional statement about a class of things, backed by multiple citations, not specific to any single cited work (e.g., "LLMs are models with massive parameter sizes [a,b,c]").
   - It is a catch-all enumeration tail with no specific referent (e.g., "...and other restoration tasks").
   Otherwise mark "verifiable" - this includes short claims, as long as they assert a specific, checkable property of one named entity (e.g., "X is an inductive framework for learning node embeddings").

4. Skip pure transitions/hedges that make no checkable statement at all (e.g., "this remains an active research area") - do not output these as claims.

5. Verbatim constraint: every word must come from the paragraph, except resolved entity names substituted for pronouns.

OUTPUT (JSON only)
{{
  "claims": [
    {{"claim": "...", "verifiable": true|false, "citation_keys": ["..."]}}
  ]
}}

EXAMPLE 1 - coordinate splitting, all verifiable

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

EXAMPLE 2 - categorical statement (unverifiable) vs specific claim

Paragraph:
"""Large Language Models (LLMs) [19, 91, 255] are advanced language models with massive parameter sizes. GPT-3 [7] showed that scaling parameters enables few-shot learning without fine-tuning."""

Output:
{{
  "claims": [
    {{"claim": "Large Language Models [19, 91, 255] are advanced language models with massive parameter sizes.", "verifiable": false, "citation_keys": ["19", "91", "255"]}},
    {{"claim": "GPT-3 [7] showed that scaling parameters enables few-shot learning without fine-tuning.", "verifiable": true, "citation_keys": ["7"]}}
  ]
}}

EXAMPLE 3 - continuation sentence (unverifiable) + enumeration tail

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
                  "section": {"type": "string", "minLength": 1},
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


# preprocess/content_parser.py
CONTENT_PARSE = """### Task
You will be given a paragraph from a survey in the field of Computer Sciences. Identify every research object that this paragraph is MAINLY INTRODUCING, PROPOSING, OR CREDITING AS A DISTINCT CONTRIBUTION being reviewed - a named or unnamed method, model, architecture, algorithm, dataset, benchmark, evaluation metric, software framework, or technical/methodological concept that the paragraph treats as a subject of discussion in its own right.

Extraction is judged by MEANING, not by capitalization or spelling. A lowercase multi-word phrase naming a specific paradigm or method family is just as valid a target as an acronym or a capitalized model name. A research object may also have NO name at all - extract it based on its citation and description alone.

### Extract an object when:
- It is the main subject of the sentence/clause: something being introduced, proposed, evaluated, or positioned as part of the literature - something a reader would reasonably expect to trace back to a specific source.
- This includes general paradigm/family names when they are the main subject being surveyed as a category of prior work (e.g. "early approaches relied on X and Y", where X and Y are method families being discussed as such).
- This includes cases where NO proper name is given at all, only a description plus a citation (e.g. "Smith et al. [12] proposed a graph-based approach for ...").

### Do NOT extract an object when:
- It is a foundational mathematical, statistical, or physical concept borrowed into the field as a generic tool.
- It is a type of task or application scenario (e.g. text generation).
- It names a cluster of works or a method family mentioned only in passing at the start/end of the paragraph as a transitional label, not as the actual subject being discussed.
- It is mentioned only as a SUPPORTING DETAIL used to explain another object's internal mechanism or dependency, rather than as the object the sentence is actually about. *Test: if you removed this term, would the sentence still be making its main point? If yes, it is a supporting detail - exclude it.*
- It is a generic engineering practice mentioned in passing (routine training/evaluation procedures) rather than a research object the survey is actually discussing.
- It is a bare common noun with no specific referent (e.g. "the model", "this approach", used anaphorically).

### For each extracted object, also identify: citation_keys
- List every citation marker in this sentence or an immediately adjacent sentence that is clearly attached to THIS object's introduction, proposal, or attribution - not attached to some other claim (e.g. a result, comparison, or statistic about it) that merely happens to appear nearby.
- If no such citation exists, citation_keys must be an empty list.
- An object can have an empty citation_keys list. An object can also have no name (see below) - these are independent, both can be true, false, or mixed.

### For each extracted object, also identify: name
- Set name to the object's proper name/acronym EXACTLY as it appears in the text, if the text gives it one.
- If the text describes the object only by what it does, with no proper name given anywhere in this paragraph, set name to empty string. Do NOT invent, paraphrase, or reconstruct a name from the description.

### Output format (JSON only, no other text)
{{
  "objects": [
    {{"citation_keys": ["..."], "name": "..." or ""}},
    ...
  ]
}}

### Examples

Input:
Early work framed the task with rule-based pattern matching [9], while later systems adopted sequence labeling [10, 11]. More recently, a retrieval-augmented framework called QueryNet (QN) [22] reported strong gains, though follow-up analyses [23] showed QN underperforms on low-resource languages.

Output:
{{
  "objects": [
    {{"citation_keys": ["9"], "name": ""}}
    {{"citation_keys": ["10", "11"], "name": ""}}
    {{"citation_keys": ["22"], "name": "QN"}}
  ]
}}
(Note: [23] supports a finding about QN's performance, not QN's introduction - QN's own citation is [22], already captured.)

Input:
All models are trained with standard cross-entropy loss and early stopping, using a held-out validation split, consistent with common practice in the broader literature.

Output:
{{
  "objects": []
}}
(Note: generic training/engineering procedures, not research objects the survey is discussing.)

Input:
Smith et al. [14] introduced a diffusion-based approach for molecular generation that models the forward process as a fixed Markov chain over atomic coordinates.

Output:
{{
  "objects": [
    {{"citation_keys": ["14"], "name": ""}}
  ]
}}
(Note: this method is never given a proper name in the paragraph - only a citation and a description. Set `name` to empty, `citation_keys` is not empty.)

Input:
The underlying dynamics are often assumed to follow a random walk, whereas the proposed Trend-Aware Recurrent Estimator (TARE) explicitly models seasonal decomposition.

Output:
{{
  "objects": [
    {{"citation_keys": [], "name": "Trend-Aware Recurrent Estimator"}},
    {{"citation_keys": [], "name": "TARE"}}
  ]
}}
(Note: "random walk" is excluded as a foundational concept external to the field. TARE is given a name but no citation accompanies its introduction anywhere in this paragraph - citation_keys is empty even though name is present.)

Input:
"Several benchmarks have been used to evaluate this task, including ObjectNet-200 [31] and a smaller diagnostic split, MiniBench [32], which was later shown to contain significant label noise [33]. The proposed method builds on a standard ResNet backbone [40] for feature extraction."

Output:
{{
  "objects": [
    {{"citation_keys": ["31"], "name": "ObjectNet-200"}},
    {{"citation_keys": ["32"], "name": "MiniBench"}}
  ]
}}
(Note: [33] supports a separate claim about label noise, not MiniBench's introduction, but MiniBench already has its own citation [32]. ResNet is excluded - it is a supporting detail explaining the proposed method's backbone, not the subject being discussed.)

Input:
Diffusion Probabilistic Models are based on U-Net [8] structure.

Output:
{{
  "objects": [
    {{"citation_keys": [], "name": "Diffusion Probabilistic Models"}}
  ]
}}
(Note: This sentence is about Diffusion Probabilistic Models; U-Net is a supporting detail explaining its internals. Do NOT extract U-Net, even though it is named and cited.)

### Input paragraph
Input: 
"{paragraph}"
Output:"""

CONTENT_PARSE_WITH_TOPICS = """### Task
You will be given the FULL TEXT of one section from a survey paper, along with the paper's title and this section's title (and sub-heading path, if any).

Your task, in one pass: 
(1) identify every research object this section discusses as a subject in its own right (same definition as below), 
(2) identify the topic(s) this section organizes its content around, 
(3) assign each extracted object to the topic(s) it belongs to.

### What counts as a research object (extract when):
- A named or unnamed method, model, architecture, algorithm, dataset, benchmark, evaluation metric, software framework, or technical concept that the section treats as a subject being introduced, proposed, evaluated, or positioned as part of the literature.
- General paradigm/family names when they are the main subject being surveyed as a category of prior work.
- Objects with NO proper name given - extract based on citation and description alone (name = null in that case).

### Do NOT extract when:
- It is a foundational concept borrowed as a generic tool, not a contribution of this field.
- It is a task/application scenario, not a research object.
- It is a SUPPORTING DETAIL explaining another object's internal mechanism, not the object actually being discussed. *Test: if removed, would the sentence still make its main point? If yes, exclude it as a supporting detail.*
- It is a generic engineering practice mentioned in passing.
- It is a bare common noun with no specific referent.

### For each extracted object: citation_keys and name
- citation_keys: citation markers clearly attached to THIS object's own introduction/attribution (may be empty list).
- name: the object's proper name exactly as in the text, or null if no proper name is ever given in this section.
- If both a full name and an abbreviation appear for the same object ANYWHERE in this section, use only the abbreviation as name, and treat all its mentions as one single object (do not duplicate).

### STRICT RULES on topics
1. A topic label MUST be grounded in the section's own language, from one of two sources only:
   (a) The section title or sub-heading path itself (or a natural sub-phrase of it).
   (b) An explicit categorization sentence in the text - a sentence that itself names categories (e.g. "these methods fall into two categories: X and Y").
2. Do NOT invent a topic label that abstracts or summarizes beyond what rule 1 allows.
3. If no explicit sub-categorization sentence exists (rule 1b), the section has exactly ONE topic: the section title itself.
4. If explicit sub-categorization sentences exist, the section MAY have multiple topics - one per named category. These are IN ADDITION to, not necessarily replacing, the section title as a topic - use judgment: if the sub-categories fully partition the section's content, use only the sub-categories; if they only cover part of the section, keep the section title as a topic for the remaining objects.
5. An object MAY belong to more than one topic if the text explicitly discusses it under more than one named category.
6. Every topic label must be copied or minimally trimmed from exact wording in the text - not paraphrased into new vocabulary.

### Output format (JSON only, no other text)
{{
  "topics": ["...", "..."],
  "objects": [
    {{
      "citation_keys": ["..."], 
      "name": "..." or null, 
      "topics": ["...", "..."]
    }},
    ...
  ]
}}

### Example

Paper title: "A Survey of Retrieval-Augmented Language Models"
Section title: "Retrieval-Augmented Generation Methods"
Section text:
"Retrieval-augmented approaches can be grouped into two categories: sparse retrieval methods, which rely on lexical matching such as BM25 [12], and dense retrieval methods, which encode queries and documents into a shared embedding space, as in Dense Passage Retrieval (DPR) [45]. Within dense retrieval, Retrieval-Augmented Generation (RAG) [50] further conditions the generator directly on retrieved passages, while Fusion-in-Decoder (FiD) instead fuses each passage's representation separately before decoding. A related line of work, proposed by Chen et al. [71], explores caching retrieved passages across queries to reduce latency, though this has not yet been evaluated on standard benchmarks."

Output:
{{
  "topics": ["sparse retrieval methods", "dense retrieval methods"],
  "entities": [
    {{"citation_keys": ["12"], "name": "BM25", "topics": ["sparse retrieval methods"]}},
    {{"citation_keys": ["45"], "name": "DPR", "topics": ["dense retrieval methods"]}},
    {{"citation_keys": ["50"], "name": "RAG", "topics": ["dense retrieval methods"]}},
    {{"citation_keys": [], "name": "FiD", "topics": ["dense retrieval methods"]}},
    {{"citation_keys": ["71"], "name": "", "topics": ["dense retrieval methods"]}}
  ]
}}
(Note: Dense Passage Retrieval/DPR, Retrieval-Augmented Generation/RAG, and Fusion-in-Decoder/FiD each collapse to a single entry using only the abbreviation. Chen et al.'s caching method has no proper name (empty string) but is still assigned to "dense retrieval methods" since it is discussed within that category's scope. The explicit categorization sentence fully partitions the section's content, so the section title itself is not used as a separate topic.)

### Input
Paper title: "{paper_title}"
Section title: "{section_title}"
Section text: "{section_text}"
Output:"""

CONTENT_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_keys": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": ["string", "null"]},
                },
                "required": ["citation_keys", "name"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["objects"],
    "additionalProperties": True,
}

CONTENT_PARSE_WITH_TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_keys": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": ["string", "null"]},
                    "topics": {"type": "array", "items": {"type": "string", "minLength": 1}},
                },
                "required": ["citation_keys", "name", "topics"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["topics", "objects"],
    "additionalProperties": True,
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
- It names a specific approach, model, dataset, benchmark, metric, or technical concept that is being surveyed, cited, or positioned as part of the related literature - something a reader would reasonably expect to trace back to a specific source.
- This includes general paradigm/family names when they are being surveyed as a category of prior work (e.g. "early approaches relied on X and Y", where X and Y are method families, not single papers).

### Do NOT extract a term when:
- It is a foundational mathematical, statistical, or physical concept borrowed into the field of {query} as a generic tool, rather than a specific named contribution of {query} itself.
- It is a certain type of task or application scenario. (e.g. text generation)
- It names a cluster of works or a method family. This usually appear at the first or the last sentence of the paragraph.
- It is mentioned only as an incidental implementation detail or generic engineering practice used to carry out the survey-external experiments being described (e.g. routine training/evaluation procedures mentioned in passing), rather than as a research object the survey is actually discussing.
- It is a bare common noun or generic descriptive word with no specific referent (e.g. "the model", "this approach", "the dataset" used anaphorically).

### For each extracted entity, also judge: locally_cited
- Set "locally_cited": true if, within this same sentence or an immediately adjacent sentence, there is a citation marker that is clearly providing the source FOR THIS SPECIFIC ENTITY - i.e. the citation is attached to this entity's introduction, proposal, or attribution, not attached to some other claim that merely happens to appear in the same sentence.
- Set "locally_cited": false if no such citation is present nearby, OR if a citation marker exists in the sentence but is clearly supporting a different claim (e.g. a result, comparison, or statistic) rather than naming the source of this entity itself.

### Additional task: alias pairs
If the paragraph contains a pattern where a full name and a short form/abbreviation appear together - in either order, with or without parentheses - record them as a pair, even if the short form does not look like a standard acronym of the full name.

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
    {{"name": "QueryNet", "locally_cited": true}},
    {{"name": "QN", "locally_cited": true}}
  ],
  "alias_pairs": [["QueryNet", "QN"]]
}}
(Note: "rule-based pattern matching" and "sequence labeling" are tasks names instead of entity names, so they are not extracted. [23] in the last clause is attached to a finding about QN's performance, not to introducing QN itself - QN's own citation is [22], already captured.)

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
(Note: "random walk" is excluded as a foundational stochastic concept external to the field. TARE is marked locally_cited: false because no citation marker accompanies its introduction in this passage - even though it is described as "proposed", no source is attached here.)

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
(Note: [33] supports a separate claim about label noise, not the introduction of MiniBench, but MiniBench already has its own citation [32] - so it is still locally_cited: true overall.)

### Input paragraph
Input: 
"{paragraph}"
Output:"""

FIND_ALL_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "locally_cited": {"type": "boolean"},
                },
                "required": ["name", "locally_cited"],
                "additionalProperties": False,
            },
        },
        "alias_pairs": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "string", "minLength": 1},
                    {"type": "string", "minLength": 1},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    "required": ["entities", "alias_pairs"],
    "additionalProperties": False,
}

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
You are given a survey section. Identify items whose topic is CLEARLY DIFFERENT from the topic promised by the section title and opening text.

### Input

Section title: "{SECTION_TITLE}"

Content (inline sentence tags, paragraph structure preserved, background sentences excluded):
{TAGGED_CONTENT}
Format: [S1] Sentence one. [S2] Sentence two. ...

Subsection titles (if any):
{SUBSECTION_LIST}
Format: X.1 Subsection one / X.2 Subsection two / ...

### Step 1 - State the promised topic

In one sentence, state the specific topic this section promises to discuss, based on the title and opening text. Be concrete: "privacy protection mechanisms in edge computing", not "privacy".

Note: a section's topic can legitimately be the previous section's topic extended along a new dimension (e.g. image generation 闂?video generation, by adding the temporal dimension). If this section's title plausibly extends the previous section's topic this way, include that connection in your statement of the promised topic 闂?sentences continuing that thread should not be flagged just because they lack an explicit keyword (e.g. a sentence about temporal modeling under "Video Generation" need not say "video").

### Step 2 - Label content sentences

For each content sentence, decide: does this sentence introduce a topic that is CLEARLY DIFFERENT from the promised topic (as stated in Step 1, including any legitimate extension)?

A sentence is DIFFERENT if:
  - It describes a specific system, method, or finding whose subject matter has no connection to the promised topic.
  - The sentence is a substantive claim about something else, not a transition, motivation, or framing statement.

A sentence is NOT DIFFERENT if:
  - It motivates, introduces, or provides context for the promised topic (even if it mentions other areas by contrast).
  - It is a comparison or contrast where the main subject is still the promised topic: "Unlike method X, our reviewed approach Y..."
  - It is a transition between sub-topics within the promised topic, including the dimension-extension case noted in Step 1.

Output: list of sentence indices (e.g. S1, S3) that are DIFFERENT. If none, output an empty list.

### Step 3 - Label subsection titles

For each subsection title, decide: does this subsection title clearly indicate a DIFFERENT topic from the promised topic?

A subsection title is DIFFERENT if:
  - It names a subject that has no plausible connection to the promised topic, even as a sub-component or related aspect.
  - Example: promised topic is "privacy protection", subsection title is "Cache Optimization" - DIFFERENT.
  - Example: promised topic is "privacy protection", subsection title is "Differential Privacy Mechanisms" - SAME.
  - Example: promised topic is "inference optimization", subsection title is "Security Considerations" - DIFFERENT.

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
  "different_sentences": ["S1", "S3"],
  "different_subsections": ["X.2", "X.5"]
}}
```

Do not include any explanation outside the JSON.
"""
# Previous section title (context only): "{PREV_SECTION_TITLE}"

# contribution/contribution_consistent.py
CONTRIBUTION_CONSISTENT = """You are an expert annotator for academic literature review papers. Your task is to verify whether a survey paper has actually delivered on a specific claim it made about its own content.

### Task

You will be given:
1. A contribution claim extracted from the paper's stated contribution.
2. Three pools of evidence extracted from the paper itself:
   - SECTION TITLES: all section/subsection headings in the paper, in document order.
   - SECTION TOPICS: for each section, the subtopics and research objects discussed there, as identified by a discourse parser.
   - CANDIDATE SENTENCES: sentences from the paper (synthesis, summary, evaluation, or comparison statements) that are most semantically related to the contribution claim, retrieved and reranked by relevance.

Your task: determine whether the CANDIDATE evidence, taken together, actually supports that the contribution claim was fulfilled in the paper's body content 闂?not just mentioned in passing, but substantively covered.

### Rules

- Use ALL THREE evidence pools jointly. A contribution claim may be supported by a title alone (e.g. a dedicated section title matching the claim), by section topics alone (e.g. a research object tag matching the claim even if no title matches), by sentences alone, or by a combination.
- Do NOT require an exact lexical match. Judge semantic correspondence (e.g. "sampling acceleration techniques" is satisfied by a section titled "Sampling Acceleration and Distillation").
- If the contribution claim is a compound/enumerative statement covering multiple items (e.g. "spanning A, B, and C"), you are given ONE such item to check at a time 闂?do not require the full compound claim to be covered by this evidence alone.
- If NO evidence in any of the three pools plausibly relates to the contribution claim, conclude NOT_FULFILLED.
- If evidence relates to the contribution claim's general topic but does not show substantive treatment (e.g. only one passing mention, no dedicated discussion), conclude PARTIALLY_FULFILLED.
- If evidence clearly shows the contribution claim was substantively addressed, conclude FULFILLED.
- You must cite which specific piece(s) of evidence (by pool and item) support your conclusion. Do not fabricate evidence not present in the given pools.
- Do not judge the quality or correctness of the content 闂?only whether the topic was substantively addressed somewhere in the paper.

### Output Foramt

Output strictly in this JSON format:
{{
  "verdict": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",
  "supporting_evidence": [
    {{"pool": "title" | "topic" | "sentence", "item": "<exact text of the evidence item>"}}
  ],
  "reasoning": "<one to two sentences explaining the judgment, referencing the evidence above>"
}}
If verdict is NOT_FULFILLED, supporting_evidence should be an empty list.

### Input
CONTRIBUTION_CLAIM:
"{claim_text}"

(This is one atomic component of the paper's broader stated contribution: "{original_contribution_sentence}")

SECTION TITLES:
{full_list_of_titles_in_order}

SECTION TOPICS:
{full_list_of_section_subtopics_and_research_objects}

CANDIDATE SENTENCES (reranked, top-{K}):
{list_of_reranked_candidate_sentences_with_section_location}

Judge whether this contribution claim is FULFILLED, PARTIALLY_FULFILLED, or NOT_FULFILLED based on the evidence above."""

CONTRIBUTION_CONSISTENT_SCHEMA = {
    'type': 'object',
    'required': ['verdict', 'supporting_evidence', 'reasoning'],
    'properties': {
        "verdict": {'enum': ['FULFILLED', 'PARTIALLY_FULFILLED', 'NOT_FULFILLED']},
        'supporting_evidence': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['pool', 'item'],
                'properties': {
                    "pool": {'enum': ['title', 'topic', 'sentence']},
                    'item': {'type': 'string'}
                },
                'additionalProperties': False
            }
        },
        'reasoning': {'type': 'string'}
    },
    "additionalProperties": False
}

# scope/topic_papers.py
PPR_TYPE = """### Task
You are classifying a research paper into ONE primary category based strictly on its title and abstract. This classification describes what kind of contribution the paper primarily makes to the literature - not what field or task it belongs to.

### Categories (choose exactly one)
1. "method" - The paper's primary contribution is a new algorithm, model, architecture, or technical approach. This applies even if the method is demonstrated on a specific domain, AS LONG AS the technical approach itself (not just its domain adaptation) is the paper's main claimed contribution.
2. "dataset" - The paper's primary contribution is the construction, collection, or release of a new dataset (raw data + annotations, for training or experiments purpose), without proposing a new evaluation protocol or a new method as its main claim.
3. "benchmark" - The paper's primary contribution is an evaluation protocol, task suite, leaderboard, or standardized comparison framework. A benchmark paper MAY include a dataset, but its main claimed contribution is the evaluation methodology/protocol itself, not merely the data.
4. "application" - The paper's primary contribution is applying an existing (not newly proposed) method or system to a specific domain or use case, where the main claimed contribution is the domain adaptation, deployment, or empirical findings in that domain - not a new technical method.
5. "unknown" - Use ONLY if, after considering categories 1-4 in order, none of them can be supported by the title and abstract. Before selecting this, you must explicitly state in "reasoning" why each of the other four categories was ruled out.

### Decision priority (apply in this order)
- If the abstract explicitly claims a new method/model/algorithm as the paper's contribution -- "method", even if it is evaluated on a single application domain.
- Else if the abstract explicitly claims a new evaluation protocol/benchmark/task suite as the contribution -- "benchmark".
- Else if the abstract explicitly claims a new dataset for training or experiments purpose (without a new evaluation protocol as the main claim) -- "dataset".
- Else if the abstract describes applying existing methods/systems to a domain, with the domain findings as the contribution -- "application".
- Else -- "unknown", with mandatory reasoning as specified above.

### Rules
- Base your judgment ONLY on the title and abstract provided. Do not use outside knowledge about the paper or its authors.
- "evidence" must be a short verbatim quote (under 20 words) copied exactly from the title or abstract that supports your category choice. Do not paraphrase the quote.

### Output format (JSON only, no other text)
{{
  "category": "method" | "dataset" | "benchmark" | "application" | "unknown",
  "evidence": "...",
  "reasoning": "..." (required only if category is "unknown", otherwise may be a brief one-sentence note)
}}

### Examples

Input:
Title: "ClinBench: A Standardized Evaluation Suite for Clinical Text Understanding"
Abstract: "We introduce ClinBench, a unified benchmark comprising 12 tasks and a leaderboard for evaluating language models on clinical text understanding. We release the evaluation harness and baseline results for 8 existing models."

Output:
{{
  "category": "benchmark",
  "evidence": "a unified benchmark comprising 12 tasks and a leaderboard",
  "reasoning": "Primary contribution is the evaluation protocol/leaderboard itself, not a new method."
}}

Input:
Title: "Diagnosing Diabetic Retinopathy with a Fine-Tuned Vision Transformer"
Abstract: "We apply a pretrained Vision Transformer, fine-tuned on a private hospital dataset, to the task of diabetic retinopathy grading, achieving strong agreement with expert ophthalmologists in a retrospective clinical study."

Output:
{{
  "category": "application",
  "evidence": "We apply a pretrained Vision Transformer, fine-tuned on a private hospital dataset, to the task of diabetic retinopathy grading",
  "reasoning": "Uses an existing architecture; the contribution is the clinical deployment/findings, not a new method."
}}

Input:
Title: "Sparse Attention with Learned Routing for Long-Context Transformers"
Abstract: "We propose a novel sparse attention mechanism that learns token routing patterns end-to-end, reducing computational cost by 40% while matching full-attention performance on long-document tasks."

Output:
{{
  "category": "method",
  "evidence": "We propose a novel sparse attention mechanism that learns token routing patterns end-to-end",
  "reasoning": "Contribution is a new technical mechanism, not merely an application."
}}

### Input
Title: "{title}"
Abstract: "{abstract}"
Output:"""

# scope/topic_coverage.py
MISSING_TOPIC_CLAIM = """Determine whether the paper **explicitly states** that a given topic is excluded, and why.

### Instructions

1. Search for explicit scope limitation statements.
2. Only accept **clear declarative claims** (e.g., We do not cover X / We exclude X)
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

# scope/uncited_entities.py
JUDGE_UNCITED_BATCH = """You are a strict academic verifier. Determine which of the following candidate papers, if any, is the ORIGINAL SOURCE of a candidate research artifact (a model, dataset, method, benchmark, or framework).

Candidate artifact (full name / abbreviation): {entity_name}

Candidate papers:
{candidate_papers}

## Decision rules (apply to each paper independently)
Judge "yes" if EITHER holds:
(1) TITLE SIGNAL 闂?the artifact (full name or abbreviation) is the main subject named in the title.
(2) ABSTRACT SIGNAL 闂?the artifact appears in a sentence with a proposing/naming cue (e.g. "we propose", "we introduce", "we present", "we call this...", "denoted as...", "termed...").

Judge "no" if the artifact is only mentioned as something being used, compared against, evaluated on, or built upon (e.g. "we adopt X", "compared with X", "following X").

If neither signal is found, output "uncertain".

Abbreviations and full names of the same artifact count as a match.

## Output (JSON only, no extra text)
{{
  "entity": "{entity_name}",
  "results": [
    {{
      "paper_index": 1,
      "decision": "yes" | "no" | "uncertain",
      "matched_rule": "title" | "abstract" | "exclusion" | "none",
      "evidence": "verbatim quote or empty string",
      "confidence": "high" | "medium" | "low"
    }},
    ... (one entry per candidate paper)
  ],
  "most_likely_source": <paper_index or null>  // only if >=1 paper decided "yes"; if multiple, pick the one with strongest evidence
}}"""

JUDGE_UNCITED_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_index": {"type": "integer", "minimum": 1},
        "decision": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "matched_rule": {"type": "string", "enum": ["title", "abstract", "exclusion", "none"]},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["paper_index", "decision", "matched_rule", "evidence", "confidence"],
    "additionalProperties": False,
}

# scope/uncited_prospective.py
QUERY_EXPAND = """You are a query rewriting assistant for an academic literature search system. You will be given an Atomic Fact (AF) 閳?a single claim extracted from a survey paper that currently has no citation. Rewrite it into a clean keyword query for an academic database (OpenAlex), so we can find candidate papers that might support or refute this claim.

## Task

Step 1 閳?Assess searchability:
Mark `searchable: false` if the claim:
- contains no specific method/dataset/task entity (i.e., a generic statement that could apply to many works)
- is a meta-statement about the survey itself, not about prior work
- is a value judgement with no factual anchor (e.g., "this remains an important direction")

Step 2 閳?If searchable, construct the query:
- 3-8 words, keyword-style (NOT a natural language question or full sentence)
- The query MUST be composed only of words/phrases taken verbatim from the claim. Do NOT introduce synonyms, paraphrases, or new terms not present in the claim text.
- No punctuation except hyphens within compound terms (e.g., "few-shot")
- No stopwords, no filler verbs ("shows that", "demonstrates")
- Prioritize the most specific/rare entity first (a specific model or dataset name outranks a generic task name)
- If the claim compares two things, include both entities
- Do NOT include the survey's own framing language (e.g., "recent work")
- `key_entities` MUST be a subset of the words/phrases actually used in the query. Do not list any entity that does not appear in the query string.

## Output format (JSON only, no other text):

{{
  "searchable": true/false,
  "reason_if_not_searchable": "<brief reason, empty string if searchable>",
  "query": "<the search query, empty string if not searchable>",
  "key_entities": ["<entity1>", "<entity2>", ...]
}}

## Examples

### Example 1 (searchable, single entity)
AF: "Speculative decoding techniques reduce inference latency by over 40% compared to standard transformer decoding."

Output:
{{
  "searchable": true,
  "reason_if_not_searchable": "",
  "query": "speculative decoding transformer inference latency",
  "key_entities": ["speculative decoding", "transformer inference latency"]
}}

### Example 2 (searchable, comparison)
AF: "Symbolic approaches tend to generalize better to unseen compositional structures compared to neural approaches."

Output:
{{
  "searchable": true,
  "reason_if_not_searchable": "",
  "query": "symbolic approaches compositional structures neural approaches",
  "key_entities": ["symbolic approaches", "compositional structures", "neural approaches"]
}}

### Example 3 (not searchable)
AF: "Robustness to distribution shift in low-resource settings remains a challenging open problem."

Output:
{{
  "searchable": false,
  "reason_if_not_searchable": "General characterization of a research gap, not a specific factual claim attributable to a particular finding",
  "query": "",
  "key_entities": []
}}

Now process the following AF:

AF: {af_text}
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

# deprecated
BACKGROUND_CHECK = """Given a claim about what a cited paper is/proposed, and that paper's title+abstract, judge whether the abstract supports this identification.

Claim: "{claim}"
Title: "{title}"
Abstract: "{abstract}"

Judge SUPPORTED if the abstract's content (possibly using different wording) is consistent with the claim's identification of what the paper is/does. Judge NOT SUPPORTED if the abstract does not provide enough information to confirm this, or describes something substantially different.

Output: {{"label": "SUPPORTED"|"NOT SUPPORTED"}}"""

EXTRACT_PROPOSED = """### Task
You will be given the title and abstract of a single paper. Identify every method, model, framework, or technique that this paper itself explicitly claims to propose, introduce, or present as its own contribution.

### Include only claims of original contribution
Extract a name only when the abstract states, in substance, "we propose/introduce/present X" - i.e. X is being introduced as new work by the authors of THIS paper.

### Do NOT include:
- Names of prior methods this paper merely uses, extends, compares against, or builds on (e.g. "we propose Y based on Z" - extract Y, not Z; "our approach combines A and B" where A, B are cited prior work - do not extract A or B)
- Generic descriptions of the paper's general topic or task without a specific named contribution.

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

EXTRACT_PROPOSED_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "evidence_sentence": {"type": "string", "minLength": 1},
                },
                "required": ["name", "evidence_sentence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposed"],
    "additionalProperties": False,
}
