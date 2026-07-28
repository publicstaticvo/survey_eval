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

# preprocess/sentences.py
RULES = """### Labels (apply Rule 1 first; then the first applicable rule among 2-7; output one primary label from: "CONTRIBUTION", "CONTRIBUTION+SCOPE", "SCOPE", "GAP", "CONTRAST", "SYNTHESIS", "SUMMARY", "BACKGROUND")

1. CONTRIBUTION - Subject is the current paper itself (we/our/this survey + propose/define/categorize/present/introduce/show/demonstrate). If the subject is a named prior work, an author list or a citation mark, use SUMMARY or CONTRAST instead. Checked first, regardless of content.
   1b. CONTRIBUTION+SCOPE - the same sentence also states a literature boundary: time range, venue, language, database, or explicit search/selection method. E.g. "We review deep learning methods for X published since 2020" -> CONTRIBUTION+SCOPE.

2. SCOPE - States literature inclusion/exclusion criteria, search methodology, or boundaries of coverage (time range, venue, language, database; "we exclude...", "we do not discuss...", "we focus only on..."). Not the field's open problems (-> GAP); not a bare contribution claim without a boundary (-> CONTRIBUTION).

3. GAP - An unresolved problem or missing capability in the field, typically pointing to future work. Excludes motivational framing that only justifies the survey's own existence. Exclude weakness statements specified on a particular method.
   - GAP: "Prior works failed to address X" (a missing capability in the field)
   - SCOPE: "We do not discuss X" (the survey's own existence)
   - SUMMARY: "Model M failed to address X" (a missing capability of a particular method)

4. CONTRAST - Explicitly compares two or more SPECIFICALLY NAMED prior works (not generic groups) via contrast markers (unlike, whereas, outperforms, compared to) or side-by-side metrics.
   - CONTRAST: "X outperforms Y on Z"
   - SUMMARY: "X suffers from poor generalization" (single object, no comparison target)
   - SUMMARY: "X has three layers while Y has five" (structural fact, no comparison marker)

5. SYNTHESIS - Organizes multiple prior works into categories/trends using collective subjects (studies, methods, approaches), or states the author's own interpretive stance (believe, argue, suggest) even in first person. Yields to CONTRAST when two+ named works are explicitly contrasted; yields to Rule 1 only if the first-person subject uses a Rule-1 contribution verb.

6. SUMMARY - Describes one or more specific named prior works without organizing, judging, or contrasting them.

7. BACKGROUND - General field context, definitions, or facts not tied to specific named works."""

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

SENTENCE_LABELS = {"CONTRIBUTION", "CONTRIBUTION+SCOPE", "GAP", "SCOPE", "CONTRAST", "SYNTHESIS", "SUMMARY", "BACKGROUND"}

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

### Functional Types (choose exactly ONE)

- SCOPE
  * Definition: States literature inclusion/exclusion criteria or search/selection methodology - what time range, venues, languages, or databases were searched, or what topics/methods are explicitly excluded and why. Do NOT assign SCOPE merely because a section is titled "Introduction" or states the survey's contributions/objectives - that default is BACKGROUND.
  * Typical titles: Scope of This Survey, Search Methodology, Inclusion/Exclusion Criteria, Survey Methodology.

- BACKGROUND
  * Definition: Prerequisite knowledge for the rest of the survey - formal definitions, notation, problem formulation, core concepts, theory, or historical development that are not the survey's own contribution. Introductory/motivational content defaults here unless it meets the SCOPE definition above.
  * Typical titles: Introduction, Background, Preliminaries, Foundations, Notation, Motivation, Problem Formulation.

- CONTENT
  * Definition: Reviews specific methods, approaches, systems, or sub-areas following the survey's organizational scheme. Default type for body sections matching no other type.
  * Typical titles: domain- or method-specific titles.

- FUTURE_WORK
  * Definition: Open problems, unsolved challenges, or directions for the research community. Includes perspective statements ("we believe", "we envision") when they constitute section-level content.
  * Typical titles: Future Work, Open Problems, Challenges, Outlook, Research Challenges.

- CONCLUSION
  * Definition: Summarizes the survey's main findings and contributions; introduces no new content or open questions.
  * Typical titles: Conclusion, Summary, Concluding Remarks.

### Content Tag Definitions (choose ALL that apply)

* METHOD - Covers specific algorithms, theories, architectures, models, or technical approaches.
* DATASET - Describes datasets, corpora, or data collection/annotation procedures.
* BENCHMARK - Discusses evaluation benchmarks, leaderboards, standard test sets, or evaluation metrics/protocols.
* ETHICS_AND_SAFETY  - Addresses ethical considerations, fairness, bias, discrimination, privacy, safety, robustness, reliability, or adversarial vulnerabilities.
* APPLICATION - Covers real-world deployment, industrial use cases, or scenario-based selection guidance for reviewed systems.
* GENERAL - FALLBACK ONLY. Assign this tag if and only if none of the tags above (`METHOD`, `DATASET`, `BENCHMARK`, `ETHICS_AND_SAFETY`, `APPLICATION`) clearly applies to this section. NEVER combine `GENERAL` with another tag.

### Decision Notes
- If opening text is empty: decide from section/document title alone.
- If the title is "Discussion" only: open problems -> FUTURE_WORK; findings summary -> CONCLUSION; scope exclusions -> SCOPE.
- Subsections inherit no type from their parent; classify each independently.
- Perspective sentences ("we believe") do not change a section's type; that is a sentence-level phenomenon.
- A body section titled after a substantive technical area is CONTENT even if it opens with 1-3 motivating sentences; reserve BACKGROUND for sections whose entire purpose is prerequisite knowledge.

### Output format (JSON only)
{{
  "functional_type": "...",
  "content_tags": ["...", "..."],
  "confidence": 0.0
}}"""

SECTION_LABELS = {'SCOPE', 'BACKGROUND', 'CONTENT', 'FUTURE_WORK', 'CONCLUSION'}

CONTENT_TAGS = {'METHOD', 'DATASET', 'BENCHMARK', 'ETHICS_AND_SAFETY', 'APPLICATION', 'GENERAL'}

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

- Return {{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE"}} if: The grammatical subject is a named prior system, paper, model, dataset, or method - not the current survey. Example triggers: "[Named System] / This method is proposed / introduced / presented / designed / consists of"

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

- sentence:CONTRAST
  * Definition: The claim is verified by finding CONTRAST sentences (explicit cross-work contrasts using "unlike", "in contrast", "compared to") or multi-dimensional comparison tables in the body.
  * Use when: "comparative study", "we compare X and Y along dimensions", "Table N contrasts systems across criteria".

- sentence:SYNTHESIS
  * Definition: Verified by finding SYNTHESIS sentences that organize multiple works into categories, trends, or unified abstractions (e.g., "these methods fall into three families...").
  * Use when: "we synthesize findings across approaches", "methods are unified under a common framework", "we propose a taxonomy of X".

- section:EVALUATION
  * Definition: Verified by finding a section with functional type EVALUATION - a dedicated section for systematic evaluation or comparison of systems using benchmarks, metrics, or tables.
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
      {{"section": "document", "type": "sentence:CONTRAST", "target": "computational overhead comparison across privacy-preserving approaches"}}
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

CONTRIBUTION_LABELS = {"sentence:CONTRAST", "sentence:SYNTHESIS", "section:EVALUATION", "section:FUTURE_WORK", "tag:METHOD", "tag:DATASET", "tag:BENCHMARK", "tag:ETHICS_AND_SAFETY", "tag:APPLICATION", "coverage"}

# preprocess/claim_segmentation.py
CLAIM_SEGMENTATION = '''You are a precise claim extractor for citation verification. Extract minimal, independently verifiable claims from the sentences marked <E id="...">...</E> below. Unmarked sentences are context only - never extract claims from them, use them only to resolve references inside <E> sentences.

### INPUT
"""{range}"""

### RULES

1. Atomicity: split "and"-joined coordinate predicates/objects sharing one citation into separate claims, one per item - UNLESS the joint statement only makes sense together (e.g., "X and Y jointly demonstrated Z"). Do NOT split "or"-joined items; keep them as one claim (splitting would turn "at least one holds" into two independently-required claims).

2. Citation attachment:
   - Multiple markers supporting the same statement stay in one claim.
   - A pronoun/implicit reference inherits its antecedent's citation key(s); substitute the resolved entity name for the pronoun.
   - No citation, no valid antecedent to inherit from 閳?citation_keys = [].

3. Source tracking: every claim must list the id(s) of every <E> sentence it draws content from - the sentence containing the predicate, plus (if a pronoun/implicit reference was resolved) the sentence containing the antecedent. A claim built entirely from one sentence has one id.

4. Ambiguity flag - mark "verifiable": false if ANY apply:
   - The reference (pronoun/implicit mention) cannot be resolved to a specific antecedent within the given text.
   - It is a categorical/definitional statement about a class of things, backed by multiple citations, not specific to any single cited work.
   - It is a catch-all enumeration tail with no specific referent (e.g., "...and other applications").
   - It only restates/continues a prior statement without adding a new checkable detail.
   - Its subject is a generic technique/design pattern/paradigm rather than one identifiable named entity (paper, method, model, dataset, system).
   Otherwise mark "verifiable": true.

5. Skip pure transitions/hedges with no checkable statement at all (e.g., "this remains an active research area") - do not output these as claims.

6. Verbatim constraint: every word must come from the text, except resolved entity names substituted for pronouns.

### OUTPUT (JSON only)
{{
  "claims": [
    {{"claim": "...", "verifiable": true|false, "citation_keys": ["..."], "source_ids": ["..."]}}
  ]
}}

### EXAMPLE 1 - and/or splitting, cross-sentence coreference, generic-subject ambiguity

"""<E id="1">[12] introduced GraphSAGE, an inductive framework for learning node embeddings on large graphs.</E> <E id="2">It achieves strong results on node classification and link prediction.</E> <E id="3">Sinusoidal or learned time embeddings can be used for time conditioning [8].</E>"""

Output:
{{
  "claims": [
    {{"claim": "[12] introduced GraphSAGE, an inductive framework for learning node embeddings on large graphs.", "verifiable": true, "citation_keys": ["12"], "source_ids": ["1"]}},
    {{"claim": "GraphSAGE achieves strong results on node classification.", "verifiable": true, "citation_keys": ["12"], "source_ids": ["1", "2"]}},
    {{"claim": "GraphSAGE achieves strong results on link prediction.", "verifiable": true, "citation_keys": ["12"], "source_ids": ["1", "2"]}},
    {{"claim": "Sinusoidal or learned time embeddings can be used for time conditioning.", "verifiable": false, "citation_keys": ["8"], "source_ids": ["3"]}}
  ]
}}

### EXAMPLE 2 - unresolved reference, categorical statement, enumeration tail

"""He proposed a two-stage training approach for it, achieving strong performance on the benchmark [44]. <E id="4">Large Language Models (LLMs) [19, 91, 255] are advanced language models with massive parameter sizes.</E> <E id="5">This has enabled progress on visual question answering, and other multimodal applications.</E>"""

Output:
{{
  "claims": [
    {{"claim": "Large Language Models are advanced language models with massive parameter sizes.", "verifiable": false, "citation_keys": ["19", "91", "255"], "source_ids": ["4"]}},
    {{"claim": "Large Language Models have enabled progress on visual question answering.", "verifiable": true, "citation_keys": ["19", "91", "255"], "source_ids": ["5"]}},
    {{"claim": "Large Language Models have enabled progress on other multimodal applications.", "verifiable": false, "citation_keys": ["19", "91", "255"], "source_ids": ["5"]}}
  ]
}}

CHECK before output: every claim has a non-empty source pointing to <E> sentence id(s) actually used, "or" was not split while "and" was, every unresolved reference is flagged verifiable:false rather than guessed, and no pure transition/hedge sentence was extracted.
'''

CLAIM_SCHEMA = {
    "type": "object",
    "required": ["claim", "verifiable", "citation_keys", "source_ids"],
    "properties": {
        "claim": {"type": "string", "minLength": 1},
        "verifiable": {"type": "boolean"},
        "citation_keys": {"type": "array", "items": {"type": "string"}},
        "source_ids": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}
    }
}

CLAIMS_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": CLAIM_SCHEMA}}
}

CONTRIBUTION_SCHEMA = {
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
CONTENT_PARSE_WITH_TOPICS = """### Task
You will read one section of a survey paper. Extract:
(1) research objects -- specific methods, models, datasets, benchmarks, or frameworks from the literature that this section discusses
(2) the topic(s) this section is organized around, each anchored to verbatim textual evidence
(3) which topic each object belongs to, anchored to verbatim textual evidence

### Objects: what to extract
Look for sentences of the form "[Author(s)] [citation] proposed/introduced/developed/found/showed/presented X" -- extract X as one object, tied to that citation.

X may have a proper name (e.g. "DDPM") or no proper name at all. If X has no proper name, output name as an empty string "" -- do not write a description in its place.

Key test when a sentence lists several items after the citation's verb:
- If the list items ARE the specific thing the cited work did/built (its own technical content), extract the whole list as ONE object representing that contribution (name = "" if it has no proper name).
- If the list items are pre-existing factors/dimensions that the cited work merely organizes, unifies, or analyzes together (signal words: "as a coherent/unified X", "spans/covers A, B, and C"), extract only the framework itself as ONE object with name = "". Do NOT extract the individual listed factors as separate objects.

Do NOT extract:
- Anything with no citation attached (generic background/common-knowledge statements).
- Evaluation metrics used to judge other objects, unless the metric itself is the cited contribution.

### Topics
Default: the section's own title is one topic, with is_explicit = false and definition_span = null.

Only add additional topics if the text contains an explicit sentence that itself names sub-categories (e.g. "these fall into two categories: X and Y") AND the rest of the section actually follows that split. For each such topic:
- label: copied verbatim from the text, not invented.
- is_explicit: true.
- definition_span: the verbatim clause from section_text that states the defining criterion of this sub-category (e.g. for "sparse retrieval methods", the clause "which rely on lexical matching such as BM25 [12]"). This span must be an exact substring of section_text -- it will be checked post-hoc.

### Assigning objects to topics -- evidence-grounding rule
For each object, only assign it to a specific sub-topic (an is_explicit=true topic) if you can quote a verbatim span from section_text that directly ties that object to that sub-topic (e.g. it appears within the defining clause, or a later sentence explicitly places it "within" / "under" / "as a type of" that sub-topic).

If no such direct textual link exists for any sub-topic -- even when the section has explicit sub-topics and the object is merely discussed in that general vicinity -- do NOT guess based on proximity or ordering. Instead:
- assign topics = [the section title] (the default topic),
- and set evidence_span to the sentence that introduces/describes the object itself (without a topic-linking claim).

For every object that IS assigned to an explicit sub-topic, evidence_span must be the verbatim span that supports that specific assignment, not just a general description of the object.

### Output format (JSON only, no other text)
{{
  "topics": [
    {{"label": "...", "is_explicit": true/false, "definition_span": "..." or null}},
    ...
  ],
  "objects": [
    {{
      "citation_keys": ["..."],
      "name": "..." or null,
      "topics": ["..."],
      "evidence_span": "..." or null
    }},
    ...
  ]
}}

### Examples

Example 1

Paper title: "A Survey of Retrieval-Augmented Language Models"
Section title: "Retrieval-Augmented Generation Methods"
Section text:
"Retrieval-augmented approaches can be grouped into two categories: sparse retrieval methods, which rely on lexical matching such as BM25 [12], and dense retrieval methods, which encode queries and documents into a shared embedding space, as in Dense Passage Retrieval (DPR) [45]. Within dense retrieval, Retrieval-Augmented Generation (RAG) [50] further conditions the generator directly on retrieved passages, while Fusion-in-Decoder (FiD) instead fuses each passage's representation separately before decoding. A related line of work, proposed by Chen et al. [71], explores caching retrieved passages across queries to reduce latency, though this has not yet been evaluated on standard benchmarks."

Output:
{{
  "topics": [
    {{"label": "sparse retrieval methods", "is_explicit": true, "definition_span": "which rely on lexical matching such as BM25 [12]"}},
    {{"label": "dense retrieval methods", "is_explicit": true, "definition_span": "which encode queries and documents into a shared embedding space, as in Dense Passage Retrieval (DPR) [45]"}}
  ],
  "objects": [
    {{"citation_keys": ["12"], "name": "BM25", "topics": ["sparse retrieval methods"], "evidence_span": "which rely on lexical matching such as BM25 [12]"}},
    {{"citation_keys": ["45"], "name": "DPR", "topics": ["dense retrieval methods"], "evidence_span": "which encode queries and documents into a shared embedding space, as in Dense Passage Retrieval (DPR) [45]"}},
    {{"citation_keys": ["50"], "name": "RAG", "topics": ["dense retrieval methods"], "evidence_span": "Within dense retrieval, Retrieval-Augmented Generation (RAG) [50] further conditions the generator directly on retrieved passages"}},
    {{"citation_keys": [], "name": "FiD", "topics": ["dense retrieval methods"], "evidence_span": "Within dense retrieval, Retrieval-Augmented Generation (RAG) [50] further conditions the generator directly on retrieved passages, while Fusion-in-Decoder (FiD) instead fuses each passage's representation separately before decoding"}},
    {{"citation_keys": ["71"], "name": "", "topics": ["Retrieval-Augmented Generation Methods"], "evidence_span": "A related line of work, proposed by Chen et al. [71], explores caching retrieved passages across queries to reduce latency"}}
  ]
}}
(Note: BM25, DPR each ground directly in their defining clause. RAG and FiD are both explicitly placed "within dense retrieval" by the same sentence, so both cite that sentence as evidence. Chen et al.'s caching method is discussed after the two-way split but is never explicitly placed "within dense retrieval" or any sub-topic -- unlike the earlier version of this example, it is NOT assigned to "dense retrieval methods" on the basis of proximity. It falls back to the section title as its topic, since no direct textual link to either sub-topic exists.)

Example 2

Paper title: "A Survey of Diffusion Models"
Section title: "Likelihoods, Weighting, and Training Objectives"
Section text:
"Diffusion models can be trained through variational lower bounds, denoising losses, or hybrid objectives. Nichol and Dhariwal [31] found that learning reverse-process variances and modifying the objective improved both sample quality and log-likelihood. Karras et al. [44] analyzed noise levels, preconditioning, loss weighting, and sampler design as a coherent design space, showing that many empirical improvements can be understood as better numerical and statistical choices rather than wholly new model families."

Output:
{{
  "topics": [
    {{"label": "Likelihoods, Weighting, and Training Objectives", "is_explicit": false, "definition_span": null}}
  ],
  "objects": [
    {{"citation_keys": ["31"], "name": "", "topics": ["Likelihoods, Weighting, and Training Objectives"], "evidence_span": "Nichol and Dhariwal [31] found that learning reverse-process variances and modifying the objective improved both sample quality and log-likelihood"}},
    {{"citation_keys": ["44"], "name": "", "topics": ["Likelihoods, Weighting, and Training Objectives"], "evidence_span": "Karras et al. [44] analyzed noise levels, preconditioning, loss weighting, and sampler design as a coherent design space"}}
  ]
}}
(Note: The first sentence lists training-objective types with no citation attached, so nothing is extracted from it. No explicit categorization sentence partitions this section, so both objects fall under the single default topic, is_explicit = false, definition_span = null.)

### Input
Paper title: "{paper_title}"
Section title: "{section_title}"
Section text:
"{section_text}"
Output:"""

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

# preprocess/get_reference_surveys.py
REFERENCE_SURVEY_SELECT = """You are a professional academic researcher selecting reference surveys to evaluate a target survey titled "{query}".

### Input
You are given a list of candidate papers with their titles and abstracts.

### Definition: Reference Survey
A **reference survey** is a field-level survey that:
- Treats the query topic as its PRIMARY organizing focus, not as a tool or method applied within a different domain.
- Organizes the literature into coherent conceptual or methodological dimensions (e.g., taxonomies, categorizations, design spaces).
- Covers multiple distinct sub-topics within the query field.
- Would be consulted by an expert to judge whether another survey on this topic has missed important topics or references.

### Inclusion Criteria for Reference Surveys (ALL must be satisfied)
1. The query topic is the main research object AND the primary organizing principle of the paper.
2. At least THREE distinct sub-topics within the query field are covered.
3. The paper synthesizes existing literature rather than reporting original experimental results.

### Exclusion Criteria (ANY triggers exclusion from BOTH tiers)
- Not a survey: excludes benchmarks, position papers, tutorials, or original research papers.
- Primary subject is a downstream application domain, with the query topic appearing only as the method used (e.g., "Transformers for Medical Imaging" is excluded when evaluating a survey on Transformers).
- Covers only ONE task or sub-area within the query field.
- Mentions the query topic only as background or one method among many.

### Classification Instructions
- First apply exclusion criteria. If any exclusion criterion is met, discard the candidate entirely.
- For remaining candidates, apply inclusion criteria. If ALL inclusion criteria are satisfied -- reference survey. Otherwise -- discard.
- Select at most 3 surveys in total.
- Be conservative: fewer is better than including a marginal candidate.
- If no candidate meets reference survey criteria, return an empty list.

### Candidate Surveys
{candidates}

### Output Format
Return JSON only, no extra text.
{{
  "reference_surveys": [
    {{
      "title": "...",
      "covered_subtopics": ["subtopic1", "subtopic2", "subtopic3"],
      "reason": "one sentence explaining why this qualifies as a reference survey"
    }}
  ]
}}
"""

REFERENCE_SURVEY_SCHEMA = {
  "type": "object",
  "properties": {
    "reference_surveys": {
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
    }
  },
  "required": ["reference_surveys"],
  "additionalProperties": False
}

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

Use a strict contradiction standard:
- Mark REFUTED only when the evidence explicitly contradicts the core factual content of the claim.
- Do not mark REFUTED merely because the evidence uses different terminology, a broader/narrower formulation, or an implementation-level description that is compatible with the claim.
- For method descriptions, treat high-level descriptions and implementation descriptions as compatible unless they cannot both be true.
- If the evidence is related but does not clearly support or clearly contradict the claim, choose NEUTRAL.
- If you mark REFUTED, include a concise contradiction_reason explaining why the claim and evidence cannot both be true.

Your output should be a single JSON object only:

```json
{{
  "judgment": "SUPPORTED" | "REFUTED" | "NEUTRAL",
  "evidence": "verbatim evidence from the cited paper, if judgment == SUPPORTED or REFUTED" | "" (if judgment == NEUTRAL),
  "contradiction_reason": "why the claim and evidence cannot both be true, if judgment == REFUTED" | ""
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

Note: a section's topic can legitimately be the previous section's topic extended along a new dimension (e.g. image generation -- video generation, by adding the temporal dimension). If this section's title plausibly extends the previous section's topic this way, include that connection in your statement of the promised topic -- sentences continuing that thread should not be flagged just because they lack an explicit keyword (e.g. a sentence about temporal modeling under "Video Generation" need not say "video").

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
2. Two pools of evidence extracted from the paper itself:
   - SECTION TITLES AND TOPICS: all section/subsection headings in the paper, in document order, along with the subtopics and research objects discussed in each sections, as identified by a discourse parser.
   - CANDIDATE SENTENCES: sentences from the paper (synthesis, summary, evaluation, or comparison statements) that are most semantically related to the contribution claim, retrieved and reranked by relevance.

Your task: determine whether the CANDIDATE evidence, taken together, actually supports that the contribution claim was fulfilled in the paper's body content -- not just mentioned in passing, but substantively covered.

### Rules

- Use BOTH evidence pools jointly. A contribution claim may be supported by a section alone (e.g. a dedicated section title matching the claim), by section topics or objects alone (e.g. a research object tag matching the claim even if no section title matches), by sentences alone, or by a combination.
- Do NOT require an exact lexical match. Judge semantic correspondence (e.g. "sampling acceleration techniques" is satisfied by a section titled "Sampling Acceleration and Distillation").
- If the contribution claim is a compound/enumerative statement covering multiple items (e.g. "spanning A, B, and C"), you are given ONE such item to check at a time -- do not require the full compound claim to be covered by this evidence alone.
- If NO evidence in either pool plausibly relates to the contribution claim, conclude NOT_FULFILLED.
- If evidence relates to the contribution claim's general topic but does not show substantive treatment (e.g. only one passing mention, no dedicated discussion), conclude PARTIALLY_FULFILLED.
- If evidence clearly shows the contribution claim was substantively addressed, conclude FULFILLED.
- You must cite which specific piece(s) of evidence (by pool and item) support your conclusion. Do not fabricate evidence not present in the given pools.
- For `section` evidence, copy the exact section label after `- `, e.g. `Section 7.1 Image Synthesis`.
- For `topic` evidence, copy only the topic name after `Topic:`.
- For `object` evidence, copy only the object name after `Object:`.
- For `sentence` evidence, copy the exact candidate sentence item.
- Do not judge the quality or correctness of the content -- only whether the topic was substantively addressed somewhere in the paper.

### Output Foramt

Output strictly in this JSON format:
{{
  "verdict": "FULFILLED" | "PARTIALLY_FULFILLED" | "NOT_FULFILLED",
  "supporting_evidence": [
    {{"pool": "section" | "topic" | "object" | "sentence", "item": "<exact text of the evidence item>"}}
  ],
  "reasoning": "<one to two sentences explaining the judgment, referencing the evidence above>"
}}
If verdict is NOT_FULFILLED, supporting_evidence should be an empty list.

### Input
CONTRIBUTION_CLAIM:
"{claim_text}"

(This is one atomic component of the paper's broader stated contribution: "{original_contribution_sentence}")

SECTION TITLES AND TOPICS:
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
                    "pool": {'enum': ['section', 'topic', 'object', 'sentence']},
                    'item': {'type': 'string'}
                },
                'additionalProperties': False
            }
        },
        'reasoning': {'type': 'string'}
    },
    "additionalProperties": False
}

# scope/missing_topic_detection.py
MISSING_TOPIC_NAME = """### Task
Name a NOVEL Leiden paper community and assign every applicable content tag.

Survey query: {query}
Representative community papers:
{papers}

### Naming rules

- Use 3-6 words.
- The name must be a subordinate research direction of the survey query, not a restatement of the survey query.
- Describe only the common research direction. Do not evaluate importance, novelty, or relation to other fields.
- Be as specific as possible while covering most representative papers. Do not broaden a name merely to cover unrelated papers.

### Content tags

Select all applicable tags:
METHOD, DATASET, BENCHMARK, ETHICS_AND_SAFETY, TOOLKIT, APPLICATION, GENERAL.

* METHOD - Covers specific algorithms, theories, architectures, models, or technical approaches.
* DATASET - Describes datasets, corpora, or data collection/annotation procedures.
* BENCHMARK - Discusses evaluation benchmarks, leaderboards, standard test sets, or evaluation metrics/protocols.
* ETHICS_AND_SAFETY  - Addresses ethical considerations, fairness, bias, discrimination, privacy, safety, robustness, reliability, or adversarial vulnerabilities.
* APPLICATION - Covers real-world deployment, industrial use cases, or scenario-based selection guidance for reviewed systems.
* GENERAL - FALLBACK ONLY. Assign this tag if and only if none of the tags above (`METHOD`, `DATASET`, `BENCHMARK`, `ETHICS_AND_SAFETY`, `APPLICATION`) clearly applies to this section. NEVER combine `GENERAL` with another tag.

### Evidence rules

- Every `paper_index` must identify representative papers supporting the name/tags.
- Every evidence quote must be copied verbatim from the matching title or abstract.
- Do not rely on outside knowledge.

### Output JSON only
{{
  "topic_name": "...",
  "content_tags": ["METHOD", "APPLICATION"],
  "evidence": [
    {{"paper_index": 1, "quote": "..."}}
  ]
}}"""

MISSING_TOPIC_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_name": {"type": "string", "minLength": 1},
        "content_tags": {
            "type": "array",
            "items": {"enum": sorted(CONTENT_TAGS)},
            "minItems": 1,
            "uniqueItems": True,
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_index": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string", "minLength": 1},
                },
                "required": ["paper_index", "quote"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
    },
    "required": ["topic_name", "content_tags", "evidence"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"content_tags": {"contains": {"const": "GENERAL"}}}},
            "then": {"properties": {"content_tags": {"maxItems": 1}}},
        },
    ],
}

MISSING_TOPIC_DECISION = """### Task
Determine whether a Leiden paper community is already covered by an existing survey topic,
is a novel research direction, or contains multiple directions that require another Leiden split.

Survey query: {query}

Existing survey topics:
{existing_topics}

Representative community papers:
{papers}

### Decision labels

- COVERED: the representative papers are substantively covered by one or more existing survey topics. Name those topics exactly in `covered_topics`.
- NOVEL: the papers form a distinct research direction not substantively covered by the existing survey topics.
- MIXED: the representative papers contain multiple unrelated or insufficiently unified directions. This label requests one more Leiden split; do not use it merely because the papers are broad.

### Evidence rules

- Every `paper_index` must identify representative papers supporting the decision.
- Every evidence quote must be copied verbatim from the matching title or abstract.
- For COVERED, `covered_topics` must contain only exact names from the supplied topics.
- For NOVEL and MIXED, `covered_topics` must be empty.
- Do not rely on outside knowledge.

### Output JSON only
{{
  "decision": "COVERED" | "NOVEL" | "MIXED",
  "covered_topics": ["exact existing topic name"],
  "evidence": [
    {{"paper_index": 1, "quote": "..."}}
  ]
}}"""

MISSING_TOPIC_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["COVERED", "NOVEL", "MIXED"]},
        "covered_topics": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_index": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string", "minLength": 1},
                },
                "required": ["paper_index", "quote"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
    },
    "required": ["decision", "covered_topics", "evidence"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"decision": {"const": "COVERED"}}},
            "then": {"properties": {"covered_topics": {"minItems": 1}}},
        },
        {
            "if": {"properties": {"decision": {"enum": ["NOVEL", "MIXED"]}}},
            "then": {"properties": {"covered_topics": {"maxItems": 0}}},
        },
    ],
}

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
(1) TITLE SIGNAL -- the artifact (full name or abbreviation) is the main subject named in the title.
(2) ABSTRACT SIGNAL -- the artifact appears in a sentence with a proposing/naming cue (e.g. "we propose", "we introduce", "we present", "we call this...", "denoted as...", "termed...").

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

# fact/uncited_claims.py
QUERY_EXPAND = """You are a query rewriting assistant for an academic literature search system. You will be given an Atomic Fact (AF) -- a single claim extracted from a survey paper that currently has no citation. Rewrite it into a clean keyword query for an academic database (OpenAlex), so we can find candidate papers that might support or refute this claim.

## Task

Step 1 -- Assess searchability:
Mark `searchable: false` if the claim:
- contains no specific method/dataset/task entity (i.e., a generic statement that could apply to many works)
- is a meta-statement about the survey itself, not about prior work
- is a value judgement with no factual anchor (e.g., "this remains an important direction")

Step 2 -- If searchable, construct the query:
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
  "reasoning": "..." (required for every category; if category is "unknown", explicitly explain why each of method/dataset/benchmark/application was ruled out; otherwise give a brief one-sentence justification)
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


CITATION_WARRANT = """### Task
You are judging whether a candidate uncited paper creates a citation obligation inside an already-covered subsection of a survey.

This is NOT missing-topic detection. The survey section is assumed to already cover the subfield. Your task is to decide whether this candidate should be cited within that covered discussion.

### Theory
Use citation as disciplinary attribution: a missing reference matters when omitting it weakens attribution, positioning, claim support, comparison, benchmark coverage, or the reader's understanding of the covered literature.

### Labels
Choose exactly one:
1. "concept_symbol_or_landmark_warrant" - The candidate is presented by the supplied evidence as foundational, seminal, first, major, state-of-the-art, benchmark-defining, or a representative concept-symbol for the covered discussion.
2. "attribution_warrant" - The candidate appears to be the source that should be credited for a method, dataset, benchmark, term, result, or line of work already discussed in the section.
3. "taxonomy_or_scope_warrant" - The candidate is needed because the section's existing covered taxonomy/scope includes this kind of work and the candidate is a representative in-scope instance.
4. "claim_support_or_counterevidence_warrant" - The candidate supports, qualifies, updates, or challenges a specific claim made in the section.
5. "benchmark_dataset_evaluation_warrant" - The candidate provides an important benchmark, dataset, evaluation protocol, metric, or systematic comparison relevant to the section.
6. "recency_update_warrant" - The candidate is a recent in-scope work that materially updates the section's covered discussion before the survey evaluation date.
7. "weak_related_work_suggestion" - The candidate is relevant and possibly useful, but the provided evidence does not show a clear citation obligation.
8. "no_obligation" - The candidate is off-scope for this covered section, redundant with already cited work, or the evidence does not support citing it here.

### Decision rules
- Do not reward relevance alone. A relevant paper without a clear warrant is "weak_related_work_suggestion".
- Do not infer a missing topic. If the section text does not already cover the topic, choose "no_obligation" or "weak_related_work_suggestion"; topic gaps are handled elsewhere.
- Prefer stronger warrant labels only when the section text, existing citations, candidate abstract, or graph evidence shows why this paper is needed here.
- Use PPR/graph evidence as retrieval evidence, not as a final decision by itself.
- If the candidate only shares a citation neighborhood with cited papers but no citation role is clear, choose "weak_related_work_suggestion".

### Input
Survey query:
{query}

Section title:
{section_title}

Section topics extracted from already-covered text:
{topics}

Section text:
{section_text}

Already cited papers in this section:
{cited_papers}

Candidate uncited paper:
Title: {candidate_title}
Abstract: {candidate_abstract}

PPR / graph evidence:
{graph_evidence}

### Output format (JSON only, no other text)
{{
  "warrant_label": "concept_symbol_or_landmark_warrant" | "attribution_warrant" | "taxonomy_or_scope_warrant" | "claim_support_or_counterevidence_warrant" | "benchmark_dataset_evaluation_warrant" | "recency_update_warrant" | "weak_related_work_suggestion" | "no_obligation",
  "citation_obligation": true | false,
  "evidence": "short quote or concise evidence from the supplied input",
  "reasoning": "one or two sentences explaining why this label follows from the supplied section/candidate/graph evidence"
}}
Output:"""
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

