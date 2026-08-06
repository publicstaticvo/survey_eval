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
RULES = """### Labels
Apply the rules in order and assign exactly one primary label from: "TEXTUAL", "CONTRIBUTION", "CONTRIBUTION+SCOPE", "SCOPE", "GAP", "COMPARISON", "SYNTHESIS", "SUMMARY", "BACKGROUND".

1. TEXTUAL - States what a particular section, subsection, figure, table, or other paper component contains or does. The scope is a component of the paper rather than the paper as a whole. Examples: "Section 2 reviews subtopic X", "In this section, we overview X", and "Figure 2 illustrates X".

2. CONTRIBUTION - States what the current survey itself presents, proposes, defines, categorizes, introduces, or demonstrates at document scope. The grammatical subject is the current paper, such as "we", "our survey", "this paper", or "this survey". A claim about a named prior work is not CONTRIBUTION.
   2b. CONTRIBUTION+SCOPE - The same document-level contribution sentence also states a literature boundary, such as a time range, venue, language, database, or explicit search or selection method.

3. SCOPE - States the survey's inclusion or exclusion criteria, search methodology, or coverage boundary. Do not use SCOPE for a field open problem, a document-level contribution without a boundary, or a statement that merely describes a section.

4. GAP - States an unresolved problem, limitation, missing capability, or future research direction at cross-work or field level. A limitation of one named method is SUMMARY unless the sentence generalizes it to a broader research gap. A statement that the survey itself excludes a topic is SCOPE.

5. COMPARISON - Establishes a relation between at least two distinct research objects, including named works, method families, tasks, datasets, benchmarks, categories, or theoretical properties. The relation may concern performance, efficiency, strengths, weaknesses, trade-offs, applicability, similarity, difference, or a shared evaluation dimension. Named individual papers are not required. A sentence that only describes one object is SUMMARY; two objects mentioned without a comparative relation remain SUMMARY.

6. SYNTHESIS - Integrates multiple prior works into a trend, category, common mechanism, general principle, or higher-level interpretation. An explicit comparison remains COMPARISON when its primary function is to relate two or more objects; a sentence may be SYNTHESIS when it draws a broader conclusion from that relation.

7. SUMMARY - Describes one or more specific prior works, methods, datasets, or results without organizing them into a broader interpretation and without establishing a comparative relation.

8. BACKGROUND - Provides general definitions, field context, or facts not tied to a specific prior work and not functioning as one of the labels above."""

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

SENTENCE_LABELS = {"TEXTUAL", "CONTRIBUTION", "CONTRIBUTION+SCOPE", "GAP", "SCOPE", "COMPARISON", "SYNTHESIS", "SUMMARY", "BACKGROUND"}

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
Given a section or subsection from a literature review, assign exactly one functional type describing its rhetorical role.

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

### Decision Notes
- If opening text is empty: decide from section/document title alone.
- If the title is "Discussion" only: open problems -> FUTURE_WORK; findings summary -> CONCLUSION; scope exclusions -> SCOPE.
- Subsections inherit no type from their parent; classify each independently.
- Perspective sentences ("we believe") do not change a section's type; that is a sentence-level phenomenon.
- A body section titled after a substantive technical area is CONTENT even if it opens with 1-3 motivating sentences; reserve BACKGROUND for sections whose entire purpose is prerequisite knowledge.

### Output format (JSON only)
{{
  "functional_type": "...",
  "confidence": 0.0
}}"""

SECTION_LABELS = {'SCOPE', 'BACKGROUND', 'CONTENT', 'FUTURE_WORK', 'CONCLUSION'}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "functional_type": {"enum": sorted(SECTION_LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["functional_type", "confidence"],
    "additionalProperties": False,
}

# preprocess/contribution_classify.py
CONTRIBUTION_CLASSIFICATION = """### Task
You classify one sentence that has already been identified as a possible document-level contribution of a literature survey. Extract only claims made by the current survey about what the complete paper contributes, covers, analyzes, or compares. Do not extract claims whose scope is a section, subsection, figure, table, or other paper component; those belong to TEXTUAL_CLASSIFICATION.

### Input
Sentence: "{S}"
Context: "{CONTEXT}"

### Exclusion rules
Return excluded=true with reason PRIOR_WORK_FALSE_POSITIVE when the grammatical subject is a named prior work, method, dataset, author, or citation rather than the current survey. Return excluded=true with reason TEXTUAL_CLAIM when the sentence describes a section, subsection, figure, table, or other component. Return excluded=true with reason NON_VERIFIABLE for a bare contribution heading, a generic quality claim without a concrete target, or a document-organization statement without a substantive promise.

### Extraction rules
For every remaining document-level promise, return one claim. Use section="document". Use type="sentence:COMPARISON" for a promise to compare research objects or findings, type="sentence:SYNTHESIS" for a promise to integrate or interpret multiple works, type="section:FUTURE_WORK" for a promise to discuss open problems or future directions, and type="coverage" for a concrete literature area or object the survey promises to cover. The target must be a concise description of what later content must substantiate. Do not invent a target that is not stated in the sentence.

### Output JSON only
If excluded: {{"excluded": true, "reason": "PRIOR_WORK_FALSE_POSITIVE | TEXTUAL_CLAIM | NON_VERIFIABLE"}}
If not excluded: {{"excluded": false, "claims": [{{"section": "document", "type": "...", "target": "..."}}]}}"""

TEXTUAL_CLASSIFICATION = """### Task
You classify one sentence that describes the organization or content of a particular component of a literature survey. Extract each verifiable component-level promise and do not extract document-level contribution claims.

### Input
Sentence: "{S}"
Context: "{CONTEXT}"

### Exclusion rules
Return excluded=true with reason CONTRIBUTION_CLAIM when the sentence states what the complete survey or paper contributes at document scope. Return excluded=true with reason NON_VERIFIABLE for a bare heading, a generic organization statement with no substantive content, or a sentence that does not make a component-level promise.

### Extraction rules
For every remaining promise, return one claim. The section field must identify the component's scope: an exact section number, section title, "this section", "Figure N", or "Table N". Use type="sentence:TEXTUAL". The target must state what the named component is promised to contain or do. Do not convert a component-level scope into a document-level contribution.

### Output JSON only
If excluded: {{"excluded": true, "reason": "CONTRIBUTION_CLAIM | NON_VERIFIABLE"}}
If not excluded: {{"excluded": false, "claims": [{{"section": "...", "type": "sentence:TEXTUAL", "target": "..."}}]}}"""

COMPARISON_CLASSIFICATION = """### Task
Extract the comparative relation expressed by one sentence from a literature survey. This prompt performs comparison-field extraction only; it does not assign the sentence's primary rhetorical label.

### Input
Sentence: "{S}"
Context: "{CONTEXT}"

### Decision
Return excluded=true if the sentence does not establish a relation between at least two research objects. Objects may be named papers, methods, method families, tasks, datasets, benchmarks, categories, or theoretical properties. Mentioning two objects without a comparative relation is excluded.

### Output fields
For an included sentence, return every distinct comparison relation with comparison_targets, comparison_dimensions, comparison_relation, and verbatim_evidence. comparison_targets must use wording from the input. comparison_dimensions must name the shared axes such as performance, efficiency, robustness, applicability, or design. comparison_relation must state the relation without adding facts. verbatim_evidence must be copied exactly from the sentence.

### Output JSON only
{{"excluded": false, "comparisons": [{{"comparison_targets": ["...", "..."], "comparison_dimensions": ["..."], "comparison_relation": "...", "verbatim_evidence": "..."}}]}}
Or: {{"excluded": true, "reason": "NO_COMPARATIVE_RELATION"}}"""

GAP_CLASSIFICATION = """### Task
Extract the research-gap or future-direction claim expressed by one sentence from a literature survey. This prompt performs gap-field extraction only; it does not assign the sentence's primary rhetorical label.

### Input
Sentence: "{S}"
Context: "{CONTEXT}"

### Decision
Return excluded=true if the sentence only describes one named work's limitation without generalizing it, states the survey's own inclusion boundary, or does not identify an unresolved problem, missing capability, or future direction.

### Output fields
For an included sentence, return gap_scope as one of local, cross_work, or field; gap_status as one of reported, synthesized, or proposed; target as a concise statement of the unresolved issue using only the input's meaning; and verbatim_evidence copied exactly from the sentence. Use local only when a limitation is explicitly generalized beyond a single work's result; otherwise a single-work description is excluded.

### Output JSON only
{{"excluded": false, "gap_scope": "local|cross_work|field", "gap_status": "reported|synthesized|proposed", "target": "...", "verbatim_evidence": "..."}}
Or: {{"excluded": true, "reason": "NO_RESEARCH_GAP"}}"""

TEXTUAL_SCHEMA = {
    "type": "object",
    "oneOf": [
        {"properties": {"excluded": {"const": True}, "reason": {"enum": ["CONTRIBUTION_CLAIM", "NON_VERIFIABLE"]}}, "required": ["excluded", "reason"], "additionalProperties": False},
        {"properties": {"excluded": {"const": False}, "claims": {"type": "array", "items": {"type": "object", "properties": {"section": {"type": "string", "minLength": 1}, "type": {"const": "sentence:TEXTUAL"}, "target": {"type": "string", "minLength": 1}}, "required": ["section", "type", "target"], "additionalProperties": False}}}, "required": ["excluded", "claims"], "additionalProperties": False}
    ]
}

COMPARISON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {"properties": {"excluded": {"const": True}, "reason": {"const": "NO_COMPARATIVE_RELATION"}}, "required": ["excluded", "reason"], "additionalProperties": False},
        {"properties": {"excluded": {"const": False}, "comparisons": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"comparison_targets": {"type": "array", "minItems": 2, "items": {"type": "string", "minLength": 1}}, "comparison_dimensions": {"type": "array", "items": {"type": "string", "minLength": 1}}, "comparison_relation": {"type": "string", "minLength": 1}, "verbatim_evidence": {"type": "string", "minLength": 1}}, "required": ["comparison_targets", "comparison_dimensions", "comparison_relation", "verbatim_evidence"], "additionalProperties": False}}}, "required": ["excluded", "comparisons"], "additionalProperties": False}
    ]
}

GAP_SCHEMA = {
    "type": "object",
    "oneOf": [
        {"properties": {"excluded": {"const": True}, "reason": {"const": "NO_RESEARCH_GAP"}}, "required": ["excluded", "reason"], "additionalProperties": False},
        {"properties": {"excluded": {"const": False}, "gap_scope": {"enum": ["local", "cross_work", "field"]}, "gap_status": {"enum": ["reported", "synthesized", "proposed"]}, "target": {"type": "string", "minLength": 1}, "verbatim_evidence": {"type": "string", "minLength": 1}}, "required": ["excluded", "gap_scope", "gap_status", "target", "verbatim_evidence"], "additionalProperties": False}
    ]
}

CONTRIBUTION_LABELS = {"sentence:COMPARISON", "sentence:SYNTHESIS", "sentence:TEXTUAL", "section:EVALUATION", "section:FUTURE_WORK", "tag:METHOD", "tag:DATASET", "tag:BENCHMARK", "tag:ETHICS_AND_SAFETY", "tag:APPLICATION", "coverage"}

# preprocess/claim_segmentation.py
CLAIM_SEGMENTATION = '''You are a precise claim extractor for citation verification. Extract minimal, independently verifiable claims from the sentences marked <E id="...">...</E> below. Unmarked sentences are context only - never extract claims from them, use them only to resolve references inside <E> sentences.

### INPUT
"""{range}"""

### RULES

1. Atomicity: split "and"-joined coordinate predicates/objects sharing one citation into separate claims, one per item - UNLESS the joint statement only makes sense together (e.g., "X and Y jointly demonstrated Z"). Do NOT split "or"-joined items; keep them as one claim (splitting would turn "at least one holds" into two independently-required claims).

2. Citation attachment:
   - Multiple markers supporting the same statement stay in one claim.
   - A pronoun/implicit reference inherits its antecedent's citation key(s); substitute the resolved entity name for the pronoun.
   - No citation, no valid antecedent to inherit from 闁?citation_keys = [].

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
            "reason": {"enum": ["PRIOR_WORK_FALSE_POSITIVE", "TEXTUAL_CLAIM", "CONTRIBUTION_CLAIM", "NON_VERIFIABLE"]}
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
MISSING_TOPIC_DECISION = """### Task
Classify one Leiden community of uncited papers against the survey's already discussed topics. Use only the supplied titles, abstracts, query, and topic labels.

Survey query: {query}
Existing survey topics:
{existing_topics}
Representative community papers:
{papers}

### Labels
- COVERED: the community is substantively within one or more existing topics. List every matching supplied topic exactly in covered_topics.
- NOVEL: the community is query-relevant and forms a coherent research direction not substantively covered by an existing topic. Give a concise community_name and a concrete reason.
- MIXED: the representative papers instantiate multiple directions and another Leiden split is warranted. List every direction that participates in the mixture in mixed_topics and explain the mixture.
- UNRELATED: the community is not relevant to the survey query. Give a concrete reason.

### Evidence rules
- Every evidence quote must be copied verbatim from the matching title or abstract; do not paraphrase.
- COVERED requires one or more exact supplied topic names and no mixed_topics.
- NOVEL and UNRELATED require a non-empty reason and have no covered_topics or mixed_topics.
- MIXED requires at least two mixed_topics, a non-empty reason, and no covered_topics.

### Output JSON only
{{
  "decision": "COVERED" | "NOVEL" | "MIXED" | "UNRELATED",
  "covered_topics": ["exact existing topic name"],
  "community_name": "concise name required for NOVEL, otherwise empty string",
  "mixed_topics": ["direction represented in the community"],
  "reason": "required for NOVEL, MIXED, and UNRELATED; otherwise an empty string",
  "evidence": [
    {{"paper_index": 1, "quote": "..."}}
  ]
}}"""

MISSING_TOPIC_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["COVERED", "NOVEL", "MIXED", "UNRELATED"]},
        "covered_topics": {"type": "array", "items": {"type": "string"}},
        "community_name": {"type": "string"},
        "mixed_topics": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
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
    "required": ["decision", "covered_topics", "community_name", "mixed_topics", "reason", "evidence"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"decision": {"const": "COVERED"}}},
            "then": {"properties": {"covered_topics": {"minItems": 1}, "community_name": {"maxLength": 0}, "mixed_topics": {"maxItems": 0}, "reason": {"maxLength": 0}}},
        },
        {
            "if": {"properties": {"decision": {"const": "NOVEL"}}},
            "then": {"properties": {"covered_topics": {"maxItems": 0}, "community_name": {"minLength": 1}, "mixed_topics": {"maxItems": 0}, "reason": {"minLength": 1}}},
        },
        {
            "if": {"properties": {"decision": {"const": "UNRELATED"}}},
            "then": {"properties": {"covered_topics": {"maxItems": 0}, "community_name": {"maxLength": 0}, "mixed_topics": {"maxItems": 0}, "reason": {"minLength": 1}}},
        },
        {
            "if": {"properties": {"decision": {"const": "MIXED"}}},
            "then": {"properties": {"covered_topics": {"maxItems": 0}, "community_name": {"maxLength": 0}, "mixed_topics": {"minItems": 2}, "reason": {"minLength": 1}}},
        },
    ],
}

REFERENCE_ANCHOR_RELEVANCE = """### Task
Judge whether each candidate paper is substantively relevant to the supplied survey subtopic and therefore merits inclusion in a short missing-reference shortlist. Use the subtopic, its survey context, and each candidate abstract. Do not infer relevance from a title alone. A relevant candidate must directly study the subtopic, one of its named research objects, or a method/task/dataset central to it. For each relevant candidate, copy a verbatim supporting span from the candidate abstract; do not paraphrase. For each irrelevant candidate, return an empty evidence string.

Survey topic: {query}
Subtopic: {topic}
Survey context: {section_context}
Candidate papers:
{candidate_papers}

### Output JSON only
{{
  "papers": [
    {{"paper_id": "...", "relevant": true, "verbatim_evidence": "...", "reason": "..."}}
  ]
}}"""

REFERENCE_ANCHOR_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "minLength": 1},
                    "relevant": {"type": "boolean"},
                    "verbatim_evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["paper_id", "relevant", "verbatim_evidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["papers"],
    "additionalProperties": False,
}

NOVELTY_COMPARISON = """### Task
Assess whether the target survey states a substantive contribution that differentiates it from the supplied prior surveys on the same target topic. Organizational roadmaps and generic claims of comprehensiveness are not substantive differences. Base the decision only on the supplied text. Copy all evidence verbatim; do not paraphrase.

Target topic: {query}
Target-survey contribution text:
{contribution_text}
Prior surveys:
{reference_surveys}

### Output JSON only
{{
  "differentiated": true,
  "target_quote": "verbatim target-survey contribution span, or empty if absent",
  "prior_quote": "verbatim prior-survey title/abstract span supporting the comparison, or empty if unavailable",
  "reason": "concise comparison"
}}"""

NOVELTY_COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "differentiated": {"type": "boolean"},
        "target_quote": {"type": "string"},
        "prior_quote": {"type": "string"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["differentiated", "target_quote", "prior_quote", "reason"],
    "additionalProperties": False,
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
JUDGE_UNCITED_BATCH = """### Task
You are a strict academic verifier. Determine whether each candidate paper is the original source of a named research artifact (model, dataset, method, benchmark, metric, task, concept, or framework).

Candidate artifact: {entity_name}
Candidate abstracts:
{candidate_papers}

### Decision rules
Judge yes only when the abstract explicitly says that the paper proposes, introduces, presents, names, defines, releases, or creates the artifact. Ignore titles completely: title occurrence is not evidence of authorship. Judge no when the artifact is only used, compared, evaluated, or mentioned as prior work. Return uncertain when the abstract lacks enough information. For every yes decision, copy the proposing sentence verbatim from the abstract; do not paraphrase.

### Output JSON only
{{
  "entity": "{entity_name}",
  "results": [
    {{"paper_index": 1, "decision": "yes" | "no" | "uncertain", "matched_rule": "abstract" | "exclusion" | "none", "evidence": "verbatim abstract quote or empty string", "confidence": "high" | "medium" | "low"}}
  ],
  "most_likely_source": null
}}
Use null whenever no paper is judged yes. Otherwise return the index of the strongest yes decision."""

JUDGE_UNCITED_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_index": {"type": "integer", "minimum": 1},
        "decision": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "matched_rule": {"type": "string", "enum": ["abstract", "exclusion", "none"]},
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
Determine whether an uncited candidate has a specific citation role inside an already-covered survey subfield. Relevance alone is insufficient.

Survey query: {query}
Section title: {section_title}
Section topics: {topics}
Section text: {section_text}
Already cited papers in this subfield: {cited_papers}
Candidate paper:
Title: {candidate_title}
Abstract: {candidate_abstract}
Citation-neighborhood retrieval evidence: {graph_evidence}

### Labels
Choose exactly one: concept_symbol_or_landmark_warrant, attribution_warrant, taxonomy_or_scope_warrant, claim_support_or_counterevidence_warrant, benchmark_dataset_evaluation_warrant, recency_update_warrant, weak_related_work_suggestion, no_obligation.
Set citation_obligation=true only for the first six labels. Do not infer a missing topic. A candidate discovered through the citation graph but lacking a concrete role is weak_related_work_suggestion or no_obligation.

### Evidence rule
Evidence must be a verbatim quote copied from the supplied section text, cited-paper summaries, candidate title, or candidate abstract. Do not paraphrase. Graph proximity alone cannot be the evidence for an obligation.

### Output JSON only
{{
  "warrant_label": "concept_symbol_or_landmark_warrant" | "attribution_warrant" | "taxonomy_or_scope_warrant" | "claim_support_or_counterevidence_warrant" | "benchmark_dataset_evaluation_warrant" | "recency_update_warrant" | "weak_related_work_suggestion" | "no_obligation",
  "citation_obligation": true | false,
  "evidence": "verbatim quote from the supplied textual input",
  "reasoning": "one or two sentences explaining the citation role"
}}"""

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

TAXONOMY_FRAMEWORK_PROBLEM_PROMPT = """### Task
Audit one evidence-bounded taxonomy candidate. Do not criticize stylistic choices.

Survey topic: {topic}
Candidate kind: {candidate_kind}
Sibling labels or category: {artifact}
Survey context: {context}
Retrieved overlap evidence: {external_evidence}

### Labels
Choose exactly one: MISSING_CATEGORY_DEFINITION, OVERLAPPING_CATEGORIES, MIXED_ORGANIZING_AXES, NO_COMMENT.
- MISSING_CATEGORY_DEFINITION applies only to an undefined category candidate whose supplied context contains neither a definition nor a defining citation.
- For sibling_partition_direct, OVERLAPPING_CATEGORIES requires the supplied survey context itself to establish overlapping category boundaries; do not infer overlap from names alone.
- For sibling_partition_retrieval, OVERLAPPING_CATEGORIES requires a retrieved paper whose abstract verbatim establishes simultaneous membership in both siblings.
- MIXED_ORGANIZING_AXES requires survey text showing that sibling definitions use incompatible organizing principles.

### Evidence rule
For any finding, survey_quote must be copied verbatim from the survey context. A retrieval-based overlap must also include an external_quote copied verbatim from retrieved evidence. A direct judgment must leave external_quote empty. Do not paraphrase.

### Output JSON only
{{
  "problem_type": "MISSING_CATEGORY_DEFINITION" | "OVERLAPPING_CATEGORIES" | "MIXED_ORGANIZING_AXES" | "NO_COMMENT",
  "comment": "one concise reviewer-facing finding",
  "implicated_labels": ["..."],
  "survey_quote": "verbatim survey quote, or empty for NO_COMMENT",
  "external_quote": "verbatim retrieved quote only for retrieval-based overlap",
  "evidence_summary": "why the evidence supports the finding",
  "alternative_interpretation": "residual ambiguity, or empty",
  "confidence": 0.0
}}"""

TAXONOMY_FRAMEWORK_PROBLEM_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_type": {"type": "string", "enum": ["MISSING_CATEGORY_DEFINITION", "OVERLAPPING_CATEGORIES", "MIXED_ORGANIZING_AXES", "NO_COMMENT"]},
        "comment": {"type": "string"},
        "implicated_labels": {"type": "array", "items": {"type": "string"}},
        "survey_quote": {"type": "string"},
        "external_quote": {"type": "string"},
        "evidence_summary": {"type": "string"},
        "alternative_interpretation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["problem_type", "comment", "implicated_labels", "survey_quote", "external_quote", "evidence_summary", "alternative_interpretation", "confidence"],
    "additionalProperties": False,
}

EVIDENCE_SUPPORT_INSUFFICIENT_PROMPT = """### Task
Decide whether a strong or synthesis claim has a visible argument chain in the supplied survey span. An argument chain is a logical, comparative, or empirical bridge from stated observations to the claim; merely repeating the conclusion is not support.

Survey topic: {topic}
Candidate claim: {claim}
Search span ({search_scope}): {support_span}

### Decision
Return NO_COMMENT if the span contains a sufficient argument chain. Otherwise return ARGUMENT_SUPPORT_FAILURE. Do not use external literature or unstated domain knowledge.

### Evidence rule
For ARGUMENT_SUPPORT_FAILURE, claim_quote must copy the claim verbatim and support_quote must be an empty string. For NO_COMMENT, support_quote must copy the supporting bridge verbatim from the search span. Do not paraphrase either field.

### Output JSON only
{{
  "problem_type": "ARGUMENT_SUPPORT_FAILURE" | "NO_COMMENT",
  "comment": "one concise reviewer-facing finding, or empty string for NO_COMMENT",
  "claim_quote": "verbatim claim quote for a failure, otherwise empty string",
  "support_quote": "verbatim reasoning bridge for NO_COMMENT, otherwise empty string",
  "reasoning": "why the span does or does not supply the required bridge",
  "confidence": 0.0
}}"""

EVIDENCE_SUPPORT_INSUFFICIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "problem_type": {"type": "string", "enum": ["ARGUMENT_SUPPORT_FAILURE", "NO_COMMENT"]},
        "comment": {"type": "string"},
        "claim_quote": {"type": "string"},
        "support_quote": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["problem_type", "comment", "claim_quote", "support_quote", "reasoning", "confidence"],
    "additionalProperties": False,
}

# Finding role assignment
AFFECTED_CLAIM_PROMPT = """### Task
Identify the single survey sentence that would have to be corrected to resolve the supplied finding. Do not judge severity and do not assign a review role.

### Input

Finding: {finding}

Finding evidence: {finding_evidence}

context: {context}

### Selection rule
Search the given context. Select only a sentence whose content is directly contradicted, unsupported, or made structurally incorrect by the finding. Do not select a sentence merely because it discusses a related topic, and do not infer an unstated downstream consequence. If no explicit sentence is directly implicated, return an empty affected_claim; this includes structural omissions for which the missing statement does not exist in the survey.

### Evidence rule
affected_claim must be either an empty string or one complete verbatim sentence copied from the supplied paragraph or section context. Do not paraphrase, combine, shorten, or repair the sentence. reason must explain why that exact sentence is directly affected, or why no explicit affected sentence exists. Do not output Weakness, Requested Change, severity, importance, sentence type, or any other field.

### Output JSON only
{{
  "affected_claim": "one complete verbatim sentence, or an empty string",
  "reason": "one concise explanation"
}}"""

AFFECTED_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "affected_claim": {"type": "string"},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["affected_claim", "reason"],
    "additionalProperties": False,
}

CONTENT_PARSE_WITH_TOPICS = """### Task
You will read one section of a survey paper. Extract:
(1) research objects -- specific methods, models, datasets, benchmarks, or frameworks from the literature that this section discusses
(2) the topic(s) this section is organized around, each anchored to verbatim textual evidence
(3) which topic each object belongs to, anchored to verbatim textual evidence

### Topics
Use a list of topic objects. Each topic object must contain:
- label: copied verbatim from the text, not invented.
- anchor_type: one of section_title, text_span, or inferred.
- evidence_span: exact substring from section_text that anchors the topic, or null when the section title alone is the anchor.

Default: the section's own title is one topic, with anchor_type = section_title and evidence_span = null.

Only add additional topics if the text contains an explicit sentence that itself names sub-categories AND the rest of the section actually follows that split. For each such topic, set anchor_type = text_span and evidence_span to the exact clause that states the defining criterion.

### Assigning objects to topics
For each object, only assign it to a specific sub-topic if you can quote a verbatim span from section_text that directly ties that object to that sub-topic. If no such direct textual link exists, assign the object to the default section topic.

### Output format (JSON only, no other text)
{{
  "topics": [
    {{"label": "...", "anchor_type": "section_title|text_span|inferred", "evidence_span": "..." or null}}
  ],
  "objects": [
    {{
      "citation_keys": ["..."],
      "name": "..." or null,
      "topics": ["..."],
      "evidence_span": "..." or null
    }}
  ]
}}

### Input
Paper title: "{paper_title}"
Section title: "{section_title}"
Section text:
"{section_text}"
Output:"""

CONTENT_PARSE_WITH_TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "anchor_type": {"type": "string", "enum": ["section_title", "text_span", "inferred"]},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["label", "anchor_type", "evidence_span"],
                "additionalProperties": False,
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_keys": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": ["string", "null"]},
                    "topics": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["citation_keys", "name", "topics", "evidence_span"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["topics", "objects"],
    "additionalProperties": True,
}

LITERATURE_POOL_RELEVANCE = """### Task
You are filtering candidate papers for a survey evaluation literature pool. Given the survey topic and a batch of candidate titles and abstracts, decide whether each candidate is substantively related to the survey topic. Mark relevant=true only if the paper studies the same research topic, a direct subtopic, a method family, a task, a dataset/benchmark, or an application branch that a survey on this topic could reasonably cover. Reject candidates that only share generic words, belong to a different sense of an ambiguous phrase, or are merely broad background. For each relevant candidate, copy one verbatim evidence span from its title or abstract; do not paraphrase. For each irrelevant candidate, set verbatim_evidence to an empty string.

### Output format (JSON only, no other text)
{{
  "papers": [
    {{"paper_id": "...", "relevant": true, "verbatim_evidence": "..."}}
  ]
}}

### Input
Survey topic: "{query}"
Candidate papers:
{papers}
Output:"""

LITERATURE_POOL_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "minLength": 1},
                    "relevant": {"type": "boolean"},
                    "verbatim_evidence": {"type": "string"},
                },
                "required": ["paper_id", "relevant", "verbatim_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["papers"],
    "additionalProperties": False,
}
