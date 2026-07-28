SYSTEM = """You are an expert reviewer with broad knowledge of machine learning and natural language processing research. You will be given the full text of a survey paper. Your task is to assess its overall quality."""

CC_PROMPT = """You are an expert reviewer with broad knowledge of machine learning and natural language processing research. The survey paper to be evaluated is in {input_dir}, as a LaTeX source directory/file or parsed JSON file. Your task is to assess its overall quality.

Because input processing may be imperfect, do not report purely format-level or preprocessing artifacts as weaknesses. For example, placeholder author metadata such as "Author Names", reference formatting artifacts, missing rendered figures/tables, or empty-looking algorithm/table boxes caused by input conversion must not appear in the weakness report.

{requirements}

Write your output as a JSON file in {output_file}."""

USER_PLAIN = """Below is the full text of a survey paper. Read it carefully and evaluate its quality.

<survey>
{SURVEY_FULL_TEXT}
</survey>

<input_processing_note>
Because input processing may be imperfect, do not report purely format-level or preprocessing artifacts as weaknesses. For example, placeholder author metadata such as "Author Names", reference formatting artifacts, missing rendered figures/tables, or empty-looking algorithm/table boxes caused by input conversion must not appear in the weakness report.
</input_processing_note>

Provide your evaluation in the following JSON format. Do not include any text outside the JSON object.

{{
  "summary": "<2-4 sentence overview of the survey's topic, scope, and overall impression>",
  "strengths": [
    "<strength 1>",
    "<strength 2>"
  ],
  "weaknesses": [
    {{
      "issue": "<concise description of the problem>",
      "location": "<section/paragraph/sentence where this occurs, quoted or paraphrased>",
    }}
  ],
  "comments": [
    {{
      "issue": "<concise description of a less critical observation>",
      "location": "<section/paragraph/sentence where this occurs>"
    }}
  ],
  "overall_score": <integer 0-100>,
  "score_justification": "<1-3 sentences explaining the score>"
}}"""

USER_ARISE = """Below is the full text of a survey paper, followed by a rubric you must use to guide your assessment.

<survey>
{SURVEY_FULL_TEXT}
</survey>

<input_processing_note>
Because input processing may be imperfect, do not report purely format-level or preprocessing artifacts as weaknesses. For example, placeholder author metadata such as "Author Names", reference formatting artifacts, missing rendered figures/tables, or empty-looking algorithm/table boxes caused by input conversion must not appear in the weakness report.
</input_processing_note>

<evaluation_rubric>
Score the survey on each of the following criteria using a 1-5 scale (1 = does not meet this criterion at all, 5 = fully meets this criterion). For each criterion, briefly justify your score with reference to specific content in the survey.

You are an expert academic reviewer. Please evaluate the provided literature review / survey paper using the rubric below. For each criterion, assign a score from 5 (excellent) to 1 (poor). Provide brief justifications for each score if possible, and give an overall summary of strengths and weaknesses.

---

Category: Scope
1. Criterion: Objectives
   Score 5: Clearly stated in both abstract and introduction; specific, measurable, and well-scoped.
   Score 4: Clear in one section (e.g., introduction) but lacks precision or full specificity.
   Score 3: Vague or generic; lacks clear focus.
   Score 2: Unclear or implicit; requires significant inference.
   Score 1: No objectives stated or inferable.

2. Criterion: Relevance
   Score 5: Directly aligns with high-impact, current trends and core issues of the field.
   Score 4: Generally relevant to the broader topic, though not urgent or cutting-edge.
   Score 3: Partially related to the topic; connection is weak or indirect.
   Score 2: Weak or outdated relevance to the field.
   Score 1: Not relevant to the field at all.

3. Criterion: Audience
   Score 5: Clear academic or interdisciplinary targeting; tone and content match intended readers.
   Score 4: Generally appropriate tone, but audience could be more explicitly defined.
   Score 3: Somewhat unclear who the intended audience is.
   Score 2: Confusing or poorly targeted content.
   Score 1: No discernible audience.

---

Category: Literature
4. Criterion: Comprehensiveness
   Score 5: ≥ 30 citations, spanning multiple subfields, including up-to-date and foundational works.
   Score 4: Mostly complete coverage with only minor omissions.
   Score 3: Some omissions or limited to a narrow domain.
   Score 2: Major omissions in key areas.
   Score 1: Sparse or incomplete coverage.

5. Criterion: Balance
   Score 5: Discusses strengths, weaknesses, and multiple perspectives fairly.
   Score 4: Balanced with only minor bias.
   Score 3: Somewhat unbalanced but acknowledges alternative views to some extent.
   Score 2: One-sided; presents only one viewpoint.
   Score 1: Highly biased or promotional.

6. Criterion: Currency
   Score 5: Up-to-date, including recent preprints and conference papers.
   Score 4: Mostly recent, with few older references.
   Score 3: Some outdated dominance; mix of old and new.
   Score 2: Mostly dated content.
   Score 1: Ignores recent developments entirely.

---

Category: Analysis
7. Criterion: Depth
   Score 5: Theoretically rigorous, layered insight; goes beyond description to critical analysis.
   Score 4: Moderate depth with some analytical reasoning.
   Score 3: Descriptive only; lacks critical evaluation.
   Score 2: Minimal or weak analysis.
   Score 1: No meaningful analysis.

8. Criterion: Integration
   Score 5: Seamless integration of multiple perspectives, themes, or frameworks.
   Score 4: Good integration across sections.
   Score 3: Partial integration; some ideas remain siloed.
   Score 2: Mostly disconnected ideas.
   Score 1: Disjointed and fragmented.

9. Criterion: Gaps
   Score 5: Clearly identifies open challenges and unresolved research gaps.
   Score 4: Mentions some gaps, but not fully developed.
   Score 3: Surface-level mention of limitations.
   Score 2: Barely addresses open questions.
   Score 1: Ignores all research gaps.

---

Category: Originality
10. Criterion: Novelty
    Score 5: Introduces a new taxonomy, framework, or domain synthesis.
    Score 4: Novel combination of existing ideas.
    Score 3: Slightly original; minor new perspective.
    Score 2: Mostly derivative.
    Score 1: No original contribution.

11. Criterion: Advancement
    Score 5: Provides strong guidance for future research and practice.
    Score 4: Moderate contribution; advances the field somewhat.
    Score 3: Incremental value.
    Score 2: Minimal progress.
    Score 1: No advancement.

12. Criterion: Redundancy Avoidance
    Score 5: Clearly distinct from prior surveys; avoids overlap.
    Score 4: Mostly unique with minimal overlap.
    Score 3: Moderate overlap with existing work.
    Score 2: Largely redundant.
    Score 1: Highly repetitive.

---

Category: Organization
13. Criterion: Logical Flow
    Score 5: Excellent transitions and well-structured argument flow.
    Score 4: Mostly clear flow.
    Score 3: Basic structure but some issues.
    Score 2: Poor transitions; difficult to follow.
    Score 1: Chaotic and disorganized.

14. Criterion: Section Clarity
    Score 5: Well-labeled, crystal-clear sections and subsections.
    Score 4: Mostly clear.
    Score 3: Confusing or overly long sections.
    Score 2: Unclear or unlabeled structure.
    Score 1: No clear structure.

15. Criterion: Summarization
    Score 5: Effective use of summaries, recaps, and visual aids.
    Score 4: Some synthesis and clear structure.
    Score 3: Minimal synthesis.
    Score 2: Almost no summarization.
    Score 1: No summary or synthesis.

---

Category: Presentation
16. Criterion: Language
    Score 5: Clear, academic language throughout; grammatically flawless.
    Score 4: Mostly well-written with minor issues.
    Score 3: Clumsy tone or occasional grammatical errors.
    Score 2: Poor grammar or clarity issues.
    Score 1: Unreadable or ungrammatical.

17. Criterion: Visuals (figures, tables)
    Score 5: Strong figures/tables that effectively support and enhance content.
    Score 4: Good visuals with minor issues.
    Score 3: Basic visuals, not well-integrated.
    Score 2: Irrelevant or low-quality visuals.
    Score 1: No meaningful visuals.

18. Criterion: Formatting
    Score 5: Clean, consistent, professional styles (headings, citations, spacing).
    Score 4: Minor formatting issues.
    Score 3: Inconsistent formatting.
    Score 2: Distracting formatting problems.
    Score 1: Disorganized formatting.

---

Category: References
19. Criterion: Accuracy
    Score 5: Accurate, traceable, properly formatted references.
    Score 4: Minor format issues; mostly accurate.
    Score 3: Some mismatched or incomplete entries.
    Score 2: Multiple citation errors.
    Score 1: Unreliable or incorrect citations.

20. Criterion: Appropriateness
    Score 5: Highly relevant, current, and foundational sources.
    Score 4: Mostly appropriate with minor filler.
    Score 3: Some irrelevant or filler sources.
    Score 2: Many low-quality sources.
    Score 1: Poor citation quality overall.

---

Final Score: Sum of all 20 criteria (max 100).  
Please also provide a brief overall assessment of the paper's strengths and weaknesses.

You may use your own judgment and knowledge of the field to apply these criteria. You are not restricted to a fixed order or fixed procedure — use whatever approach you find most effective to assess the survey against this rubric.
</evaluation_rubric>

Provide your evaluation in the following JSON format. Do not include any text outside the JSON object.

{{
  "summary": "<2-4 sentence overview of the survey's topic, scope, and overall impression>",
  "strengths": [
    "<strength 1>",
    "<strength 2>"
  ],
  "weaknesses": [
    {{
      "issue": "<concise description of the problem>",
      "location": "<section/paragraph/sentence where this occurs, quoted or paraphrased>",
    }}
  ],
  "comments": [
    {{
      "issue": "<concise description of a less critical observation>",
      "location": "<section/paragraph/sentence where this occurs>"
    }},
  ],
  "rubric_scores": {{
    "<criterion name>": <integer 1-5>,
    "...": "..."
  }}
}}"""

USER_TRUSTSURVEY = """Below is the full text of a survey paper, followed by the TrustSurvey evaluation framework you must use to guide your assessment.

<survey>
{SURVEY_FULL_TEXT}
</survey>

<input_processing_note>
Because input processing may be imperfect, do not report purely format-level or preprocessing artifacts as weaknesses. For example, placeholder author metadata such as "Author Names", reference formatting artifacts, missing rendered figures/tables, or empty-looking algorithm/table boxes caused by input conversion must not appear in the weakness report.
</input_processing_note>

<evaluation_framework>
TrustSurvey evaluates survey trustworthiness through evidence-linked sub-judgments. Do not collapse the assessment into a vague holistic impression. Each reported weakness should correspond to a concrete checkable issue, with a specific location or evidence pointer in the survey whenever possible.

TrustSurvey separates survey-quality concerns by verifiability level:

A-Level: internally verifiable from the survey itself, its cited papers, and codifiable norms of survey writing. A-Level issues can be checked from the document's own structure, citations, claims, and stated promises.
- Gap and future-work discussion insufficient: the survey lacks an explicit discussion of open problems, research gaps, limitations of current work, or future research directions, or includes only a perfunctory mention.
- Comparison insufficient: the survey summarizes works individually but does not explicitly contrast named methods, systems, datasets, or findings along meaningful dimensions.
- Method evaluation insufficient: the survey does not provide a dedicated or systematic evaluation-oriented discussion, such as benchmark-based comparison, metric discussion, performance analysis, or empirical comparison.
- Synthesis / original viewpoint insufficient: the survey mainly lists prior work and lacks cross-paper synthesis, taxonomy, trend analysis, organizing abstractions, or an explicit authorial perspective.
- Scope / inclusion-criteria declaration missing: the survey does not make its coverage boundary inspectable, e.g., by stating search strategy, inclusion or exclusion criteria, time span, venue scope, language scope, or topic exclusions.
- Contribution statement missing: the survey does not explicitly state what it contributes as a survey, such as a taxonomy, synthesis, organizing framework, coverage boundary, or practical guidance.
- Internal inconsistency: the survey makes a scope, contribution, section-title, or organizational promise that is contradicted or not substantively fulfilled by the body content.
- Hallucination, internal/cited-source side: the survey contains non-existent citations, misattributes a cited paper, or makes a claim about a cited source that does not match that source.

B-Level: externally verifiable given a suitable literature pool. B-Level issues require external evidence, but the obligation should still be determinate rather than a matter of taste.
- Hallucination, external side: the survey makes an uncited factual claim that is contradicted by relevant literature.
- Missing specific references: the survey omits a specific reference recoverable from external evidence, including reference surveys, uncited named research objects, or subtopic-relevant landmark papers.
- Missing specific topics: the survey omits a content category or method family recoverable from the external literature pool and not explicitly scoped out.

C-Level: disagreement-prone even with complete evidence. These issues are outside TrustSurvey's automated scope and should not be treated as primary detected weaknesses unless clearly grounded in A/B evidence.
- Suggestions on adding references without a determinate omission.
- Taxonomy or framework preference problems.
- Evidence depth or argument support judgments that require graded expert taste.
- Writing clarity, presentation, visualization quality, contribution novelty, or venue-fit judgments.

Evaluate the survey using A-Level and B-Level checks. For each weakness, prefer evidence-linked, itemized findings over broad commentary. If you mention a C-Level concern, put it in comments rather than weaknesses unless it is tied to a concrete A/B-Level violation.
</evaluation_framework>

Provide your evaluation in the following JSON format. Do not include any text outside the JSON object.

{
  "summary": "<2-4 sentence overview of the survey's topic, scope, and overall impression>",
  "strengths": [
    "<strength 1>",
    "<strength 2>"
  ],
  "weaknesses": [
    {
      "issue": "<concise description of the A-Level or B-Level problem>",
      "issue_type": "<>"
      "location": "<section/paragraph/sentence where this occurs, quoted or paraphrased>",
      "evidence": "<specific internal or external evidence supporting the finding>",
    }
  ],
  "comments": [
    {
      "issue": "<concise description of a less critical or C-Level observation>",
      "level": "A" | "B" | "C",
      "location": "<section/paragraph/sentence where this occurs>"
    }
  ],
  "overall_score": <integer 0-100>,
  "score_justification": "<1-3 sentences explaining the score>"
}"""
