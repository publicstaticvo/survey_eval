SYSTEM = """You are an expert reviewer with broad knowledge of machine learning and natural language processing research. You will be given the full text of a survey paper. Your task is to assess its overall quality."""

CC_PROMPT = """You are an expert reviewer with broad knowledge of machine learning and natural language processing research. The survey paper to be evaluated is in {input_dir}. Your task is to assess its overall quality. 

{requirements}

Write your output as a JSON file in {output_file}."""

USER_PLAIN = """Below is the full text of a survey paper. Read it carefully and evaluate its quality.

<survey>
{SURVEY_FULL_TEXT}
</survey>

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
      "severity": "major" | "minor"
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
      "severity": "major" | "minor"
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

USER_TRUSTSURVEY = """Below is the full text of a survey paper, followed by an evaluation framework you must use to guide your assessment.

<survey>
{SURVEY_FULL_TEXT}
</survey>

<evaluation_framework>
This framework defines four levels of survey quality. For each level, consider the listed concerns. These concerns are derived from an analysis of 525 real peer review comments on 170 survey papers.

LEVEL 1 — MINIMAL VALIDITY
- Does the paper exhibit the structure and intent of a literature survey (as opposed to a method/research paper)?
- Does it state a clear review methodology (search strategy, inclusion/exclusion criteria)?
- Does every major section engage substantively with cited literature, rather than reading as a bare list?
- Is there a discussion of future directions or open problems?

LEVEL 2 — FACTUAL INTEGRITY
- Do all citations refer to papers that actually exist?
- Are citations made to the original/canonical version of a work (e.g., published version rather than an outdated preprint), where applicable?
- Does every factual claim attributed to a cited work actually match what that work says? Flag any claim that misrepresents, exaggerates, or contradicts its cited source.

LEVEL 3 — STRUCTURAL INTEGRITY
- Does the survey's stated contribution/scope (as declared in the abstract and introduction) match what the paper actually delivers?
- Does each section's content match what its title promises?
- Are there internal contradictions between what is claimed and what is presented elsewhere in the paper?

LEVEL 4 — DOMAIN INTEGRITY
- Are there important, well-known works in this area that are conspicuously missing from the citation list?
- Are there named methods, concepts, or claims introduced without any supporting citation?
- Are there important subtopics of this field (e.g., datasets/benchmarks, real-world deployment, ethical/safety considerations) that are entirely absent, without explanation?
- Does the survey's own comparative or synthesizing discussion imply the existence of topics or categories that are never actually discussed?

You may use your own judgment and knowledge of the field to apply these criteria. You are not restricted to a fixed order or fixed procedure — use whatever approach you find most effective to assess the survey against this framework.
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
      "issue": "<concise description of the problem>",
      "location": "<section/paragraph/sentence where this occurs, quoted or paraphrased>",
      "severity": "major" | "minor"
    }
  ],
  "comments": [
    {
      "issue": "<concise description of a less critical observation>",
      "location": "<section/paragraph/sentence where this occurs>"
    }
  ],
  "overall_score": <integer 0-100>,
  "score_justification": "<1-3 sentences explaining the score>"
}"""

