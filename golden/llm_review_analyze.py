import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from survey_eval.agent.tools.utility.request_utils import SessionManager
from survey_eval.agent.tools.utility.evidence_check import EvidenceCheck
from survey_eval.agent.tools.utility.tool_config import ToolConfig
from survey_eval.agent.tools.utility.llmclient import AsyncChat
from survey_eval.agent.tools.utils import extract_json
import asyncio
import json

PROBLEM_CATEGORIES = [
    "missing_specific_references", 
    "missing_specific_topics", 
    "references_insufficient",
    "coverage_insufficient",
    "contribution_novelty_insufficient", 
    "synthesis_depth_insufficient", 
    "comparison_analysis_insufficient",
    "methodology_transparency_insufficient", 
    "future_work_or_limitation_missing", 
    "internal_inconsistency",
    "taxonomy_framework_problem", 
    "factual_hallucination_or_technical_error", 
    "evidence_support_insufficient",  
    "writing_clarity_problem", 
    "venue_contribution_mismatch",
    "missing_visualization"
]
PROMPT = """You are an expert NLP annotator specializing in academic peer review analysis. Your task is to analyze a survey paper review and identify which of the 16 predefined weakness categories are mentioned by the reviewer.

## Task

Read the review and identify ALL problems the reviewer mentions. For each problem found, assign one category from the list below and quote the exact text as evidence. If the review mentions any weakness that does not belong to these 16 categories, label it as "Others" and provide the corresponding evidence.

## Categories
1. missing_specific_references — reviewer points out that specific papers, works, or authors are missing from the survey's citations or reference list
2. missing_specific_topics — reviewer points out that specific topics, sub-areas, directions, or recent developments are absent or insufficiently covered
3. references_insufficient — reviewer indicates there are too few citations, but does NOT name the titles of specific papers that should be included
4. coverage_insufficient — reviewer indicates the coverage is too narrow or incomplete, but does NOT name specific missing topics, directions, or sub-areas
5. contribution_novelty_insufficient — reviewer argues the survey lacks a new perspective, organizational framework, or incremental value compared to existing surveys on the same topic
6. synthesis_depth_insufficient — reviewer notes the survey merely lists or describes papers without integrating findings, identifying patterns, or deriving higher-level insights across works
7. comparison_analysis_insufficient — reviewer notes the absence of systematic cross-work comparison, quantitative performance analysis, or clear positioning of methods relative to each other
8. methodology_transparency_insufficient — reviewer finds the survey's own methodology unclear, including search strategy, inclusion/exclusion criteria, paper selection rationale, or annotation protocol
9. future_work_or_limitation_missing — reviewer notes missing or superficial treatment of limitations, open problems, research gaps, or future directions
10. internal_inconsistency — reviewer identifies contradictions between the stated scope and actual content, between different sections, or within the survey's own arguments or taxonomy
11. taxonomy_framework_problem — reviewer finds the survey's classification scheme, organizational framework, or category definitions flawed, overlapping, unjustified, or internally inconsistent
12. factual_hallucination_or_technical_error — reviewer identifies technical errors, factual inaccuracies, misattributed findings, or claims that contradict the cited sources
13. evidence_support_insufficient — reviewer finds that conclusions, generalizations, or evaluative claims lack adequate supporting evidence or logical justification
14. writing_clarity_problem — reviewer criticizes unclear writing, vague expression, poor sentence structure, ambiguous terminology, or difficulty following the text
15. venue_contribution_mismatch — reviewer argues the paper's scope, depth, or contribution type does not meet the expectations of the target venue
16. missing_visualization — reviewer notes the absence or insufficiency of figures, tables, diagrams, or other visual aids that would aid comprehension

## Special Extraction Rules
- For missing_specific_references: Extract the specific missing paper titles the reviewer names. Return them as a list of strings in the `missed_references` field within the same problem item.
- For missing_specific_topics: Extract the specific missing topics, directions, or sub-areas the reviewer names. Return them as a list of strings in the `missed_topics` field within the same problem item.

## Output format
Return JSON only:

{
  "problems": [
    {
      "category_id": <int>,
      "category_name": <str>,
      "evidence": "<exact quote from the review, copied verbatim>",
      "missed_references": [<str>, ...],
      "missed_topics": [<str>, ...]
    }
  ]
}

## Notes on output fields:
- `missed_references` is required when category_name is "missing_specific_references". Otherwise, omit this field or set it to [].
- `missed_topics` is required when category_name is "missing_specific_topics". Otherwise, omit this field or set it to [].
- Both fields must be placed inside each individual problem item in the problems array.
- If no clear problem is mentioned, return `"problems": []`. Do not infer problems not explicitly stated in the review.
"""
DATA_DIR = Path("data")
FIELDS = {"strengths", "weaknesses", "questions", "requested_changes", "Relecture",
          'strengths_and_weaknesses', 'reasons_to_reject', 'reason_to_accept', "questions_for_the_authors", 
          'Reasons_to_reject', 'Reason_to_accept', "Questions_for_the_Authors"}


class ReviewAnalyze(AsyncChat):
    def __init__(self, config: ToolConfig):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.check = EvidenceCheck(config)

    def _availability(self, response, context):
        try:
            data = extract_json(response)
        except json.JSONDecodeError:
            print(response)
            raise
        for x in data['problems']:
            if x['category_name'].lower() != "others":
                assert x['category_name'] == PROBLEM_CATEGORIES[x['category_id'] - 1], (x['category_id'], x['category_name'])
            assert self.check.verify([x['evidence']], context['inputs']), x['evidence']
            if x['category_name'] == "missing_specific_references":
                assert x['missed_references']
                for y in x['missed_references']: assert self.check.verify([y], context['inputs']), y
        return {"id": context['id'], "problems": data['problems']}

    def _organize_inputs(self, inputs):
        return [{"role": 'system', 'content': PROMPT}, {'role': 'user', 'content': inputs['inputs']}], {**inputs}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def iter_input_files() -> list[Path]:
    return sorted(path for path in DATA_DIR.glob("*.json") if not path.name.startswith("_"))
    

async def _process_single_paper(inputs: Path):
    inputs = read_json(inputs)
    llm = ReviewAnalyze(ToolConfig())
    tasks = [asyncio.create_task(llm.call(inputs={"inputs": x['text'], 'id': x['review_id']})) \
             for x in inputs['reviews'] if any(y in x['content'] for y in FIELDS)]
    reviews = {}
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            if result: reviews[result['id']] = result['problems']
        except Exception as e:
            print(f"ReviewAnalyze {e}")
    per_review_count = {}
    for p in reviews.values():
        for x in p:
            per_review_count[x['category_name']] = per_review_count.get(x['category_name'], 0) + 1
    return reviews, per_review_count


async def main_async():
    await SessionManager.init()
    try:
        files = iter_input_files()
        tasks = [asyncio.create_task(_process_single_paper(path)) for path in files]
        results = await asyncio.gather(*tasks)
        reviews, per_paper_count, per_review_count = [], {}, {}
        for x in results:
            try:
                a, b = x
                reviews.append(a)
                for k, v in b.items(): 
                    per_paper_count[k] = per_paper_count.get(k, 0) + 1
                    per_review_count[k] = per_review_count.get(k, 0) + v
            except Exception as e: 
                print(e)
        print(per_paper_count, per_review_count)
        write_json(Path("review_analyze.json"), reviews)
    finally:
        await SessionManager.close()    


if __name__ == "__main__":
    asyncio.run(main_async())