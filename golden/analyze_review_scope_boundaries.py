import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


from survey_eval.agent.tools.utility.evidence_check import EvidenceCheck
from survey_eval.agent.tools.utility.llmclient import AsyncChat
from survey_eval.agent.tools.utility.request_utils import SessionManager
from survey_eval.agent.tools.utility.tool_config import ToolConfig
from survey_eval.agent.tools.utility.utils import extract_json


FILES = {
    "venue": "venue_contribution_mismatch.jsonl",
    "writing": "writing_clarity_problem.jsonl",
    "visualization": "missing_visualization.jsonl",
}

ROUTES = {
    "internal_inconsistency",
    "factual_hallucination_or_technical_error",
    "taxonomy_framework_problem",
    "evidence_support_insufficient",
    "future_work_or_limitation_missing",
    "comparison_analysis_insufficient",
    "synthesis_depth_insufficient",
    "methodology_transparency_insufficient",
    "contribution_novelty_insufficient",
    "missing_specific_references",
    "missing_specific_topics",
    "none",
}

SCOPES = {
    "ROUTE_EXISTING",
    "STANDALONE_CANDIDATE",
    "RESIDUAL_OUT_OF_SCOPE",
    "AMBIGUOUS",
}

CLUSTERS = {
    "venue": {
        "PURE_VENUE_FIT",
        "SURVEY_GENRE_FIT",
        "NOVELTY_OR_VALUE_ADD",
        "AUDIENCE_RELEVANCE",
        "BREADTH_DEPTH_OR_RIGOR",
        "METHOD_OR_EVIDENCE_EXPECTATION",
        "MISCLASSIFIED_OTHER",
        "MIXED",
    },
    "writing": {
        "COPYEDITING_OR_FORMATTING",
        "LOCAL_SEMANTIC_AMBIGUITY",
        "UNDEFINED_TERM_OR_NOTATION",
        "ORGANIZATION_OR_FLOW",
        "INSUFFICIENT_EXPLANATION",
        "CLAIM_SCOPE_OR_OBJECTIVE_UNCLEAR",
        "TECHNICAL_OR_FACTUAL_ERROR",
        "INTERNAL_INCONSISTENCY",
        "CITATION_OR_ATTRIBUTION",
        "MISCLASSIFIED_OTHER",
        "MIXED",
    },
    "visualization": {
        "MISSING_OVERVIEW_OR_TAXONOMY_FIGURE",
        "MISSING_COMPARISON_TABLE",
        "MISSING_EXPLANATORY_DIAGRAM_OR_EXAMPLE",
        "FIGURE_OR_TABLE_READABILITY",
        "FIGURE_OR_TABLE_INCORRECT_OR_INCONSISTENT",
        "FIGURE_OR_TABLE_CONTENT_INCOMPLETE",
        "PURE_AESTHETIC_PREFERENCE",
        "MISCLASSIFIED_OTHER",
        "MIXED",
    },
}

PROMPT = """### Task
You are coding peer-review concerns to determine the construct boundary of a survey-auditing system. Classify every supplied record into exactly one cluster. Then decide whether the concern should be routed to one of the existing audit categories, retained as a standalone trustworthiness candidate, treated as a residual venue/style/presentation preference outside the construct, or marked ambiguous. Do not infer a defect that the reviewer did not state.

### Scope labels
ROUTE_EXISTING: the concern identifies a truth, consistency, evidence, genre-obligation, citation, or literature-coverage defect already represented by an existing route. STANDALONE_CANDIDATE: the concern affects reliable interpretation or auditability but is not adequately represented by an existing route. RESIDUAL_OUT_OF_SCOPE: the concern is only venue preference, copyediting, aesthetics, or an optional presentation suggestion. AMBIGUOUS: the quoted review is too underspecified to decide.

### Existing routes
internal_inconsistency; factual_hallucination_or_technical_error; taxonomy_framework_problem; evidence_support_insufficient; future_work_or_limitation_missing; comparison_analysis_insufficient; synthesis_depth_insufficient; methodology_transparency_insufficient; contribution_novelty_insufficient; missing_specific_references; missing_specific_topics; none.

### Cluster definitions
{cluster_definitions}

### Requirements
Return one result for every input id and preserve the ids exactly. The script retains the original evidence verbatim by id, so do not repeat or paraphrase it. Use route_to=none unless scope=ROUTE_EXISTING. For mixed comments, choose MIXED only when no single cluster captures the principal concern; route_to should identify the principal existing audit route when one exists.

### Output
Return JSON only: {{"results":[{{"id":0,"cluster":"CLUSTER","scope":"ROUTE_EXISTING|STANDALONE_CANDIDATE|RESIDUAL_OUT_OF_SCOPE|AMBIGUOUS","route_to":"route-or-none","rationale":"one concise sentence"}}]}}
"""

CLUSTER_DEFINITIONS = {
    "venue": "PURE_VENUE_FIT: the named venue is simply judged inappropriate; SURVEY_GENRE_FIT: survey/review papers as a contribution type are judged unsuitable; NOVELTY_OR_VALUE_ADD: insufficient novelty, insight, or value beyond prior surveys; AUDIENCE_RELEVANCE: insufficient interest or relevance to the venue readership; BREADTH_DEPTH_OR_RIGOR: concrete criticism of survey breadth, depth, unification, or rigor; METHOD_OR_EVIDENCE_EXPECTATION: concrete demand for experiments, meta-analysis, or supported claims; MISCLASSIFIED_OTHER: the evidence principally belongs to another existing audit category; MIXED: inseparable combination of multiple clusters.",
    "writing": "COPYEDITING_OR_FORMATTING: grammar, typos, citation style, typography, or proofreading; LOCAL_SEMANTIC_AMBIGUITY: a sentence, phrase, referent, or relation cannot be interpreted; UNDEFINED_TERM_OR_NOTATION: terminology, acronym, symbol, equation component, or category lacks a definition; ORGANIZATION_OR_FLOW: ordering, section structure, density, or narrative flow; INSUFFICIENT_EXPLANATION: a method, claim, equation, result, or concept is not explained enough to understand or assess it; CLAIM_SCOPE_OR_OBJECTIVE_UNCLEAR: the survey's objectives, scope, message, or main claims are unclear; TECHNICAL_OR_FACTUAL_ERROR: the apparent clarity complaint actually identifies a technical or factual error; INTERNAL_INCONSISTENCY: the evidence identifies a contradiction or inconsistent terminology/formalism; CITATION_OR_ATTRIBUTION: the issue concerns missing, malformed, or ambiguous attribution; MISCLASSIFIED_OTHER: the evidence principally belongs to another existing audit category; MIXED: inseparable combination of multiple clusters.",
    "visualization": "MISSING_OVERVIEW_OR_TAXONOMY_FIGURE: request for a global overview, taxonomy, roadmap, or conceptual summary; MISSING_COMPARISON_TABLE: request for a table/plot comparing methods, datasets, results, or trade-offs; MISSING_EXPLANATORY_DIAGRAM_OR_EXAMPLE: request for a figure/example explaining a mechanism, architecture, workflow, or concept; FIGURE_OR_TABLE_READABILITY: labels, resolution, layout, caption, density, or legibility; FIGURE_OR_TABLE_INCORRECT_OR_INCONSISTENT: visual content contradicts text, data, taxonomy, or its stated purpose; FIGURE_OR_TABLE_CONTENT_INCOMPLETE: an existing visual omits information needed to interpret its content; PURE_AESTHETIC_PREFERENCE: visual appearance or optional illustration without a concrete auditability deficit; MISCLASSIFIED_OTHER: the evidence principally belongs to another existing audit category; MIXED: inseparable combination of multiple clusters.",
}


class ScopeBoundaryCoder(AsyncChat):
    def __init__(self, config: ToolConfig, family: str):
        super().__init__(config.llm_server_info, config.sampling_params)
        self.family = family
        self.check = EvidenceCheck(config)

    def _organize_inputs(self, inputs):
        records = inputs["records"]
        prompt = PROMPT.format(cluster_definitions=CLUSTER_DEFINITIONS[self.family])
        payload = json.dumps(records, ensure_ascii=False)
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ], {"records": records}

    def _availability(self, response, context):
        data = extract_json(response)
        results = data["results"]
        records = {item["id"]: item for item in context["records"]}
        assert len(results) == len(records)
        assert {item["id"] for item in results} == set(records)
        for item in results:
            assert item["cluster"] in CLUSTERS[self.family]
            assert item["scope"] in SCOPES
            assert item["route_to"] in ROUTES
            if item["scope"] == "ROUTE_EXISTING":
                assert item["route_to"] != "none"
            else:
                assert item["route_to"] == "none"
        return results


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def classify_family(config, family, records, batch_size, concurrency):
    coder = ScopeBoundaryCoder(config, family)
    semaphore = asyncio.Semaphore(concurrency)

    async def classify_batch(batch):
        async with semaphore:
            return await coder.call(inputs={"records": batch}, max_tokens=8192)

    batches = []
    for start in range(0, len(records), batch_size):
        batch = []
        for index, record in enumerate(records[start:start + batch_size], start=start):
            batch.append({"id": index, "evidence": record["evidence"]})
        batches.append(asyncio.create_task(classify_batch(batch)))

    coded = {}
    for task in asyncio.as_completed(batches):
        for item in await task:
            coded[item["id"]] = item

    assert len(coded) == len(records)
    return [{**record, **coded[index]} for index, record in enumerate(records)]


def summarize(items):
    return {
        "records": len(items),
        "clusters": dict(Counter(item["cluster"] for item in items).most_common()),
        "scope": dict(Counter(item["scope"] for item in items).most_common()),
        "routes": dict(Counter(item["route_to"] for item in items).most_common()),
        "cluster_by_scope": {
            cluster: dict(Counter(item["scope"] for item in items if item["cluster"] == cluster).most_common())
            for cluster in sorted({item["cluster"] for item in items})
        },
    }


async def main_async(args):
    config = ToolConfig.from_yaml(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    all_summaries = {}
    await SessionManager.init()
    try:
        selected = set(args.families.split(",")) if args.families else set(FILES)
        for family, filename in FILES.items():
            if family not in selected:
                continue
            records = read_jsonl(input_dir / filename)
            coded = await classify_family(config, family, records, args.batch_size, args.concurrency)
            write_json(output_dir / f"{Path(filename).stem}_scope_clusters.json", coded)
            all_summaries[family] = summarize(coded)
            print(f"{family}: {len(coded)} records")
        write_json(output_dir / "scope_cluster_summary.json", all_summaries)
    finally:
        await SessionManager.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=REPO_ROOT / "agent.yaml", type=Path)
    parser.add_argument("--input-dir", default=Path(__file__).parent / "review_analyze")
    parser.add_argument("--output-dir", default=Path(__file__).parent / "review_analyze" / "scope_clusters")
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--concurrency", default=4, type=int)
    parser.add_argument("--families", default="")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
