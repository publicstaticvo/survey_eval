from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools.preprocess.get_reference_surveys import GetReferenceSurveys
from agent.tools.utility.academic_engine import get_academic_engine
from agent.tools.utility.request_utils import SessionManager
from agent.tools.utility.tool_config import ToolConfig
from agent.tools.utility.utils import normalize_text
from golden.r5 import QUERY_BY_PAPER_ID, resolve_reference_paper


GOLDEN_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = GOLDEN_DIR / "R5-openalex.jsonl"
DETAIL_OUT = GOLDEN_DIR / "R5-missing-reference-analysis.jsonl"
SUMMARY_OUT = GOLDEN_DIR / "R5-missing-reference-analysis-summary.json"
REFERENCE_SURVEY_OUT = GOLDEN_DIR / "R5-reference-survey-unseen-coverage.jsonl"
RESOLVED_MISSING_CACHE = GOLDEN_DIR / "R5-resolved-missing-references.jsonl"
TRUE_MISSING_LABELS = {"cites", "cited_by", "unseen"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_engine(input_path: Path) -> str:
    name = input_path.name.lower()
    if "openalex" in name:
        return "openalex"
    return "semantic_scholar"


def parse_date(value: Any) -> date | None:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            return None
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            if fmt == "%Y":
                return datetime.strptime(text[:4], fmt).date()
            return datetime.strptime(text[: len(fmt)], fmt).date()
        except ValueError:
            continue
    return None


def publication_date(paper: dict[str, Any]) -> date | None:
    return parse_date(paper.get("publication_date") or paper.get("publicationDate") or paper.get("year"))


def submission_date(record: dict[str, Any]) -> date | None:
    return parse_date(record.get("submission_cdate") or record.get("submission_date"))


def load_pdf_content_submission_dates() -> dict[int, dict[str, Any]]:
    dates = {}
    for path in (GOLDEN_DIR / "pdf_content").glob("*.json"):
        match = re.match(r"^(\d{3})_", path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        paper_id = int(match.group(1))
        dates[paper_id] = {
            "submission_date": data.get("submission_date") or "",
            "submission_cdate": data.get("submission_cdate"),
            "pdf_content_path": str(path),
        }
    return dates


def paper_submission_date(paper_id: int, record: dict[str, Any], pdf_dates: dict[int, dict[str, Any]]) -> date | None:
    from_pdf = pdf_dates.get(paper_id) or {}
    return parse_date(from_pdf.get("submission_cdate") or from_pdf.get("submission_date")) or submission_date(record)


def normalize_id(value: Any) -> str:
    return str(value or "").replace("https://openalex.org/", "").strip()


def paper_ids(paper: dict[str, Any]) -> set[str]:
    ids = set()
    for key in ("id", "paperId", "corpusId"):
        if paper.get(key):
            ids.add(normalize_id(paper[key]))
    raw_ids = paper.get("ids")
    if isinstance(raw_ids, dict):
        ids.update(normalize_id(value) for value in raw_ids.values() if value)
    elif isinstance(raw_ids, (list, tuple, set)):
        ids.update(normalize_id(value) for value in raw_ids if value)
    return {item for item in ids if item}


def reference_key(reference: dict[str, Any]) -> str:
    title = normalize_text(reference.get("title", ""))
    location = reference.get("location") or {}
    return json.dumps(
        {
            "title": title,
            "location_type": location.get("type") or "",
            "location_link": normalize_text(location.get("link", "")),
        },
        sort_keys=True,
    )


def citation_count_by_eval_date(paper: dict[str, Any], eval_date: date | None) -> int | None:
    total = paper.get("cited_by_count")
    if total is None:
        total = paper.get("citationCount")
    if total is None:
        return None
    try:
        total = int(total or 0)
    except (TypeError, ValueError):
        return None
    if eval_date is None:
        return total
    counts_by_year = paper.get("counts_by_year") or []
    future_count = 0
    for item in counts_by_year:
        try:
            year = int(item.get("year"))
            count = int(item.get("cited_by_count") or item.get("citationCount") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if year > eval_date.year:
            future_count += count
    return max(0, total - future_count)


def load_resolved_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for item in read_jsonl(path):
        key = item.get("reference_key")
        if key:
            cache[key] = item.get("paper") or {}
    return cache


def write_resolved_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    records = [
        {"reference_key": key, "paper": paper}
        for key, paper in sorted(cache.items())
        if paper
    ]
    write_jsonl(path, records)


def load_unique_paper_records(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        if record.get("record_type") == "overall":
            continue
        paper_id = record.get("paper_id")
        if paper_id is None or paper_id in records:
            continue
        records[int(paper_id)] = record
    return records


def true_missing_references(record: dict[str, Any]) -> list[dict[str, Any]]:
    seen = set()
    refs = []
    for reference in record.get("paper_missing_references") or []:
        label = reference.get("literature_pool_label") or "unseen"
        if label == "cited_papers":
            continue
        key = reference_key(reference)
        if key in seen:
            continue
        seen.add(key)
        output = dict(reference)
        output["literature_pool_label"] = label if label in TRUE_MISSING_LABELS else "unseen"
        refs.append(output)
    return refs


def match_reference(reference: dict[str, Any], paper: dict[str, Any]) -> bool:
    ref_title = reference.get("title") or ""
    paper_title = paper.get("title") or ""
    if ref_title and title_match(ref_title, paper_title):
        return True
    location = reference.get("location") or {}
    link = normalize_text(location.get("link", ""))
    if not link:
        return False
    for value in paper_ids(paper):
        if normalize_text(value) == link:
            return True
    external_ids = paper.get("external_ids") or paper.get("externalIds") or {}
    return any(normalize_text(value) == link for value in external_ids.values() if value)


def title_match(left: str, right: str) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return False
    shorter, longer = sorted((left_norm, right_norm), key=len)
    if shorter in longer and len(shorter) / max(1, len(longer)) >= 0.75:
        return True
    distance = Levenshtein.distance(left_norm, right_norm)
    return distance <= max(1, int(0.1 * max(len(left_norm), len(right_norm))))


def percentile_at_or_below(values: list[Any], target: Any) -> float | None:
    valid = [value for value in values if value is not None]
    if target is None or not valid:
        return None
    return 100.0 * sum(1 for value in valid if value <= target) / len(valid)


def load_literature_pool(path: Path, eval_date: date | None) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    pool = []
    distributions = {"publication_date": [], "citation_count_by_eval_date": []}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            paper = item.get("paper") or {}
            pool_key = item.get("pool_key") or f"row:{len(pool)}"
            pub_date = publication_date(paper)
            citation_count = citation_count_by_eval_date(paper, eval_date)
            item["_pool_key"] = pool_key
            item["_publication_date"] = pub_date
            item["_citation_count_by_eval_date"] = citation_count
            pool.append(item)
            distributions["publication_date"].append(pub_date)
            distributions["citation_count_by_eval_date"].append(citation_count)
    return pool, distributions


def summarize_numbers(values: list[int | float | None]) -> dict[str, Any]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(numeric),
        "mean": mean(numeric),
        "median": median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


async def resolve_reference_metadata(
    reference: dict[str, Any],
    engine: Any,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = reference_key(reference)
    if key in cache:
        return cache[key]
    try:
        paper = await resolve_reference_paper(reference, engine)
    except Exception:
        paper = None
    paper = paper or {}
    cache[key] = paper
    return paper


async def build_local_analysis(
    input_path: Path,
    engine_name: str,
    config: ToolConfig,
    resolved_cache_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_unique_paper_records(input_path)
    details = []
    pdf_dates = load_pdf_content_submission_dates()
    engine = get_academic_engine(config)
    resolved_cache = load_resolved_cache(resolved_cache_path)
    pool_dir = GOLDEN_DIR / f"{engine_name}_literature_pools"

    for paper_id, record in sorted(records.items()):
        refs = true_missing_references(record)
        eval_date = paper_submission_date(paper_id, record, pdf_dates)
        pdf_submission = pdf_dates.get(paper_id) or {}
        pool_path = pool_dir / f"{paper_id:03d}.jsonl"
        if not pool_path.exists():
            for reference in refs:
                resolved_paper = await resolve_reference_metadata(reference, engine, resolved_cache)
                pub_date = publication_date(resolved_paper)
                citation_count = citation_count_by_eval_date(resolved_paper, eval_date)
                details.append({
                    "paper_id": paper_id,
                    "query": record.get("query") or QUERY_BY_PAPER_ID.get(paper_id, ""),
                    "submission_date": eval_date.isoformat() if eval_date else "",
                    "submission_date_source": "pdf_content" if pdf_submission else "r5_record",
                    "literature_pool_label": reference["literature_pool_label"],
                    "missing_reference": reference,
                    "resolved_missing_title": resolved_paper.get("title", ""),
                    "publication_date": pub_date.isoformat() if pub_date else "",
                    "publication_minus_submission_days": (pub_date - eval_date).days if pub_date and eval_date else None,
                    "citation_count_by_eval_date": citation_count,
                    "pool_error": f"missing literature pool: {pool_path}",
                })
            continue

        pool, distributions = load_literature_pool(pool_path, eval_date)
        for reference in refs:
            matched = next((item for item in pool if match_reference(reference, item.get("paper") or {})), None)
            resolved_paper = (matched or {}).get("paper") or await resolve_reference_metadata(reference, engine, resolved_cache)
            pub_date = publication_date(resolved_paper)
            citation_count = citation_count_by_eval_date(resolved_paper, eval_date)
            pool_key = (matched or {}).get("_pool_key")
            pub_diff_days = (pub_date - eval_date).days if pub_date and eval_date else None
            publication_percentile = percentile_at_or_below(distributions["publication_date"], pub_date)
            citation_percentile = percentile_at_or_below(distributions["citation_count_by_eval_date"], citation_count)
            details.append({
                "paper_id": paper_id,
                "query": record.get("query") or QUERY_BY_PAPER_ID.get(paper_id, ""),
                "submission_date": eval_date.isoformat() if eval_date else "",
                "submission_date_source": "pdf_content" if pdf_submission else "r5_record",
                "literature_pool_label": reference["literature_pool_label"],
                "missing_reference": reference,
                "matched_pool_key": pool_key,
                "matched_pool_label": (matched or {}).get("label"),
                "matched_pool_title": ((matched or {}).get("paper") or {}).get("title", ""),
                "resolved_missing_title": resolved_paper.get("title", ""),
                "publication_date": pub_date.isoformat() if pub_date else "",
                "publication_minus_submission_days": pub_diff_days,
                "publication_date_percentile_in_pool": publication_percentile,
                "citation_count_by_eval_date": citation_count,
                "citation_count_percentile_in_pool": citation_percentile,
                "literature_pool_size": len(pool),
                "literature_pool_publication_date_count": sum(value is not None for value in distributions["publication_date"]),
                "literature_pool_citation_count": sum(value is not None for value in distributions["citation_count_by_eval_date"]),
            })

    write_resolved_cache(resolved_cache_path, resolved_cache)
    by_label = defaultdict(list)
    for item in details:
        by_label[item["literature_pool_label"]].append(item)
    summary = {
        "input_path": str(input_path),
        "engine": engine_name,
        "true_missing_label_frequency": dict(sorted(Counter(item["literature_pool_label"] for item in details).items())),
        "num_unique_papers": len(records),
        "num_true_missing_references": len(details),
        "percentile_semantics": {
            "publication_date_percentile_in_pool": "0-100; percentage of literature-pool papers with publication_date <= the missing paper publication_date. Higher means newer relative to the pool.",
            "citation_count_percentile_in_pool": "0-100; percentage of literature-pool papers with citation_count_by_eval_date <= the missing paper citation count. Higher means more cited relative to the pool.",
        },
        "by_label": {
            label: {
                "count": len(items),
                "publication_minus_submission_days": summarize_numbers([item.get("publication_minus_submission_days") for item in items]),
                "publication_date_percentile_in_pool": summarize_numbers([item.get("publication_date_percentile_in_pool") for item in items]),
                "citation_count_by_eval_date": summarize_numbers([item.get("citation_count_by_eval_date") for item in items]),
                "citation_count_percentile_in_pool": summarize_numbers([item.get("citation_count_percentile_in_pool") for item in items]),
                "matched_in_literature_pool": sum(1 for item in items if item.get("matched_pool_key")),
                "resolved_missing_metadata": sum(1 for item in items if item.get("resolved_missing_title")),
            }
            for label, items in sorted(by_label.items())
        },
    }
    return details, summary


async def reference_survey_coverage(details: list[dict[str, Any]], config: ToolConfig) -> list[dict[str, Any]]:
    engine = get_academic_engine(config)
    getter = GetReferenceSurveys(config)
    by_paper_id = defaultdict(list)
    for item in details:
        if item.get("literature_pool_label") == "unseen":
            by_paper_id[item["paper_id"]].append(item)

    records = []
    for paper_id, unseen_items in sorted(by_paper_id.items()):
        query = unseen_items[0].get("query") or QUERY_BY_PAPER_ID.get(paper_id, "")
        surveys = await getter(query)
        reference_surveys = surveys.get("reference_surveys") or []
        reference_papers = []
        for survey in reference_surveys:
            meta = survey.get("openalex") or survey.get("semantic_scholar") or {}
            survey_ids = list(paper_ids(meta))
            refs = []
            for survey_id in survey_ids:
                try:
                    payload = await engine.get_references(survey_id, limit=9999)
                except TypeError:
                    payload = await engine.get_references(survey_id, limit=9999, fields=None)
                except Exception as exc:
                    records.append({
                        "paper_id": paper_id,
                        "query": query,
                        "record_type": "reference_survey_error",
                        "reference_survey_title": meta.get("title", ""),
                        "error": str(exc),
                    })
                    continue
                refs = payload.get("results") or []
                if refs:
                    break
            reference_papers.extend(refs)

        for item in unseen_items:
            reference = item.get("missing_reference") or {}
            covered = any(match_reference(reference, ref) for ref in reference_papers)
            records.append({
                "paper_id": paper_id,
                "query": query,
                "record_type": "unseen_reference_coverage",
                "missing_reference": reference,
                "in_reference_survey_references": covered,
                "reference_survey_count": len(reference_surveys),
                "reference_survey_reference_count": len(reference_papers),
            })
    frequency = Counter(
        "in" if item.get("in_reference_survey_references") else "not_in"
        for item in records
        if item.get("record_type") == "unseen_reference_coverage"
    )
    records.insert(0, {
        "record_type": "overall",
        "unseen_reference_survey_reference_frequency": dict(sorted(frequency.items())),
    })
    return records


def config_from_args(args: argparse.Namespace) -> ToolConfig:
    config = ToolConfig(default_academic_search_engine=args.engine)
    if args.config:
        config = ToolConfig.from_yaml(args.config)
        config = replace(config, default_academic_search_engine=args.engine)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--engine", choices=["openalex", "semantic_scholar"], default=None)
    parser.add_argument("--detail-out", type=Path, default=DETAIL_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--reference-survey-out", type=Path, default=REFERENCE_SURVEY_OUT)
    parser.add_argument("--resolved-cache", type=Path, default=RESOLVED_MISSING_CACHE)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--include-reference-surveys", action="store_true")
    args = parser.parse_args()
    args.engine = args.engine or infer_engine(args.input)

    async def runner():
        await SessionManager.init()
        try:
            config = config_from_args(args)
            details, summary = await build_local_analysis(args.input, args.engine, config, args.resolved_cache)
            if args.include_reference_surveys:
                coverage = await reference_survey_coverage(details, config)
            else:
                coverage = None
            return details, summary, coverage
        finally:
            await SessionManager.close()

    details, summary, coverage = asyncio.run(runner())
    write_jsonl(args.detail_out, details)
    write_json(args.summary_out, summary)
    print(f"wrote {args.detail_out} records={len(details)}")
    print(f"wrote {args.summary_out}")
    if coverage is not None:
        write_jsonl(args.reference_survey_out, coverage)
        print(f"wrote {args.reference_survey_out} records={len(coverage)}")


if __name__ == "__main__":
    main()
