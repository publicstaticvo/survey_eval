from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "downloaded" / "selected_manifest.json"
CLASSIFIED_ROOT = BASE.parent / "agent" / "test_output" / "golden_surveys"
INPUT_ROOT = BASE.parent / "agent" / "test_inputs" / "golden_surveys"
OUT_JSON = BASE / "defect_injection_plans.json"
OUT_MD = BASE / "defect_injection_plans.md"

TARGET_LABELS = {"SUMMARY", "SYNTHESIS", "COMPARISON", "EVALUATION", "BACKGROUND"}
BAD_SECTION_WORDS = {
    "introduction",
    "background",
    "preliminar",
    "related work",
    "discussion",
    "future",
    "conclusion",
    "appendix",
    "limitation",
}


UNSUPPORTED_SCOPE_TOPICS = {
    "retrieval-augmented_generation_for_large_language_models_a_survey": "end-to-end robotic manipulation systems and embodied RAG deployments",
    "instruction_tuning_for_large_language_models_a_survey": "federated on-device instruction tuning protocols and accelerator-level deployment",
    "a_survey_on_evaluation_of_large_language_models": "legal compliance auditing and clinical deployment certification for LLM evaluators",
    "harnessing_the_power_of_llms_in_practice_a_survey_on_chatgpt_and_beyond": "quantum computing applications and embedded-device compiler optimization with ChatGPT",
    "augmented_language_models_a_survey": "protein-structure wet-lab automation and autonomous laboratory control",
    "a_survey_on_in-context_learning": "privacy-preserving federated in-context learning systems and hardware scheduling",
    "towards_reasoning_in_large_language_models_a_survey": "formal verification of deployed autonomous-vehicle controllers using LLM reasoning",
    "multimodal_learning_with_transformers_a_survey": "blockchain consensus protocols and secure smart-contract verification",
    "image_data_augmentation_for_deep_learning_a_survey": "reinforcement-learning policy optimization and robot navigation benchmarks",
    "transformers_in_time_series_a_survey": "medical image segmentation transformers and radiology workflow deployment",
    "survey_of_hallucination_in_natural_language_generation": "hardware-level mitigation of hallucination on edge accelerators",
    "transformers_in_medical_imaging_a_survey": "machine translation, dialogue summarization, and text-only language modeling",
    "generalized_out-of-distribution_detection_a_survey": "cryptographic protocol verification and secure multiparty computation",
    "a_survey_on_multi-modal_summarization": "time-series forecasting and financial anomaly detection",
    "towards_efficient_synchronous_federated_training_a_survey_on_system_optimization_strategies": "neural radiance fields and 3D scene reconstruction",
    "asynchronous_federated_learning_on_heterogeneous_devices_a_survey": "text-to-image diffusion sampling and prompt engineering",
    "a_survey_of_deep_reinforcement_learning_in_recommender_systems_a_systematic_review_and_future_directions": "medical image registration and molecular docking",
    "a_survey_of_exploration_methods_in_reinforcement_learning": "database transaction processing and SQL query optimization",
    "survey_of_low-resource_machine_translation": "protein folding, molecular dynamics simulation, and clinical trial design",
    "neuron-level_interpretation_of_deep_nlp_models_a_survey": "federated learning communication scheduling and wireless resource allocation",
}


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def slug_from_json_path(path: str) -> str:
    return Path(path).stem


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def citation_keys(citations: Any) -> list[str]:
    if isinstance(citations, dict):
        return [str(v) for v in citations.values() if str(v)]
    if isinstance(citations, list):
        return [str(v) for v in citations if str(v)]
    if isinstance(citations, str) and citations:
        return [citations]
    return []


def section_number(section_id: Any) -> int | None:
    text = str(section_id)
    m = re.match(r"(\d+)", text)
    return int(m.group(1)) if m else None


def section_path(parent: str, section: dict[str, Any]) -> str:
    sid = section.get("section_id", "")
    title = normalize_text(section.get("title", ""))
    label = f"{sid} {title}".strip()
    return f"{parent} > {label}" if parent else label


def iter_sections(sections: list[dict[str, Any]], parent: str = "", depth: int = 1):
    for sec in sections or []:
        path = section_path(parent, sec)
        yield sec, path, depth
        yield from iter_sections(sec.get("sections", []) or [], path, depth + 1)


def paragraph_sentences(paragraph: Any) -> list[dict[str, Any]]:
    if isinstance(paragraph, dict):
        return [s for s in paragraph.get("sentences", []) or [] if isinstance(s, dict)]
    if isinstance(paragraph, list):
        return [s for s in paragraph if isinstance(s, dict)]
    return []


def iter_sentences(sections: list[dict[str, Any]]):
    for sec, path, depth in iter_sections(sections):
        top = section_number(sec.get("section_id"))
        for pidx, para in enumerate(sec.get("paragraphs", []) or [], 1):
            for sidx, sent in enumerate(paragraph_sentences(para), 1):
                yield {
                    "section": sec,
                    "path": path,
                    "depth": depth,
                    "top_section": top,
                    "paragraph_index": pidx,
                    "sentence_index": sidx,
                    "sentence": sent,
                }


def eligible_top_sections(paper: dict[str, Any]) -> set[int]:
    nums = [
        section_number(sec.get("section_id"))
        for sec in paper.get("sections", []) or []
        if section_number(sec.get("section_id")) is not None
    ]
    nums = sorted(set(n for n in nums if n is not None))
    if len(nums) <= 4:
        return set(nums[1:-1])
    return set(nums[2:-2])


def in_deleted_section(item: dict[str, Any], deleted_path: str) -> bool:
    return bool(deleted_path) and (item["path"] == deleted_path or item["path"].startswith(deleted_path + " > "))


def choose_deleted_subsection(
    paper: dict[str, Any],
    eligible_tops: set[int],
    reserved_sentence_paths: set[str] | None = None,
    reserved_citation_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    reserved_sentence_paths = reserved_sentence_paths or set()
    reserved_citation_keys = reserved_citation_keys or set()
    scored = []
    for sec, path, depth in iter_sections(paper.get("sections", []) or []):
        top = section_number(sec.get("section_id"))
        if depth < 2 or top not in eligible_tops:
            continue
        title = normalize_text(sec.get("title", ""))
        lower = title.lower()
        if any(word in lower for word in BAD_SECTION_WORDS):
            continue
        sent_count = 0
        cit_count = 0
        for para in sec.get("paragraphs", []) or []:
            for sent in paragraph_sentences(para):
                if any(r == path or r.startswith(path + " > ") or path.startswith(r + " > ") for r in reserved_sentence_paths):
                    sent_count = -999
                    break
                if any(k in reserved_citation_keys for k in citation_keys(sent.get("citations"))):
                    sent_count = -999
                    break
                sent_count += 1
                cit_count += len(citation_keys(sent.get("citations")))
            if sent_count < 0:
                break
        if sent_count < 2:
            continue
        functional = sec.get("functional_type", "")
        score = sent_count + cit_count * 2 + (10 if functional in {"CONTENT", "TAXONOMY"} else 0)
        scored.append((score, path, sec, sent_count, cit_count))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: item[0])
    _score, path, sec, sent_count, cit_count = scored[0]
    return {
        "section_id": sec.get("section_id"),
        "title": normalize_text(sec.get("title", "")),
        "path": path,
        "sentence_count": sent_count,
        "citation_mention_count": cit_count,
        "reason": "middle method/content subsection with substantial text and citations",
    }


def ref_title(paper: dict[str, Any], key: str) -> str:
    entry = (paper.get("citations") or {}).get(key)
    if isinstance(entry, dict):
        return normalize_text(entry.get("title") or entry.get("ref_string") or key)
    return normalize_text(entry or key)


def choose_deleted_citations(
    paper: dict[str, Any],
    eligible_tops: set[int],
    deleted_path: str,
    reserved_keys: set[str],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sections: dict[str, set[str]] = {}
    sample: dict[str, str] = {}
    for item in iter_sentences(paper.get("sections", []) or []):
        if item["top_section"] not in eligible_tops or in_deleted_section(item, deleted_path):
            continue
        sent = item["sentence"]
        if sent.get("label") not in TARGET_LABELS and sent.get("label") is not None:
            continue
        keys = [k for k in citation_keys(sent.get("citations")) if k not in reserved_keys]
        for key in keys:
            counts[key] += 1
            sections.setdefault(key, set()).add(item["path"])
            sample.setdefault(key, normalize_text(sent.get("text", "")))
    chosen = []
    for key, count in counts.most_common(8):
        if len(chosen) >= 6:
            break
        chosen.append(
            {
                "citation_key": key,
                "reference_title": ref_title(paper, key),
                "mention_count_in_eligible_middle": count,
                "sections": sorted(sections.get(key, []))[:3],
                "sample_sentence": sample.get(key, ""),
                "planned_action": "delete this citation marker throughout the selected middle discussion; if the surrounding sentence only introduces this work, delete that local sentence as well",
            }
        )
    return chosen


def has_non_citation_number(text: str) -> bool:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return bool(re.search(r"\b\d+(?:\.\d+)?%?\b", text))


def modify_number(text: str) -> str:
    body = re.sub(r"\[[^\]]+\]", " ", text)
    m = re.search(r"\b(\d+)(?:\.\d+)?(%?)\b", body)
    if m:
        value = int(m.group(1))
        replacement = str(value * 2 if value > 1 else value + 3) + m.group(2)
        start = text.find(m.group(0))
        if start >= 0:
            return text[:start] + replacement + text[start + len(m.group(0)) :]
    return text.rstrip(".") + ", yielding an unsupported 50% improvement."


def modify_reverse(text: str) -> str:
    replacements = [
        (r"\bimproves?\b", "degrades"),
        (r"\benhances?\b", "weakens"),
        (r"\boutperforms?\b", "underperforms"),
        (r"\bachieves?\b", "fails to achieve"),
        (r"\bcan\b", "cannot"),
        (r"\bare able to\b", "are unable to"),
        (r"\bis able to\b", "is unable to"),
    ]
    for pattern, repl in replacements:
        if re.search(pattern, text, flags=re.I):
            return re.sub(pattern, repl, text, count=1, flags=re.I)
    return text.rstrip(".") + ", but the cited work actually shows the opposite effect."


def modify_attribution(text: str, wrong_title: str) -> str:
    clean_title = wrong_title or "an unrelated method"
    return text.rstrip(".") + f". This change should attribute the result to '{clean_title}', even though that work does not make this claim."


def modify_scope(text: str, bogus_topic: str) -> str:
    if not text:
        return f"We further provide a dedicated survey of {bogus_topic}, although the body does not cover that scope."
    return text.rstrip(".") + f". In addition, this survey systematically covers {bogus_topic}."


def choose_fact_errors(
    paper: dict[str, Any],
    eligible_tops: set[int],
    deleted_path: str,
    deleted_keys: set[str],
) -> list[dict[str, Any]]:
    candidates = []
    numeric = []
    for item in iter_sentences(paper.get("sections", []) or []):
        if item["top_section"] not in eligible_tops or in_deleted_section(item, deleted_path):
            continue
        sent = item["sentence"]
        keys = [k for k in citation_keys(sent.get("citations")) if k not in deleted_keys]
        text = normalize_text(sent.get("text", ""))
        if not keys or not text or sent.get("environment_type") != "text" or len(re.sub(r"[^A-Za-z]", "", text)) < 40:
            continue
        if sent.get("label") not in TARGET_LABELS and sent.get("label") is not None:
            continue
        item = {**item, "keys": keys, "text": text}
        candidates.append(item)
        if has_non_citation_number(text):
            numeric.append(item)

    chosen = []
    used_keys: set[str] = set()
    used_locations: set[tuple[str, int, int]] = set()

    def add(item: dict[str, Any], error_type: str, modified: str, rationale: str):
        loc = (item["path"], item["paragraph_index"], item["sentence_index"])
        if loc in used_locations:
            return False
        if any(k in used_keys for k in item["keys"]):
            return False
        used_locations.add(loc)
        used_keys.update(item["keys"])
        chosen.append(
            {
                "error_type": error_type,
                "section_path": item["path"],
                "paragraph_index": item["paragraph_index"],
                "sentence_index": item["sentence_index"],
                "citation_keys": item["keys"],
                "reference_titles": [ref_title(paper, k) for k in item["keys"]],
                "original_sentence": item["text"],
                "modified_sentence": modified,
                "rationale": rationale,
            }
        )
        return True

    for item in numeric:
        if add(item, "numeric distortion", modify_number(item["text"]), "change a reported count/percentage/parameter so the cited work no longer supports the statement"):
            break

    for item in candidates:
        if add(item, "conclusion reversal", modify_reverse(item["text"]), "reverse the cited conclusion or claimed effect"):
            break

    for item in candidates:
        wrong_key = next((k for k in (paper.get("citations") or {}) if k not in set(item["keys"]) | deleted_keys), "")
        if add(item, "attribution error", modify_attribution(item["text"], ref_title(paper, wrong_key)), "attribute the cited claim to an unrelated reference"):
            break

    for item in candidates:
        if len(chosen) >= 4:
            break
        modified = item["text"].rstrip(".") + " and completely solves the central limitation discussed in this area."
        add(item, "unsupported overgeneralization", modified, "inflate a limited result into a universal solution not supported by the citation")

    return chosen[:4]


def choose_scope_change(paper: dict[str, Any], slug: str) -> dict[str, Any]:
    intro = (paper.get("sections") or [{}])[0] if paper.get("sections") else {}
    contribution_sentences = []
    for item in iter_sentences([intro]):
        sent = item["sentence"]
        text = normalize_text(sent.get("text", ""))
        if sent.get("label") in {"CONTRIBUTION", "SCOPE"} or "contribution" in text.lower() or "section" in text.lower():
            contribution_sentences.append((item, text))
    if not contribution_sentences:
        for item in iter_sentences([intro]):
            text = normalize_text(item["sentence"].get("text", ""))
            if text:
                contribution_sentences.append((item, text))
                break
    item, text = contribution_sentences[0] if contribution_sentences else ({}, "")
    bogus_topic = UNSUPPORTED_SCOPE_TOPICS.get(slug, "a separate application domain not covered by the paper body")
    return {
        "section_path": item.get("path", "1 Introduction"),
        "paragraph_index": item.get("paragraph_index"),
        "sentence_index": item.get("sentence_index"),
        "original_sentence": text,
        "modified_sentence": modify_scope(text, bogus_topic),
        "unsupported_scope_added": bogus_topic,
        "rationale": "alter the introduction/contribution scope so it promises material absent from the body",
    }


def build_plan(item: dict[str, Any]) -> dict[str, Any]:
    slug = slug_from_json_path(item["json"])
    classified_path = CLASSIFIED_ROOT / slug / "02_classified_paper.json"
    source = "classified"
    if classified_path.exists():
        paper = load_json(classified_path)
    else:
        source = "test_input"
        fallback_path = INPUT_ROOT / f"{slug}.json"
        if not fallback_path.exists():
            fallback_path = Path(item["json"])
            source = "downloaded_parser_json"
        loaded = load_json(fallback_path)
        paper = loaded.get("full_text", loaded) if isinstance(loaded, dict) else {}

    eligible = eligible_top_sections(paper)
    scope_change = choose_scope_change(paper, slug)
    fact_errors = choose_fact_errors(paper, eligible, "", set())
    fact_keys = {k for error in fact_errors for k in error["citation_keys"]}
    deleted_citations = choose_deleted_citations(paper, eligible, "", fact_keys)
    deleted_keys = {c["citation_key"] for c in deleted_citations}
    reserved_paths = {error["section_path"] for error in fact_errors}
    deleted_subsection = choose_deleted_subsection(paper, eligible, reserved_paths, fact_keys | deleted_keys)
    deleted_path = deleted_subsection["path"] if deleted_subsection else ""
    if any(k in deleted_keys for k in fact_keys):
        raise RuntimeError(f"Overlap in {slug}")
    if any(in_deleted_section({"path": error["section_path"]}, deleted_path) for error in fact_errors):
        raise RuntimeError(f"Fact error falls in deleted subsection for {slug}")

    return {
        "title": item["title"],
        "slug": slug,
        "publication_date": item["publication_date"],
        "query": item["query"],
        "arxiv_id": item["arxiv_id"],
        "source_used": source,
        "eligible_middle_top_sections": sorted(eligible),
        "non_overlap_constraints": {
            "deleted_subsection_path": deleted_path or None,
            "deleted_citation_keys": sorted(deleted_keys),
            "fact_error_citation_keys": sorted(fact_keys),
            "note": "fact errors avoid the deleted subsection and deleted citation keys; deleted citations are selected after reserving fact-error citations",
        },
        "fact_errors": fact_errors,
        "structural_contradiction": scope_change,
        "citation_or_topic_missing": {
            "delete_core_citations": deleted_citations,
            "delete_method_subsection": deleted_subsection
            or {
                "planned_action": "no qualifying middle method/content subsection found after excluding first two sections, last two sections, conclusion/future/appendix",
            },
        },
    }


def write_markdown(plans: list[dict[str, Any]]) -> None:
    lines = ["# Defect Injection Plans", ""]
    for idx, plan in enumerate(plans, 1):
        lines.extend(
            [
                f"## {idx}. {plan['title']}",
                f"- slug: `{plan['slug']}`",
                f"- date/query/arXiv: {plan['publication_date']} / `{plan['query']}` / `{plan['arxiv_id']}`",
                f"- eligible middle top sections: {plan['eligible_middle_top_sections']}",
                f"- non-overlap: delete subsection `{plan['non_overlap_constraints']['deleted_subsection_path']}`; delete citation keys {plan['non_overlap_constraints']['deleted_citation_keys']}; fact citation keys {plan['non_overlap_constraints']['fact_error_citation_keys']}",
                "",
                "### Factual Errors",
            ]
        )
        for eidx, err in enumerate(plan["fact_errors"], 1):
            lines.extend(
                [
                    f"{eidx}. {err['error_type']} in `{err['section_path']}` P{err['paragraph_index']} S{err['sentence_index']}",
                    f"   - citations: {err['citation_keys']}",
                    f"   - refs: {'; '.join(err['reference_titles'])}",
                    f"   - original: {err['original_sentence']}",
                    f"   - modify to: {err['modified_sentence']}",
                    f"   - rationale: {err['rationale']}",
                ]
            )
        sc = plan["structural_contradiction"]
        lines.extend(
            [
                "",
                "### Structural Contradiction",
                f"- location: `{sc['section_path']}` P{sc['paragraph_index']} S{sc['sentence_index']}",
                f"- original: {sc['original_sentence']}",
                f"- modify to: {sc['modified_sentence']}",
                f"- unsupported added scope: {sc['unsupported_scope_added']}",
                "",
                "### Citation Or Topic Missing",
                "- delete core citations:",
            ]
        )
        for cite in plan["citation_or_topic_missing"]["delete_core_citations"]:
            lines.append(
                f"  - `{cite['citation_key']}` ({cite['reference_title']}), mentions={cite['mention_count_in_eligible_middle']}, sections={cite['sections']}"
            )
        sub = plan["citation_or_topic_missing"]["delete_method_subsection"]
        if sub.get("path"):
            lines.extend(
                [
                    f"- delete subsection: `{sub['path']}`",
                    f"- subsection reason: {sub['reason']} (sentences={sub['sentence_count']}, citation_mentions={sub['citation_mention_count']})",
                ]
            )
        else:
            lines.append(f"- delete subsection: {sub.get('planned_action')}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest = load_json(MANIFEST)
    plans = [build_plan(item) for item in manifest]
    OUT_JSON.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(plans)
    print(json.dumps({"plans": len(plans), "json": str(OUT_JSON), "markdown": str(OUT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()






