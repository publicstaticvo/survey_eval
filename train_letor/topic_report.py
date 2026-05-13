import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.tools.preprocess.build_sources import AnchorSurveySource
from agent.tools.preprocess.utils import extract_json
from agent.tools.utility.grobidpdf import PaperParser
from agent.tools.utility.llmclient import AsyncChat
from agent.tools.utility.openalex import get_openalex_client
from agent.tools.utility.tool_config import ToolConfig

PARTIAL_DIR = Path(__file__).resolve().parent / "outputs_partial"
PAPER_DIR = Path(__file__).resolve().parent / "paper"
OVERWRITE = True


SECTION_TOPIC_PROMPT = """
You are analyzing a survey paper section heading.
Decide whether the section is a substantive topic section, meaning it describes a concrete research subarea, method family, task, dataset family, problem class, or application area.

Return JSON only:
{"is_topic_section": true/false, "reason": "short reason"}

Non-topic examples: Introduction, Background, Related Work, Discussion, Conclusion, Future Work, Limitations, Appendix, Experiments, Results.

Heading: {heading}
Nearby citation count: {citation_count}
"""


DOMAIN_TOPIC_MERGE_PROMPT = """
You are given section headings from multiple anchor survey papers in the same research domain.
Merge semantically equivalent topic headings into unified topics.

Return JSON only:
{
  "topics": [
    {
      "topic_name": "human readable topic name",
      "source_headings": {"survey_id": ["original heading"]}
    }
  ]
}

Requirements:
1. Keep only substantive research topics.
2. Merge equivalent or near-equivalent headings.
3. Do not invent topics not supported by headings.

Input headings:
{headings_json}
"""


class SectionTopicJudgeClient(AsyncChat):
    PROMPT = SECTION_TOPIC_PROMPT

    def _availability(self, response, context):
        payload = extract_json(response)
        return {
            "is_topic_section": bool(payload.get("is_topic_section", False)),
            "reason": payload.get("reason", ""),
        }


class DomainTopicMergeClient(AsyncChat):
    PROMPT = DOMAIN_TOPIC_MERGE_PROMPT

    def _availability(self, response, context):
        payload = extract_json(response)
        return payload.get("topics", []) if isinstance(payload, dict) else []


@dataclass
class SurveyTopicReportBuilder:
    config: ToolConfig

    def __post_init__(self):
        self.openalex = get_openalex_client(self.config)
        self.parser = PaperParser()
        self.anchor_source = AnchorSurveySource(self.config)
        self.section_judge = SectionTopicJudgeClient(self.config.llm_server_info, self.config.sampling_params)
        self.domain_merge = DomainTopicMergeClient(self.config.llm_server_info, self.config.sampling_params)

    def _partial_path(self, survey_item: dict[str, Any]) -> Path:
        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        raw_name = f"{survey_item.get('index') or 'unknown'}_{survey_item.get('title') or survey_item.get('id') or 'survey'}"
        safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw_name)
        safe_name = "_".join(part for part in safe_name.split("_") if part)[:120]
        return PARTIAL_DIR / f"{safe_name}.partial.json"

    def _load_state(self, path: Path) -> dict[str, Any]:
        if OVERWRITE: return {"stage": "init"}
        if not path.exists():
            return {"stage": "init"}
        try:
            with path.open(encoding="utf-8") as f:
                state = json.load(f)
            print(f"TopicReportResume: loaded {path} stage={state.get('stage')}")
            return state
        except Exception as exc:
            print(f"TopicReportResume: failed to read {path}: {exc}")
            return {"stage": "init"}

    def _save_state(self, path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
        print(f"TopicReportResume: saved {path} stage={state.get('stage')}")

    def _debug_paper(self, role: str, paper: dict[str, Any]) -> None:
        print(
            json.dumps(
                {
                    "stage": "topic_report_debug",
                    "role": role,
                    "title": paper.get("title", ""),
                    "openalex_id": paper.get("id") or paper.get("openalex_id") or "",
                },
                ensure_ascii=False,
            )
        )

    def _load_local_survey_skeleton(self, survey_item: dict[str, Any]) -> tuple[dict[str, Any], str]:
        original_index = survey_item.get("original_index", survey_item.get("index"))
        if original_index is None:
            return {}, "missing original_index"
        path = PAPER_DIR / f"{original_index}.json"
        if not path.exists():
            return {}, f"local survey file not found: {path}"
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            return {}, f"failed to read local survey file {path}: {type(exc).__name__}: {exc}"
        if isinstance(payload, dict) and payload.get("sections") is not None:
            return payload, ""
        if isinstance(payload, dict) and isinstance(payload.get("skeleton"), dict):
            return payload["skeleton"], ""
        return {}, f"local survey file is not a Paper skeleton: {path}"

    async def download_openalex_paper(self, work_id: str):
        work_id = (work_id or "").replace("https://openalex.org/", "").strip()
        try:
            xml_content = await self.openalex.download_work_content(work_id)
        except Exception as exc:
            # TODO: 本地可部署 GROBID 后，在这里改成备用 PDF 下载/解析逻辑。
            raise RuntimeError(f"OpenAlex content download failed for {work_id}: {exc}") from exc
        try:
            return self.parser.parse(xml_content, mode="strict")
        except Exception as exc:
            # TODO: 本地可部署 GROBID 后，在这里改成更宽松的 XML/PDF 解析回退逻辑。
            raise RuntimeError(f"GROBID XML parse failed for {work_id}: {exc}") from exc

    def _collect_section_citations(self, section: dict) -> list[str]:
        cited = set()
        for paragraph in section.get("paragraphs", []):
            for sentence in paragraph:
                cited.update(sentence.get("citations", []) or [])
        for child in section.get("sections", []):
            cited.update(self._collect_section_citations(child))
        return sorted(cited)

    def _iter_sections(self, paper_skeleton: dict):
        def _walk(section: dict):
            yield section
            for child in section.get("sections", []):
                yield from _walk(child)

        for section in paper_skeleton.get("sections", []):
            yield from _walk(section)

    async def _build_sections(self, paper_skeleton: dict) -> list[dict[str, Any]]:
        sections = []

        async def _single(section: dict):
            heading = (section.get("title") or "").strip()
            cited_papers = self._collect_section_citations(section)
            if not heading:
                return None
            try:
                judgment = await self.section_judge.call(
                    inputs={"heading": heading, "citation_count": len(cited_papers)}
                )
            except Exception:
                judgment = {"is_topic_section": False, "reason": "LLM judgment failed"}
            return {
                "heading": heading,
                "is_topic_section": judgment["is_topic_section"],
                "cited_papers": cited_papers,
                "inferred_topic": "",
            }

        tasks = [asyncio.create_task(_single(section)) for section in self._iter_sections(paper_skeleton)]
        for task in asyncio.as_completed(tasks):
            item = await task
            if item:
                sections.append(item)
        return sections

    async def build_survey_record(self, survey_item: dict[str, Any]) -> dict[str, Any]:
        survey_id = survey_item['id']
        self._debug_paper("main_survey", {"id": survey_id, "title": survey_item.get("title", "")})
        skeleton, survey_load_error = self._load_local_survey_skeleton(survey_item)
        if not skeleton:
            print(
                json.dumps(
                    {
                        "stage": "topic_report_debug",
                        "role": "main_survey_local_load_failed",
                        "title": survey_item.get("title", ""),
                        "openalex_id": survey_id,
                        "error": survey_load_error,
                    },
                    ensure_ascii=False,
                )
            )
        anchor_data = await self.anchor_source(survey_item.get("query", survey_item.get("title", "")))
        anchor_surveys = []
        for item in anchor_data.get("anchor_surveys", {}).values():
            paper_meta = item.get("meta") or {}
            if paper_meta.get("id"):
                self._debug_paper("anchor_survey", paper_meta)
                anchor_surveys.append(paper_meta["id"])
        return {
            "survey_id": survey_id,
            "title": survey_item.get("title") or skeleton.get("title", ""),
            "domain": survey_item.get("query") or survey_item.get("domain") or survey_item.get("title", ""),
            "sections": await self._build_sections(skeleton) if skeleton else [],
            "anchor_surveys": sorted(set(anchor_surveys)),
            "main_survey_load_status": "ok" if skeleton else "failed",
            "main_survey_load_error": survey_load_error,
            "_anchor_data": anchor_data,
        }

    async def build_anchor_survey_records(self, domain_name: str, anchor_data: dict[str, Any]) -> list[dict[str, Any]]:
        records = []
        for downloaded in anchor_data.get("anchor_surveys", {}).values():
            paper_meta = downloaded.get("meta") or {}
            titles = downloaded.get("titles") or []
            survey_id = paper_meta.get("id") or paper_meta.get("title") or downloaded.get("title")
            if not survey_id:
                continue
            self._debug_paper("anchor_survey_record", {"id": survey_id, "title": paper_meta.get("title", "")})
            sections = [
                {
                    "heading": item.get("section_name", "") if isinstance(item, dict) else str(item),
                    "is_topic_section": True,
                    "cited_papers": [],
                    "inferred_topic": "",
                }
                for item in titles
                if (item.get("section_name", "") if isinstance(item, dict) else str(item)).strip()
            ]
            records.append(
                {
                    "survey_id": survey_id,
                    "title": paper_meta.get("title", ""),
                    "domain": domain_name,
                    "sections": sections,
                    "anchor_surveys": [],
                    "_anchor_data": {},
                }
            )
        return records

    async def build_domain_record(self, domain_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        heading_items = []
        for record in records:
            for section in record.get("sections", []):
                if section.get("is_topic_section"):
                    heading_items.append({"survey_id": record["survey_id"], "heading": section["heading"]})

        try:
            topics = await self.domain_merge.call(inputs={"headings_json": json.dumps(heading_items, ensure_ascii=False)})
        except Exception:
            topics = []

        survey_count = max(1, len({record["survey_id"] for record in records}))
        paper_topic_mapping: dict[str, list[str]] = {}
        unified_topics = []
        for topic in topics:
            topic_name = topic.get("topic_name", "")
            source_headings = topic.get("source_headings", {}) or {}
            representative_papers = set()
            for record in records:
                headings = set(source_headings.get(record["survey_id"], []))
                for section in record.get("sections", []):
                    if section.get("heading") in headings:
                        representative_papers.update(section.get("cited_papers", []))
            for paper_id in representative_papers:
                paper_topic_mapping.setdefault(paper_id, []).append(topic_name)
            unified_topics.append(
                {
                    "topic_name": topic_name,
                    "source_headings": source_headings,
                    "consensus_level": len(source_headings) / survey_count,
                    "representative_papers": sorted(representative_papers),
                }
            )
        return {
            "domain_name": domain_name,
            "unified_topics": unified_topics,
            "paper_topic_mapping": paper_topic_mapping,
        }

    def backfill_survey_record(self, record: dict[str, Any], domain_record: dict[str, Any]) -> dict[str, Any]:
        all_topics = [topic["topic_name"] for topic in domain_record.get("unified_topics", [])]
        covered = set()
        for section in record.get("sections", []):
            heading = section.get("heading", "")
            inferred = ""
            for topic in domain_record.get("unified_topics", []):
                if heading in set(topic.get("source_headings", {}).get(record["survey_id"], [])):
                    inferred = topic["topic_name"]
                    break
            section["inferred_topic"] = inferred
            if inferred:
                covered.add(inferred)
        record["covered_topics"] = sorted(covered)
        record["missing_topics"] = [topic for topic in all_topics if topic not in covered]
        record.pop("_anchor_data", None)
        return record

    async def build_report_for_item(self, survey_item: dict[str, Any]) -> dict[str, Any]:
        state_path = self._partial_path(survey_item)
        state = self._load_state(state_path)

        if "survey_record" not in state:
            survey_record = await self.build_survey_record(survey_item)
            state["survey_record"] = survey_record
            state["stage"] = "survey_record"
            self._save_state(state_path, state)
        else:
            survey_record = state["survey_record"]

        if "anchor_records" not in state:
            anchor_records = await self.build_anchor_survey_records(
                survey_record["domain"],
                survey_record.get("_anchor_data", {}),
            )
            state["anchor_records"] = anchor_records
            state["stage"] = "anchor_records"
            self._save_state(state_path, state)
        else:
            anchor_records = state["anchor_records"]

        if "domain_record" not in state:
            domain_inputs = [record for record in [survey_record, *anchor_records] if record.get("sections")]
            domain_record = await self.build_domain_record(survey_record["domain"], domain_inputs)
            state["domain_record"] = domain_record
            state["stage"] = "domain_record"
            self._save_state(state_path, state)
        else:
            domain_record = state["domain_record"]

        if "final_survey_record" not in state:
            final_survey_record = self.backfill_survey_record(dict(survey_record), domain_record)
            state["final_survey_record"] = final_survey_record
            state["stage"] = "complete"
            self._save_state(state_path, state)
        else:
            final_survey_record = state["final_survey_record"]

        return {"survey_record": final_survey_record, "domain_record": domain_record}
