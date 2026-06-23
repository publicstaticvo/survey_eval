import contextlib
import io
import re
import subprocess
from typing import Any

import bibtexparser

from ..utility.latex_parser.bib_parser import extract_entry_authors
from ..utility.utils import normalize_text, valid_check


_FIELD_ALIASES = {
    "authors": ("author", "authors", "authorships"),
    "author": ("author", "authors", "authorships"),
    "title": ("title",),
    "year": ("year", "publication_year", "publication_date", "publicationDate"),
    "venue": ("venue", "journal", "booktitle", "publisher", "container-title", "publication_venue"),
    "journal": ("journal", "venue", "container-title"),
    "booktitle": ("booktitle", "venue", "container-title"),
    "doi": ("doi", "DOI"),
}


class CitationCorrectnessCheck:
    """
    Check whether the bibliography fields recorded in a survey match metadata
    resolved by CitationParser. Missing fields on either side are ignored.
    """

    CHECK_FIELDS = ("title", "author", "year", "journal", "booktitle", "doi")

    def _parse_bibtex(self, text: str) -> dict[str, Any]:
        if not text.strip():
            return {}
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            database = bibtexparser.loads(text)
        return dict(database.entries[0]) if database.entries else {}

    def _command_bibtex(self, command: list[str]) -> dict[str, Any]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        except Exception:
            return {}
        if result.returncode != 0:
            return {}
        return self._parse_bibtex(result.stdout)

    def _work_ids(self, citation_data: dict[str, Any]) -> list[tuple[str, str]]:
        ids = []
        metadata = citation_data.get("metadata", {}) if isinstance(citation_data, dict) else {}
        openalex = metadata.get("openalex") or {}
        if openalex.get("id"):
            ids.append(("openalex", str(openalex["id"]).replace("https://openalex.org/", "")))
        for value in openalex.get("ids", []) or []:
            if value:
                ids.append(("openalex", str(value).replace("https://openalex.org/", "")))

        s2 = metadata.get("semantic scholar") or metadata.get("semantic_scholar") or {}
        if s2.get("paperId") or s2.get("id"):
            ids.append(("s2", str(s2.get("paperId") or s2.get("id"))))
        for value in s2.get("ids", []) or []:
            if value:
                ids.append(("s2", str(value)))

        seen = set()
        unique = []
        for source, work_id in ids:
            key = (source, work_id)
            if work_id and key not in seen:
                unique.append(key)
                seen.add(key)
        return unique

    def _candidate_metadata(self, citation_data: dict[str, Any], use_cli: bool) -> list[dict[str, Any]]:
        metadata = citation_data.get("metadata", {}) if isinstance(citation_data, dict) else {}
        candidates = [
            value for value in metadata.values()
            if isinstance(value, dict) and value
        ]
        if use_cli:
            for source, work_id in self._work_ids(citation_data):
                if source == "openalex":
                    entry = self._command_bibtex(["openalexcli", "work", work_id, "--bibtex"])
                else:
                    entry = self._command_bibtex(["s2cli", "bibtex", work_id])
                if entry:
                    candidates.append(entry)
        return candidates

    def _first_value(self, entry: dict[str, Any], field: str) -> Any:
        for key in _FIELD_ALIASES.get(field, (field,)):
            if key in entry and entry[key] not in (None, "", [], {}):
                return entry[key]
        external_ids = entry.get("external_ids") or entry.get("externalIds") or {}
        if field == "doi" and isinstance(external_ids, dict):
            return external_ids.get("DOI") or external_ids.get("doi")
        return None

    def _year(self, value: Any) -> str:
        match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
        return match.group(0) if match else ""

    def _authors(self, value: Any, entry: dict[str, Any]) -> list[str]:
        if isinstance(value, list):
            authors = []
            for item in value:
                if isinstance(item, str):
                    authors.append(item)
                elif isinstance(item, dict):
                    authors.append(item.get("name") or item.get("display_name") or "")
            return [normalize_text(author) for author in extract_entry_authors({"authors": authors})]
        if isinstance(value, str):
            return [normalize_text(author) for author in extract_entry_authors({"author": value})]
        return [normalize_text(author) for author in extract_entry_authors(entry)]

    def _matches(self, expected: dict[str, Any], candidate: dict[str, Any], field: str) -> bool | None:
        left = self._first_value(expected, field)
        right = self._first_value(candidate, field)
        if left in (None, "", [], {}) or right in (None, "", [], {}):
            return None

        if field in {"author", "authors"}:
            left_authors = set(self._authors(left, expected))
            right_authors = set(self._authors(right, candidate))
            if not left_authors or not right_authors:
                return None
            return bool(left_authors & right_authors)

        if field == "year":
            left_year = self._year(left)
            right_year = self._year(right)
            if not left_year or not right_year:
                return None
            return left_year == right_year

        if field == "doi":
            return normalize_text(str(left).removeprefix("https://doi.org/")) == normalize_text(
                str(right).removeprefix("https://doi.org/")
            )

        if field == "title":
            return valid_check(str(left), str(right), ratio=0.1)

        left_norm = normalize_text(str(left))
        right_norm = normalize_text(str(right))
        if not left_norm or not right_norm:
            return None
        return left_norm in right_norm or right_norm in left_norm

    def _check_single(self, citation_key: str, expected: Any, citation_data: dict[str, Any], use_cli: bool):
        if not isinstance(expected, dict):
            expected = {"title": str(expected or "")}
        candidates = self._candidate_metadata(citation_data or {}, use_cli)
        field_results = {}
        mismatches = []

        for field in self.CHECK_FIELDS:
            expected_value = self._first_value(expected, field)
            if expected_value in (None, "", [], {}):
                continue
            decisions = [self._matches(expected, candidate, field) for candidate in candidates]
            decisions = [decision for decision in decisions if decision is not None]
            if not decisions:
                continue
            passed = any(decisions)
            field_results[field] = "pass" if passed else "fail"
            if not passed:
                mismatches.append(field)

        return {
            "citation_key": citation_key,
            "status": "pass" if not mismatches else "fail",
            "fields": field_results,
            "mismatches": mismatches,
        }

    async def __call__(
        self,
        citations: dict[str, Any],
        paper_content_map: dict[str, Any],
        use_cli: bool = False,
    ) -> dict[str, Any]:
        results = [
            self._check_single(citation_key, citation_info, paper_content_map.get(citation_key, {}), use_cli)
            for citation_key, citation_info in (citations or {}).items()
        ]
        failed = [item for item in results if item["status"] == "fail"]
        return {
            "citation_evals": {
                "status": "pass" if not failed else "fail",
                "checked_count": len(results),
                "failed_count": len(failed),
                "results": results,
            }
        }
