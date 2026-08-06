from __future__ import annotations

import asyncio
from typing import Any


from ..utility.citation_utils import citation_keys as normalize_citation_keys
from ..utility.tool_config import ToolConfig
from .fact_check_single import SingleFactCorrectness
from .uncited_claims import UncitedClaimVerifier


class ClaimVerifier:
    def __init__(self, config: ToolConfig):
        self.fact_check = SingleFactCorrectness(config)
        self.uncited_claim_verifier = UncitedClaimVerifier(config)

    def _claim_text(self, claim: dict[str, Any]) -> str:
        return str(claim.get("claim") or claim.get("text") or "").strip()

    def _citation_keys(self, claim: dict[str, Any]) -> list[str]:
        return normalize_citation_keys(claim.get("citations") or claim.get("citation_keys") or [])

    def _split_claims(self, claim_data: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(claim_data, dict):
            claims = list(claim_data.get("claims", []) or [])
            claims.extend(claim_data.get("unverifiable_claims", []) or claim_data.get("unverifiable", []) or [])
            return [claim for claim in claims if isinstance(claim, dict) and self._claim_text(claim)]
        return [claim for claim in (claim_data or []) if isinstance(claim, dict) and self._claim_text(claim)]

    def _aggregate_fact_results(self, results: list[dict[str, Any]]) -> str:
        judgments = [item.get("fact_check", {}).get("judgment", "NEUTRAL") for item in results]
        if "SUPPORTED" in judgments:
            return "SUPPORTED"
        if "REFUTED" in judgments:
            return "REFUTED"
        if "ERROR" in judgments:
            return "ERROR"
        return "NEUTRAL"

    def _aggregate_neutral_reason(self, results: list[dict[str, Any]]) -> str:
        reasons = [item.get("fact_check", {}).get("reason", "") for item in results]
        if "evidence verify error" in reasons:
            return "evidence verify error"
        if "" in reasons:
            return ""
        if "insufficient information" in reasons:
            return "insufficient information"
        return reasons[0] if reasons else "unknown neutral type"

    def _unverifiable_result(self, claim: dict[str, Any]) -> dict[str, Any]:
        return {
            "claim": self._claim_text(claim),
            "claim_type": claim.get("claim_type", ""),
            "citation_keys": self._citation_keys(claim),
            "kind": "unverifiable",
            "judgment": "UNVERIFIABLE",
            "reason": "claim marked unverifiable by claim segmentation",
        }

    async def _fact_verification(self, claim: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        claim_text = self._claim_text(claim)
        citation_keys = self._citation_keys(claim)

        async def _single(key: str):
            citation_data = paper_content_map.get(key, {})
            if not citation_data or citation_data.get("status", 3) >= 3:
                return {"fact_check": {"claim": claim_text, "judgment": "NEUTRAL", "reason": "citation unresolved", "citation_key": key}}
            result = await self.fact_check(claim_text, citation_data)
            result["fact_check"]["citation_key"] = key
            return result

        tasks = [asyncio.create_task(_single(key)) for key in citation_keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        checked = []
        for key, result in zip(citation_keys, results):
            if isinstance(result, dict):
                checked.append(result)
            else:
                checked.append({"fact_check": {"claim": claim_text, "judgment": "ERROR", "reason": f"{type(result).__name__}: {result}", "citation_key": key}})
        judgment = self._aggregate_fact_results(checked)
        result = {"judgment": judgment, "references": [item["fact_check"] for item in checked]}
        if judgment == "NEUTRAL":
            result["reason"] = self._aggregate_neutral_reason(checked)
        return result

    async def _verify_cited_claim(self, claim: dict[str, Any], paper_content_map: dict[str, Any]) -> dict[str, Any]:
        citation_keys = self._citation_keys(claim)
        verification = await self._fact_verification(claim, paper_content_map)
        return {
            "claim": self._claim_text(claim),
            "claim_type": claim.get("claim_type", ""),
            "citation_keys": citation_keys,
            "kind": "cited_fact_check",
            **verification,
        }

    async def __call__(
        self,
        claim_data: list[dict[str, Any]] | dict[str, Any],
        paper_content_map: dict[str, Any],
        entity_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        claims = self._split_claims(claim_data)
        results_by_index: dict[int, dict[str, Any]] = {}
        cited_tasks: list[tuple[int, asyncio.Task]] = []
        uncited_claims: list[tuple[int, dict[str, Any]]] = []

        for index, claim in enumerate(claims):
            if not claim.get("verifiable", True):
                results_by_index[index] = self._unverifiable_result(claim)
                continue
            citation_keys = self._citation_keys(claim)
            if citation_keys:
                cited_tasks.append((index, asyncio.create_task(self._verify_cited_claim(claim, paper_content_map))))
            else:
                uncited_claims.append((index, claim))

        if cited_tasks:
            cited_results = await asyncio.gather(*[task for _, task in cited_tasks], return_exceptions=True)
            for (index, _), result in zip(cited_tasks, cited_results):
                if isinstance(result, dict):
                    results_by_index[index] = result
                else:
                    claim = claims[index]
                    results_by_index[index] = {
                        "claim": self._claim_text(claim),
                        "claim_type": claim.get("claim_type", ""),
                        "citation_keys": self._citation_keys(claim),
                        "kind": "cited_fact_check",
                        "judgment": "ERROR",
                        "reason": f"{type(result).__name__}: {result}",
                    }

        if uncited_claims:
            uncited_results = await self.uncited_claim_verifier(
                [claim for _, claim in uncited_claims],
                paper_content_map,
                entity_data=entity_data,
            )
            for (index, _claim), result in zip(uncited_claims, uncited_results):
                results_by_index[index] = result

        results = [results_by_index[index] for index in sorted(results_by_index)]
        supported = sum(1 for item in results if item.get("judgment") == "SUPPORTED")
        refuted = sum(1 for item in results if item.get("judgment") == "REFUTED")
        true_neutral = sum(1 for item in results if item.get("judgment") == "NEUTRAL" and item.get("reason", "") == "")
        fact_accuracy_denominator = supported + refuted + true_neutral
        fact_accuracy = supported / fact_accuracy_denominator if fact_accuracy_denominator else 1.0
        print(f"We check {len(results)} claims: {len(cited_tasks)} cited, {len(uncited_claims)} uncited, {sum(1 for item in results if item.get('judgment') == 'UNVERIFIABLE')} unverifiable")
        return {
            "fact_checks": results,
            "checked_count": len(results),
            "supported_count": supported,
            "refuted_count": refuted,
            "true_neutral_count": true_neutral,
            "fact_accuracy_denominator": fact_accuracy_denominator,
            "fact_accuracy": fact_accuracy,
        }

