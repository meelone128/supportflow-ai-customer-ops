"""Read-only aggregation of versioned, offline experiment reports for the workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentCenter:
    """One small interface for the UI; missing reports are explicit, never guessed."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def snapshot(self) -> dict[str, Any]:
        reports = self.project_root / "supportflow" / "evals" / "reports"
        corpus = self.project_root / "supportflow" / "evals" / "public_corpus"
        retrieval = self._read(corpus / "retrieval_comparison.json")
        manifest = self._read(corpus / "manifest.json")
        routing_before = self._read(reports / "kaggle-ticket-audit.json")
        routing_after = self._read(reports / "kaggle-ticket-audit-multilingual.json")

        knowledge = {
            "available": bool(retrieval and manifest),
            "report_path": "supportflow/evals/public_corpus/retrieval_comparison.json",
        }
        if retrieval and manifest:
            knowledge |= {
                "blind_test_cases": retrieval["blind_test_cases"],
                "before": retrieval["v1_target_intent_at_1"],
                "after": retrieval["v2_target_intent_at_1"],
                "delta": retrieval["delta"],
                "leakage_check": manifest["leakage_check"],
                "metric_definition": retrieval["metric_definition"],
            }

        routing = {
            "available": bool(routing_before and routing_after),
            "report_path": "supportflow/evals/reports/kaggle-ticket-audit-multilingual.json",
        }
        if routing_before and routing_after:
            before_escalated = routing_before.get("workflow_status_counts", {}).get("escalated", 0)
            after_escalated = routing_after.get("workflow_status_counts", {}).get("escalated", 0)
            before_drafts = routing_before.get("workflow_status_counts", {}).get("pending_approval", 0)
            after_drafts = routing_after.get("workflow_status_counts", {}).get("pending_approval", 0)
            routing |= {
                "cases": routing_after["cases"],
                "before_escalated": before_escalated,
                "after_escalated": after_escalated,
                "before_drafts": before_drafts,
                "after_drafts": after_drafts,
                "before_unknown": routing_before.get("workflow_category_counts", {}).get("unknown", 0),
                "after_unknown": routing_after.get("workflow_category_counts", {}).get("unknown", 0),
                "high_priority_review_capture_before": routing_before["high_priority_review_capture"],
                "high_priority_review_capture_after": routing_after["high_priority_review_capture"],
                "interpretation": routing_after["interpretation"],
            }

        return {"knowledge_update": knowledge, "multilingual_routing": routing}
