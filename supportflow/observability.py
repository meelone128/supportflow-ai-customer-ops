"""Structured, secret-safe operational events and metric-derived alerts."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("supportflow.operations")


def emit_event(event: str, **fields: Any) -> None:
    """Emit JSON for container log collectors without ticket content or credentials."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True))


def calculate_operational_alerts(metrics: dict) -> list[dict]:
    """Return actionable alerts; empty traffic is not treated as an incident."""
    alerts: list[dict] = []
    if metrics["status_counts"].get("failed", 0):
        alerts.append({"code": "failed_tickets", "severity": "critical", "message": "存在失败工单，请检查依赖并由主管恢复处理。"})
    if metrics["ticket_count"] and metrics["evidence_coverage"] < 0.8:
        alerts.append({"code": "low_evidence_coverage", "severity": "warning", "message": "证据覆盖率低于 80%，请检查知识库和检索链路。"})
    if metrics["feedback_count"] >= 3 and metrics["negative_feedback_rate"] > 0.1:
        alerts.append({"code": "negative_feedback", "severity": "warning", "message": "负向反馈率超过 10%，请优先抽样复盘。"})
    if metrics["latency_ms"]["p95"] > 2500:
        alerts.append({"code": "high_p95_latency", "severity": "warning", "message": "P95 延迟超过 2500ms，请检查模型与检索依赖。"})
    if metrics["retrieval_mode_counts"].get("local_tfidf_fallback", 0):
        alerts.append({"code": "retrieval_fallback", "severity": "info", "message": "发生检索降级，当前已使用本地索引保障服务连续性。"})
    return alerts
