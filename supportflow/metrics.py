"""Operational metrics derived from the persisted audit trail."""

from __future__ import annotations

from collections import Counter
from math import ceil


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def calculate_metrics(tickets: list[dict]) -> dict:
    total = len(tickets)
    status_counts = Counter(ticket["status"] for ticket in tickets)
    category_counts = Counter(ticket["triage"]["category"] for ticket in tickets)
    tickets_with_evidence = sum(bool(ticket.get("evidence")) for ticket in tickets)
    approvals = [ticket.get("approval") for ticket in tickets if ticket.get("approval")]
    edited_approvals = sum(bool(approval.get("edited_reply")) for approval in approvals)
    feedback = [ticket["feedback"] for ticket in tickets if ticket.get("feedback")]
    negative_labels = {"needs_edit", "incorrect", "unsafe"}
    negative_feedback = sum(item["label"] in negative_labels for item in feedback)
    knowledge_gaps = [ticket["knowledge_gap"] for ticket in tickets if ticket.get("knowledge_gap") and ticket["knowledge_gap"]["status"] != "resolved"]
    trace_events = [event for ticket in tickets for event in ticket.get("trace", [])]
    durations = [event["duration_ms"] for event in trace_events if event.get("duration_ms") is not None]
    agent_durations: dict[str, list[int]] = {}
    for event in trace_events:
        if event.get("duration_ms") is not None:
            agent_durations.setdefault(event["agent"], []).append(event["duration_ms"])
    retrieval_modes = Counter(
        event.get("metadata", {}).get("retrieval_mode")
        for event in trace_events if event.get("metadata", {}).get("retrieval_mode")
    )
    model_versions = Counter(
        event.get("metadata", {}).get("model")
        for event in trace_events if event.get("metadata", {}).get("model")
    )
    return {
        "ticket_count": total,
        "status_counts": dict(status_counts),
        "category_counts": dict(category_counts),
        "escalation_rate": round(status_counts["escalated"] / total, 3) if total else 0.0,
        "evidence_coverage": round(tickets_with_evidence / total, 3) if total else 0.0,
        "human_edit_rate": round(edited_approvals / len(approvals), 3) if approvals else 0.0,
        "feedback_count": len(feedback),
        "negative_feedback_rate": round(negative_feedback / len(feedback), 3) if feedback else 0.0,
        "feedback_label_counts": dict(Counter(item["label"] for item in feedback)),
        "knowledge_gap_count": len(knowledge_gaps),
        "latency_ms": {"p50": _percentile(durations, 0.5), "p95": _percentile(durations, 0.95)},
        "agent_average_latency_ms": {agent: round(sum(values) / len(values)) for agent, values in agent_durations.items()},
        "retrieval_mode_counts": dict(retrieval_modes),
        "model_version_counts": dict(model_versions),
    }
