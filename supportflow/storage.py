"""SQLite persistence for tickets, approval decisions, and their audit payloads."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


class TicketStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )


            connection.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    details TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ticket_annotations (
                    ticket_id TEXT NOT NULL,
                    annotator_id TEXT NOT NULL,
                    annotator_role TEXT NOT NULL,
                    label TEXT NOT NULL,
                    note TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (ticket_id, annotator_id)
                )"""
            )

    def record_knowledge_audit(self, action: str, source_id: str, version: str, actor_id: str, actor_role: str, details: dict | None = None) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO knowledge_audit_events (created_at, action, source_id, version, actor_id, actor_role, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), action, source_id, version, actor_id, actor_role, json.dumps(details or {}, ensure_ascii=False)),
            )

    def list_knowledge_audit(self, limit: int = 50) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT created_at, action, source_id, version, actor_id, actor_role, details FROM knowledge_audit_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"created_at": row[0], "action": row[1], "source_id": row[2], "version": row[3], "actor_id": row[4], "actor_role": row[5], "details": json.loads(row[6])}
            for row in rows
        ]

    def record_annotation(self, ticket_id: str, label: str, note: str | None, annotator_id: str, annotator_role: str) -> dict | None:
        if self.get(ticket_id) is None:
            return None
        annotation = {
            "ticket_id": ticket_id,
            "annotator_id": annotator_id,
            "annotator_role": annotator_role,
            "label": label,
            "note": note,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO ticket_annotations (ticket_id, annotator_id, annotator_role, label, note, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                tuple(annotation.values()),
            )
        return annotation

    def list_annotation_queue(self, annotator_id: str, limit: int = 20) -> list[dict]:
        annotated_ids: set[str]
        with self._connection() as connection:
            rows = connection.execute("SELECT ticket_id FROM ticket_annotations WHERE annotator_id = ?", (annotator_id,)).fetchall()
        annotated_ids = {row[0] for row in rows}
        eligible = [ticket for ticket in self.list() if ticket["status"] not in {"processing", "failed"} and ticket["ticket"]["id"] not in annotated_ids]
        return eligible[:limit]

    def annotation_quality(self) -> dict:
        with self._connection() as connection:
            rows = connection.execute("SELECT ticket_id, label FROM ticket_annotations ORDER BY ticket_id").fetchall()
        labels_by_ticket: dict[str, list[str]] = {}
        for ticket_id, label in rows:
            labels_by_ticket.setdefault(ticket_id, []).append(label)
        total_pairs = matching_pairs = 0
        multi_annotated = 0
        for labels in labels_by_ticket.values():
            if len(labels) < 2:
                continue
            multi_annotated += 1
            for index, label in enumerate(labels):
                for other in labels[index + 1:]:
                    total_pairs += 1
                    matching_pairs += label == other
        return {
            "annotation_count": len(rows),
            "annotated_ticket_count": len(labels_by_ticket),
            "multi_annotated_ticket_count": multi_annotated,
            "pairwise_agreement": round(matching_pairs / total_pairs, 3) if total_pairs else None,
            "label_counts": dict(Counter(label for _, label in rows)),
        }

    def prompt_experiment_results(self, experiment_id: str, minimum_annotations: int = 3) -> dict:
        with self._connection() as connection:
            rows = connection.execute("SELECT ticket_id, label FROM ticket_annotations").fetchall()
        labels_by_ticket: dict[str, list[str]] = {}
        for ticket_id, label in rows:
            labels_by_ticket.setdefault(ticket_id, []).append(label)
        groups = {arm: {"ticket_count": 0, "annotation_count": 0, "helpful_count": 0, "negative_count": 0} for arm in ("control", "treatment")}
        negative = {"needs_edit", "incorrect", "unsafe"}
        for ticket in self.list():
            details = ticket["ticket"]
            if details.get("experiment_id") != experiment_id or details.get("experiment_arm") not in groups:
                continue
            arm = details["experiment_arm"]
            groups[arm]["ticket_count"] += 1
            labels = labels_by_ticket.get(details["id"], [])
            groups[arm]["annotation_count"] += len(labels)
            groups[arm]["helpful_count"] += sum(label == "helpful" for label in labels)
            groups[arm]["negative_count"] += sum(label in negative for label in labels)
        for group in groups.values():
            count = group["annotation_count"]
            group["helpful_rate"] = round(group["helpful_count"] / count, 3) if count else None
            group["negative_rate"] = round(group["negative_count"] / count, 3) if count else None
        ready = all(group["annotation_count"] >= minimum_annotations for group in groups.values())
        return {"experiment_id": experiment_id, "minimum_annotations": minimum_annotations, "verdict": "ready_for_comparison" if ready else "insufficient_sample", "arms": groups}

    def next_id(self) -> str:
        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        return f"T-{count + 1:04d}"

    def save(self, payload: dict):
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO tickets (id, status, payload) VALUES (?, ?, ?)",
                (payload["ticket"]["id"], payload["status"], json.dumps(payload, ensure_ascii=False)),
            )

    def get(self, ticket_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT payload FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        """Find the original durable submission for a client retry key."""
        return next((ticket for ticket in self.list() if ticket.get("job", {}).get("idempotency_key") == idempotency_key), None)

    def list(self, status: str | None = None) -> list[dict]:
        query = "SELECT payload FROM tickets"
        parameters: tuple[str, ...] = ()
        if status:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY id DESC"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row[0]) for row in rows]

    def decide(self, ticket_id: str, decision: str, edited_reply: str | None = None, actor_role: str = "unknown", actor_id: str = "unknown") -> dict | None:
        payload = self.get(ticket_id)
        if payload is None:
            return None
        payload["status"] = "resolved" if decision == "approve" else "escalated"
        payload["approval"] = {"decision": decision, "edited_reply": edited_reply, "actor_role": actor_role, "actor_id": actor_id}
        if edited_reply and payload.get("draft"):
            payload["draft"]["reply"] = edited_reply
        self.save(payload)
        return payload

    def record_feedback(self, ticket_id: str, label: str, note: str | None = None, actor_role: str = "unknown", actor_id: str = "unknown") -> dict | None:
        """Persist a reviewer label without changing the ticket's workflow state."""
        payload = self.get(ticket_id)
        if payload is None:
            return None
        payload["feedback"] = {"label": label, "note": note, "actor_role": actor_role, "actor_id": actor_id}
        if label in {"needs_edit", "incorrect", "unsafe"}:
            source_ids = sorted({item["source_id"] for item in payload.get("evidence", [])})
            recommendations = {
                "needs_edit": "补充更清晰的处理步骤、边界和时效说明",
                "incorrect": "复核相关知识来源，修订不准确的事实或政策",
                "unsafe": "优先复核安全边界，并补充禁止承诺与升级规则",
            }
            payload["knowledge_gap"] = {
                "status": "open",
                "label": label,
                "note": note,
                "category": payload["triage"]["category"],
                "source_ids": source_ids,
                "recommendation": recommendations[label],
            }
        self.save(payload)
        return payload

    def list_knowledge_gaps(self) -> list[dict]:
        """Aggregate negative reviewer feedback into an actionable knowledge backlog."""
        groups: dict[tuple[str, tuple[str, ...], str], dict] = {}
        for ticket in self.list():
            gap = ticket.get("knowledge_gap")
            if not gap or gap["status"] == "resolved":
                continue
            key = (gap["category"], tuple(gap["source_ids"]), gap["status"])
            group = groups.setdefault(key, {
                "category": gap["category"], "source_ids": gap["source_ids"], "status": gap["status"],
                "ticket_ids": [], "labels": Counter(), "notes": [], "recommendation": gap["recommendation"],
            })
            group["ticket_ids"].append(ticket["ticket"]["id"])
            group["labels"][gap["label"]] += 1
            if gap.get("note"):
                group["notes"].append(gap["note"])
        priority = {"unsafe": 3, "incorrect": 2, "needs_edit": 1}
        result = []
        for group in groups.values():
            group["count"] = len(group["ticket_ids"])
            group["labels"] = dict(group["labels"])
            group["priority"] = max((priority[label] for label in group["labels"]), default=1)
            result.append(group)
        return sorted(result, key=lambda item: (-item["priority"], -item["count"], item["category"]))

    def mark_knowledge_gaps_for_verification(self, category: str, source_id: str) -> int:
        """Publishing knowledge moves matching gaps to verification, never blindly resolves them."""
        updated = 0
        verification_started_at = datetime.now(UTC).isoformat()
        for ticket in self.list():
            gap = ticket.get("knowledge_gap")
            if gap and gap["status"] == "open" and gap["category"] == category:
                gap["status"] = "verification"
                gap["verification_source_id"] = source_id
                gap["verification_started_at"] = verification_started_at
                self.save(ticket)
                updated += 1
        return updated

    @staticmethod
    def _cohort_metrics(tickets: list[dict], evaluated_source_id: str | None = None) -> dict:
        feedback = [ticket["feedback"] for ticket in tickets if ticket.get("feedback")]
        negative = sum(item["label"] in {"needs_edit", "incorrect", "unsafe"} for item in feedback)
        approvals = [ticket["approval"] for ticket in tickets if ticket.get("approval")]
        source_citation_count = sum(
            any(evidence.get("source_id") == evaluated_source_id for evidence in ticket.get("evidence", []))
            for ticket in tickets
        ) if evaluated_source_id else 0
        return {
            "ticket_count": len(tickets),
            "feedback_count": len(feedback),
            "negative_feedback_rate": round(negative / len(feedback), 3) if feedback else None,
            "human_edit_rate": round(sum(bool(item.get("edited_reply")) for item in approvals) / len(approvals), 3) if approvals else None,
            "escalation_rate": round(sum(ticket["status"] == "escalated" for ticket in tickets) / len(tickets), 3) if tickets else None,
            "evaluated_source_citation_count": source_citation_count,
            "evaluated_source_citation_rate": round(source_citation_count / len(tickets), 3) if tickets and evaluated_source_id else None,
        }

    @staticmethod
    def _outcome_assessment(baseline: dict, current: dict, minimum_feedback_samples: int) -> dict:
        """Turn cohort metrics into a conservative, operator-facing next action."""
        if baseline["feedback_count"] < minimum_feedback_samples or current["feedback_count"] < minimum_feedback_samples:
            return {
                "status": "insufficient_sample",
                "negative_feedback_rate_delta": None,
                "recommendation": "继续收集人工反馈；发布前后各达到最小样本量后再判断知识优化效果。",
            }
        if (current["evaluated_source_citation_rate"] or 0) < 0.5:
            return {
                "status": "insufficient_attribution",
                "negative_feedback_rate_delta": None,
                "recommendation": "发布后的工单较少引用该知识来源，先检查检索召回、知识分类和文档内容，再判断优化效果。",
            }
        delta = round((current["negative_feedback_rate"] or 0) - (baseline["negative_feedback_rate"] or 0), 3)
        if delta <= -0.05:
            return {
                "status": "improved",
                "negative_feedback_rate_delta": delta,
                "recommendation": "负向反馈已下降，保留当前知识版本并继续监控后续样本。",
            }
        if delta >= 0.05:
            return {
                "status": "regressed",
                "negative_feedback_rate_delta": delta,
                "recommendation": "负向反馈未改善，建议主管抽样复盘该知识来源和人工改写记录。",
            }
        return {
            "status": "neutral",
            "negative_feedback_rate_delta": delta,
            "recommendation": "负向反馈没有显著变化，建议结合人工改写和升级工单继续复盘。",
        }

    def list_knowledge_gap_outcomes(self, minimum_feedback_samples: int = 3) -> list[dict]:
        """Compare feedback cohorts before and after a knowledge-gap publication."""
        all_tickets = self.list()
        verified: dict[tuple[str, str, str], dict] = {}
        for ticket in all_tickets:
            gap = ticket.get("knowledge_gap", {})
            if gap.get("status") == "verification" and gap.get("verification_started_at"):
                key = (gap["category"], gap["verification_source_id"], gap["verification_started_at"])
                verified[key] = gap
        outcomes = []
        for (category, source_id, started_at), _ in verified.items():
            before, after = [], []
            for ticket in all_tickets:
                if ticket["triage"]["category"] != category:
                    continue
                queued_at = ticket.get("job", {}).get("queued_at")
                if not queued_at:
                    continue
                (before if queued_at < started_at else after).append(ticket)
            baseline = self._cohort_metrics(before, source_id)
            current = self._cohort_metrics(after, source_id)
            assessment = self._outcome_assessment(baseline, current, minimum_feedback_samples)
            outcomes.append({
                "category": category,
                "source_id": source_id,
                "verification_started_at": started_at,
                "baseline": baseline,
                "after": current,
                "verdict": "ready_for_comparison" if assessment["status"] not in {"insufficient_sample", "insufficient_attribution"} else assessment["status"],
                "assessment": assessment,
                "minimum_feedback_samples": minimum_feedback_samples,
            })
        return sorted(outcomes, key=lambda item: item["verification_started_at"], reverse=True)

    def save_failure(self, ticket: dict, error_type: str, job: dict | None = None) -> dict:
        """Keep a resumable, safe failure record instead of losing the request."""
        payload = {
            "ticket": ticket,
            "status": "failed",
            "triage": {"category": "unknown", "priority": "unknown", "risk_level": "high", "missing_information": []},
            "evidence": [],
            "draft": None,
            "quality": {"outcome": "failed", "reasons": ["工作流执行异常，等待人工恢复"]},
            "trace": [{"agent": "orchestrator", "message": f"执行失败：{error_type}"}],
            "failure": {"error_type": error_type, "resumable": True},
        }
        payload["job"] = (job or {"attempts": 0, "max_attempts": 3}) | {"state": "failed"}
        self.save(payload)
        return payload

    def save_processing(self, ticket: dict, previous: dict | None = None, reset_attempts: bool = False, idempotency_key: str | None = None, source: str = "workbench", submitted_by: str = "unknown") -> dict:
        """Persist a safe placeholder before a background workflow starts."""
        previous_job = (previous or {}).get("job", {})
        attempts = 0 if reset_attempts else previous_job.get("attempts", 0)
        payload = {
            "ticket": ticket,
            "status": "processing",
            "job": {
                "state": "queued",
                "attempts": attempts,
                "max_attempts": previous_job.get("max_attempts", 3),
                "recovery_count": previous_job.get("recovery_count", 0),
                "manual_retry_count": previous_job.get("manual_retry_count", 0) + int(reset_attempts),
                "idempotency_key": idempotency_key or previous_job.get("idempotency_key"),
                "source": source,
                "submitted_by": submitted_by,
                "queued_at": datetime.now(UTC).isoformat(),
            },
            "triage": {"category": "unknown", "priority": "normal", "risk_level": "medium", "missing_information": []},
            "evidence": [],
            "draft": None,
            "quality": None,
            "trace": [{"agent": "orchestrator", "message": "后台工作流已入队，等待分流、检索、生成与质检"}],
            "progress": {"stage": "queued", "message": "已创建工单，后台 Agent 正在处理"},
        }
        self.save(payload)
        return payload

    def claim_processing(self, ticket_id: str, recovered_from_restart: bool = False) -> dict | None:
        """Mark one durable task as running and count its attempt."""
        payload = self.get(ticket_id)
        if payload is None or payload["status"] != "processing":
            return None
        job = payload.setdefault("job", {"state": "queued", "attempts": 0, "max_attempts": 3, "recovery_count": 0, "manual_retry_count": 0})
        if job.get("state") == "running" and not recovered_from_restart:
            return None
        if job.get("attempts", 0) >= job.get("max_attempts", 3):
            return self.save_failure(payload["ticket"], "RetryLimitExceeded", job)
        job["state"] = "running"
        job["attempts"] = job.get("attempts", 0) + 1
        if recovered_from_restart:
            job["recovery_count"] = job.get("recovery_count", 0) + 1
        payload["progress"] = {"stage": "starting", "message": "已从持久化队列领取任务，正在启动 Agent 工作流"}
        self.save(payload)
        return payload

    def update_progress(self, ticket_id: str, stage: str, message: str) -> dict | None:
        """Update a queued ticket without overwriting its original request."""
        payload = self.get(ticket_id)
        if payload is None or payload["status"] != "processing":
            return payload
        payload["progress"] = {"stage": stage, "message": message}
        self.save(payload)
        return payload


class PostgresTicketStore(TicketStore):
    """PostgreSQL implementation that preserves the TicketStore contract.

    The psycopg dependency is imported lazily, leaving SQLite as the local,
    zero-configuration development path.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._initialize()

    @contextmanager
    def _connection(self):
        try:
            import psycopg
        except ImportError as error:  # pragma: no cover - deployment dependency
            raise RuntimeError("PostgreSQL requires psycopg. Install the production dependencies first.") from error
        connection = psycopg.connect(self.database_url)
        try:
            yield _PostgresConnection(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS knowledge_audit_events (id BIGSERIAL PRIMARY KEY, created_at TEXT NOT NULL, action TEXT NOT NULL, source_id TEXT NOT NULL, version TEXT NOT NULL, actor_id TEXT NOT NULL, actor_role TEXT NOT NULL, details TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ticket_annotations (ticket_id TEXT NOT NULL, annotator_id TEXT NOT NULL, annotator_role TEXT NOT NULL, label TEXT NOT NULL, note TEXT, updated_at TEXT NOT NULL, PRIMARY KEY (ticket_id, annotator_id))"
            )


class _PostgresConnection:
    """Compatibility adapter for this repository's small SQL surface."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query: str, parameters: tuple | list = ()):
        normalized = query.replace("?", "%s")
        if normalized.startswith("INSERT OR REPLACE INTO tickets"):
            normalized = "INSERT INTO tickets (id, status, payload) VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, payload = EXCLUDED.payload"
        elif normalized.startswith("INSERT OR REPLACE INTO ticket_annotations"):
            normalized = (
                "INSERT INTO ticket_annotations (ticket_id, annotator_id, annotator_role, label, note, updated_at) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (ticket_id, annotator_id) DO UPDATE SET annotator_role = EXCLUDED.annotator_role, label = EXCLUDED.label, note = EXCLUDED.note, updated_at = EXCLUDED.updated_at"
            )
        return self.connection.execute(normalized, parameters)


def create_ticket_store(database_path: str | Path, database_url: str | None = None) -> TicketStore:
    """Select PostgreSQL only when an explicit production URL is configured."""
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        return PostgresTicketStore(database_url)
    return TicketStore(database_path)
