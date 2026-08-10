"""One durable ticket-execution module shared by local threads and Celery workers."""

from __future__ import annotations

from dataclasses import asdict

from supportflow.domain import Ticket, TicketStatus, WorkflowResult
from supportflow.langgraph_runtime import SupportFlowGraph
from supportflow.observability import emit_event
from supportflow.storage import TicketStore


def serialize(result: WorkflowResult) -> dict:
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["triage"]["category"] = result.triage.category.value
    payload["triage"]["risk_level"] = result.triage.risk_level.value
    if result.quality:
        payload["quality"]["outcome"] = result.quality.outcome.value
    return payload


class TicketExecutor:
    """Claims and completes a persisted ticket exactly once per durable attempt."""

    def __init__(self, store: TicketStore, workflow: SupportFlowGraph):
        self.store = store
        self.workflow = workflow

    def execute(self, ticket_id: str, recovered_from_restart: bool = False) -> dict:
        current = self.store.get(ticket_id)
        if current is None:
            return {"status": "missing", "ticket_id": ticket_id}
        ticket = Ticket(**current["ticket"])
        claimed = self.store.claim_processing(ticket.id, recovered_from_restart=recovered_from_restart)
        if claimed is None or claimed["status"] != TicketStatus.PROCESSING.value:
            return claimed or {"ticket": asdict(ticket), "status": "missing"}
        try:
            run_with_progress = getattr(self.workflow, "run_with_progress", None)
            if run_with_progress:
                result = run_with_progress(ticket, thread_id=ticket.id, progress_callback=lambda stage, message: self.store.update_progress(ticket.id, stage, message))
            else:
                result = self.workflow.run(ticket, thread_id=ticket.id)
            payload = serialize(result)
            payload["job"] = claimed["job"] | {"state": "completed"}
            if customer_portal := claimed.get("customer_portal"):
                payload["customer_portal"] = customer_portal
        except Exception as error:
            latest = self.store.get(ticket.id)
            emit_event("ticket_execution_failed", ticket_id=ticket.id, error_type=type(error).__name__)
            return self.store.save_failure(asdict(ticket), type(error).__name__, (latest or {}).get("job"))
        self.store.save(payload)
        emit_event(
            "ticket_execution_completed",
            ticket_id=ticket.id,
            status=payload["status"],
            category=payload["triage"]["category"],
            trace_event_count=len(payload["trace"]),
        )
        return payload
