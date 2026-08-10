"""Task-dispatch interface with local-thread and optional Celery adapters."""

from __future__ import annotations

import os
from threading import Thread
from typing import Protocol

from supportflow.execution import TicketExecutor


class TicketDispatcher(Protocol):
    def dispatch(self, ticket_id: str, recovered_from_restart: bool = False) -> None: ...


class LocalTicketDispatcher:
    """Development adapter that keeps the existing no-infrastructure run path."""

    def __init__(self, executor: TicketExecutor):
        self.executor = executor

    def dispatch(self, ticket_id: str, recovered_from_restart: bool = False) -> None:
        Thread(
            target=self.executor.execute,
            args=(ticket_id, recovered_from_restart),
            daemon=True,
            name=f"supportflow-{ticket_id}",
        ).start()


class InlineTicketDispatcher:
    """Serverless-demo adapter that completes a ticket in the request lifecycle."""

    def __init__(self, executor: TicketExecutor):
        self.executor = executor

    def dispatch(self, ticket_id: str, recovered_from_restart: bool = False) -> None:
        self.executor.execute(ticket_id, recovered_from_restart)


class CeleryTicketDispatcher:
    """Production adapter that only enqueues a durable ticket ID to Redis/Celery."""

    def dispatch(self, ticket_id: str, recovered_from_restart: bool = False) -> None:
        from supportflow.celery_worker import execute_ticket_task

        execute_ticket_task.delay(ticket_id, recovered_from_restart)


def create_task_dispatcher(executor: TicketExecutor) -> TicketDispatcher:
    backend = os.getenv("SUPPORTFLOW_TASK_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalTicketDispatcher(executor)
    if backend == "inline":
        return InlineTicketDispatcher(executor)
    if backend == "celery":
        return CeleryTicketDispatcher()
    raise ValueError("SUPPORTFLOW_TASK_BACKEND must be 'local', 'inline', or 'celery'")
