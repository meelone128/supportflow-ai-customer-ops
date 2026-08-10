"""Celery worker entry point. Run with: celery -A supportflow.celery_worker worker -l INFO."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir

from celery import Celery

from supportflow.execution import TicketExecutor
from supportflow.knowledge import create_default_knowledge_grounder
from supportflow.langgraph_runtime import SupportFlowGraph
from supportflow.storage import create_ticket_store


redis_url = os.getenv("SUPPORTFLOW_REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("supportflow", broker=redis_url, backend=redis_url)
celery_app.conf.update(task_acks_late=True, task_track_started=True, worker_prefetch_multiplier=1)


def _executor_from_environment() -> TicketExecutor:
    data_dir = Path(os.getenv("SUPPORTFLOW_DATA_DIR", Path(gettempdir()) / "supportflow"))
    store = create_ticket_store(data_dir / "supportflow.sqlite3", os.getenv("SUPPORTFLOW_DATABASE_URL"))
    knowledge = create_default_knowledge_grounder(managed_directory=data_dir / "knowledge")
    return TicketExecutor(store, SupportFlowGraph(knowledge=knowledge))


@celery_app.task(name="supportflow.execute_ticket")
def execute_ticket_task(ticket_id: str, recovered_from_restart: bool = False) -> dict:
    return _executor_from_environment().execute(ticket_id, recovered_from_restart)
