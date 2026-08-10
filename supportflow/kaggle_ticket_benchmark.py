"""Privacy-minimised Kaggle ticket import and workflow audit.

The source dataset includes customer-related columns. This module deliberately
keeps only operational fields needed to evaluate routing; it never persists names,
emails, ages, or gender into SupportFlow's benchmark outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from supportflow.domain import Ticket
from supportflow.generation import TemplateReplyWriter
from supportflow.workflow import SupportFlowWorkflow


KAGGLE_DATASET = "suraj520/customer-support-ticket-dataset"
KAGGLE_LICENSE = "CC0-1.0"


@dataclass(frozen=True)
class BenchmarkTicket:
    ticket_id: str
    subject: str
    content: str
    source_ticket_type: str
    source_priority: str
    source_status: str
    source_channel: str
    source_resolution: str
    source_satisfaction: str


@dataclass(frozen=True)
class TicketAuditReport:
    source_dataset: str
    source_license: str
    cases: int
    workflow_status_counts: dict[str, int]
    workflow_category_counts: dict[str, int]
    high_priority_cases: int
    high_priority_sent_to_human_review: int
    high_priority_review_capture: float
    interpretation: str


def _text(row: Mapping[str, object], column: str) -> str:
    return str(row.get(column, "") or "").strip()


def benchmark_tickets_from_rows(rows: Iterable[Mapping[str, object]], limit: int = 100) -> list[BenchmarkTicket]:
    """Keep operational ticket fields only, and make the sample deterministic."""
    tickets: list[BenchmarkTicket] = []
    for row in rows:
        subject = _text(row, "Ticket Subject")
        content = _text(row, "Ticket Description")
        ticket_id = _text(row, "Ticket ID")
        if not subject or not content or not ticket_id:
            continue
        tickets.append(BenchmarkTicket(
            ticket_id=f"KAGGLE-{ticket_id}",
            subject=subject,
            content=content,
            source_ticket_type=_text(row, "Ticket Type"),
            source_priority=_text(row, "Ticket Priority").lower(),
            source_status=_text(row, "Ticket Status").lower(),
            source_channel=_text(row, "Ticket Channel").lower(),
            source_resolution=_text(row, "Resolution"),
            source_satisfaction=_text(row, "Customer Satisfaction Rating"),
        ))
        if len(tickets) >= limit:
            break
    return tickets


def audit_tickets(tickets: list[BenchmarkTicket]) -> TicketAuditReport:
    workflow = SupportFlowWorkflow(reply_writer=TemplateReplyWriter())
    statuses: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    high_priority = reviewed = 0
    for item in tickets:
        result = workflow.run(Ticket(item.ticket_id, item.subject, item.content))
        statuses[result.status.value] += 1
        categories[result.triage.category.value] += 1
        if item.source_priority in {"high", "critical"}:
            high_priority += 1
            if result.status.value in {"escalated", "pending_approval"}:
                reviewed += 1
    return TicketAuditReport(
        source_dataset=KAGGLE_DATASET,
        source_license=KAGGLE_LICENSE,
        cases=len(tickets),
        workflow_status_counts=dict(sorted(statuses.items())),
        workflow_category_counts=dict(sorted(categories.items())),
        high_priority_cases=high_priority,
        high_priority_sent_to_human_review=reviewed,
        high_priority_review_capture=round(reviewed / high_priority, 3) if high_priority else 0.0,
        interpretation="Review capture compares source priority with SupportFlow's human-review decision; it is a routing audit, not a claim that source priority is ground truth for model safety.",
    )


def load_kaggle_rows(cache_directory: Path | None = None) -> list[dict[str, str]]:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Missing optional dependency. Run: python -m pip install -r supportflow/requirements-data.txt") from error
    directory = Path(kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(cache_directory) if cache_directory else None))
    csv_path = next(directory.rglob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"No CSV file found after downloading {KAGGLE_DATASET}")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_audit_report(report: TicketAuditReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
