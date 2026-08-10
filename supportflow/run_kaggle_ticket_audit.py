"""CLI: download a public Kaggle ticket sample and audit SupportFlow routing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from supportflow.kaggle_ticket_benchmark import audit_tickets, benchmark_tickets_from_rows, load_kaggle_rows, write_audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit SupportFlow routing on public Kaggle tickets")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cache", type=Path, default=Path("supportflow/evals/kaggle_cache"))
    parser.add_argument("--report", type=Path, default=Path("supportflow/evals/reports/kaggle-ticket-audit.json"))
    arguments = parser.parse_args()
    tickets = benchmark_tickets_from_rows(load_kaggle_rows(arguments.cache), limit=arguments.limit)
    report = audit_tickets(tickets)
    write_audit_report(report, arguments.report)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
