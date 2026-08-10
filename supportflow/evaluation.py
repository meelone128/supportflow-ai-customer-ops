"""Offline retrieval evaluation, kept independent from API and UI layers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from supportflow.domain import Ticket, TicketCategory, TicketStatus
from supportflow.generation import TemplateReplyWriter
from supportflow.knowledge import KnowledgeGrounder, create_default_knowledge_grounder
from supportflow.workflow import SupportFlowWorkflow


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    category: TicketCategory
    expected_source_id: str


@dataclass(frozen=True)
class RetrievalReport:
    cases: int
    hit_at_1: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalExperimentReport:
    name: str
    cases: int
    hit_at_1: float
    mean_reciprocal_rank: float
    average_latency_ms: float
    p95_latency_ms: int
    retrieval_mode_counts: dict[str, int]


def evaluate_retrieval(grounder: KnowledgeGrounder, cases: list[RetrievalCase]) -> RetrievalReport:
    reciprocal_ranks: list[float] = []
    hits = 0
    for case in cases:
        source_ids = [evidence.source_id for evidence in grounder.find_evidence(case.query, case.category)]
        if source_ids and source_ids[0] == case.expected_source_id:
            hits += 1
        rank = next((index + 1 for index, source_id in enumerate(source_ids) if source_id == case.expected_source_id), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
    total = len(cases)
    return RetrievalReport(total, round(hits / total, 3), round(sum(reciprocal_ranks) / total, 3))


def evaluate_retrieval_experiment(name: str, grounder: KnowledgeGrounder, cases: list[RetrievalCase]) -> RetrievalExperimentReport:
    """Measure retrieval quality and latency while reporting the actual retriever used."""
    reciprocal_ranks: list[float] = []
    latencies: list[int] = []
    modes: dict[str, int] = {}
    hits = 0
    for case in cases:
        started = perf_counter()
        evidence, mode = grounder.retrieve(case.query, case.category)
        latencies.append(round((perf_counter() - started) * 1000))
        modes[mode] = modes.get(mode, 0) + 1
        source_ids = [item.source_id for item in evidence]
        if source_ids and source_ids[0] == case.expected_source_id:
            hits += 1
        rank = next((index + 1 for index, source_id in enumerate(source_ids) if source_id == case.expected_source_id), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
    total = len(cases)
    ordered_latencies = sorted(latencies)
    return RetrievalExperimentReport(
        name=name,
        cases=total,
        hit_at_1=round(hits / total, 3),
        mean_reciprocal_rank=round(sum(reciprocal_ranks) / total, 3),
        average_latency_ms=round(sum(latencies) / total, 1) if total else 0.0,
        p95_latency_ms=ordered_latencies[max(0, ceil(total * 0.95) - 1)] if total else 0,
        retrieval_mode_counts=modes,
    )


@dataclass(frozen=True)
class WorkflowCase:
    case_id: str
    subject: str
    content: str
    expected_category: TicketCategory
    expected_status: TicketStatus
    expected_source_id: str | None = None
    expect_no_draft: bool = False


@dataclass(frozen=True)
class WorkflowReport:
    cases: int
    category_accuracy: float
    status_accuracy: float
    evidence_hit_rate: float
    safety_case_pass_rate: float


def load_workflow_cases(path: Path | None = None) -> list[WorkflowCase]:
    cases_path = path or Path(__file__).parent / "evals" / "ticket_cases.json"
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return [
        WorkflowCase(
            case_id=case["case_id"],
            subject=case["subject"],
            content=case["content"],
            expected_category=TicketCategory(case["expected_category"]),
            expected_status=TicketStatus(case["expected_status"]),
            expected_source_id=case.get("expected_source_id"),
            expect_no_draft=case.get("expect_no_draft", False),
        )
        for case in raw_cases
    ]


def default_retrieval_cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(case.content, case.expected_category, case.expected_source_id)
        for case in load_workflow_cases()
        if case.expected_source_id is not None
    ]


def run_retrieval_comparison(include_bailian: bool = False) -> list[RetrievalExperimentReport]:
    """Compare the deterministic local baseline with optional Bailian semantic retrieval."""
    cases = default_retrieval_cases()
    reports = [evaluate_retrieval_experiment("local_tfidf", KnowledgeGrounder(), cases)]
    if include_bailian:
        reports.append(evaluate_retrieval_experiment("bailian_semantic", create_default_knowledge_grounder(), cases))
    return reports


def evaluate_workflow(workflow: SupportFlowWorkflow, cases: list[WorkflowCase]) -> WorkflowReport:
    category_hits = status_hits = evidence_hits = safety_hits = safety_cases = 0
    for case in cases:
        result = workflow.run(Ticket(case.case_id, case.subject, case.content))
        category_hits += result.triage.category is case.expected_category
        status_hits += result.status is case.expected_status
        if case.expected_source_id:
            evidence_hits += any(item.source_id == case.expected_source_id for item in result.evidence)
        if case.expect_no_draft:
            safety_cases += 1
            safety_hits += result.status is TicketStatus.ESCALATED and result.draft is None
    total = len(cases)
    evidence_cases = sum(case.expected_source_id is not None for case in cases)
    return WorkflowReport(
        cases=total,
        category_accuracy=round(category_hits / total, 3),
        status_accuracy=round(status_hits / total, 3),
        evidence_hit_rate=round(evidence_hits / evidence_cases, 3) if evidence_cases else 0.0,
        safety_case_pass_rate=round(safety_hits / safety_cases, 3) if safety_cases else 0.0,
    )


def run_default_workflow_evaluation() -> WorkflowReport:
    """Run a deterministic, no-API-cost regression suite against the product workflow."""
    return evaluate_workflow(SupportFlowWorkflow(reply_writer=TemplateReplyWriter()), load_workflow_cases())


def assert_workflow_regression_gate(report: WorkflowReport) -> None:
    """Fail CI when a change regresses deterministic workflow safety or grounding."""
    thresholds = {
        "category_accuracy": 0.95,
        "status_accuracy": 0.98,
        "evidence_hit_rate": 0.95,
        "safety_case_pass_rate": 1.0,
    }
    failures = [f"{name}={getattr(report, name)} < {minimum}" for name, minimum in thresholds.items() if getattr(report, name) < minimum]
    if failures:
        raise RuntimeError("Workflow regression gate failed: " + "; ".join(failures))
