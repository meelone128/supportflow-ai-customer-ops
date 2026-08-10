"""Stable domain types shared by the workflow, UI, and evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TicketCategory(StrEnum):
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_ISSUE = "product_issue"
    REFUND_POLICY = "refund_policy"
    UNKNOWN = "unknown"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketStatus(StrEnum):
    PROCESSING = "processing"
    NEW = "new"
    TRIAGED = "triaged"
    RETRIEVED = "retrieved"
    DRAFT_READY = "draft_ready"
    PENDING_APPROVAL = "pending_approval"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass(frozen=True)
class Ticket:
    id: str
    subject: str
    content: str
    customer_id: str = "demo-customer"
    experiment_id: str | None = None
    experiment_arm: str | None = None


@dataclass(frozen=True)
class TriageResult:
    category: TicketCategory
    priority: str
    risk_level: RiskLevel
    missing_information: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    source_id: str
    title: str
    excerpt: str
    score: float
    source_version: str = "baseline"


@dataclass(frozen=True)
class ResolutionDraft:
    reply: str
    proposed_action: str
    confidence: float
    requires_human_approval: bool
    generation_mode: str = "template"


@dataclass(frozen=True)
class QualityDecision:
    outcome: TicketStatus
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceEvent:
    agent: str
    message: str
    duration_ms: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    ticket: Ticket
    status: TicketStatus
    triage: TriageResult
    evidence: list[Evidence] = field(default_factory=list)
    draft: ResolutionDraft | None = None
    quality: QualityDecision | None = None
    trace: list[TraceEvent] = field(default_factory=list)
