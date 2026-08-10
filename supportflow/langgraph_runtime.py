"""LangGraph runtime adapter for the SupportFlow workflow."""

from __future__ import annotations

import operator
from time import perf_counter
from threading import Lock
from typing import Annotated, Callable
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from supportflow.business_tools import CustomerContextTool, OrderContextTool, ReadOnlyCustomerTool, ReadOnlyOrderTool
from supportflow.domain import Evidence, QualityDecision, ResolutionDraft, RiskLevel, Ticket, TraceEvent, TriageResult, WorkflowResult
from supportflow.generation import ReplyWriter, create_default_reply_writer, prompt_version_for
from supportflow.knowledge import KnowledgeGrounder, create_default_knowledge_grounder
from supportflow.policy import PolicyGate
from supportflow.workflow import SupportFlowWorkflow


class TicketState(TypedDict, total=False):
    """The shared state exchanged by the four SupportFlow agents."""

    ticket: Ticket
    triage: TriageResult
    evidence: list[Evidence]
    draft: ResolutionDraft | None
    quality: QualityDecision
    trace: Annotated[list[dict], operator.add]


class SupportFlowGraph:
    """A checkpointed state graph with explicit, finite agent transitions."""

    def __init__(
        self,
        knowledge: KnowledgeGrounder | None = None,
        policy: PolicyGate | None = None,
        reply_writer: ReplyWriter | None = None,
        order_tool: OrderContextTool | None = None,
        customer_tool: CustomerContextTool | None = None,
    ):
        self.knowledge = knowledge or create_default_knowledge_grounder()
        self.policy = policy or PolicyGate()
        self.order_tool = order_tool or ReadOnlyOrderTool.from_environment()
        self.customer_tool = customer_tool or ReadOnlyCustomerTool.from_environment()
        self.workflow = SupportFlowWorkflow(self.knowledge, self.policy, reply_writer or create_default_reply_writer())
        self._progress_callbacks: dict[str, Callable[[str, str], None]] = {}
        self._progress_lock = Lock()
        self.graph = self._build()

    def refresh_knowledge(self, knowledge: KnowledgeGrounder) -> None:
        """Swap the retriever after an operator publishes a knowledge document."""
        self.knowledge = knowledge
        self.workflow.knowledge = knowledge
        self.workflow.clear_reply_cache()

    def _build(self):
        builder = StateGraph(TicketState)
        builder.add_node("triage_agent", self._triage)
        builder.add_node("customer_context_agent", self._lookup_customer_context)
        builder.add_node("order_context_agent", self._lookup_order_context)
        builder.add_node("knowledge_agent", self._retrieve)
        builder.add_node("resolution_agent", self._draft)
        builder.add_node("quality_agent", self._quality)
        builder.add_edge(START, "triage_agent")
        builder.add_conditional_edges(
            "triage_agent",
            self._route_after_triage,
            {"customer_context_agent": "customer_context_agent", "quality_agent": "quality_agent"},
        )
        builder.add_edge("customer_context_agent", "order_context_agent")
        builder.add_edge("order_context_agent", "knowledge_agent")
        builder.add_edge("knowledge_agent", "resolution_agent")
        builder.add_edge("resolution_agent", "quality_agent")
        builder.add_edge("quality_agent", END)
        return builder.compile(checkpointer=MemorySaver())

    def _triage(self, state: TicketState):
        self._emit_progress(state["ticket"].id, "triaging", "正在识别工单分类与风险等级")
        started = perf_counter()
        triage = self.policy.triage(state["ticket"].content)
        return {"triage": triage, "trace": [self._trace("triage_agent", f"category={triage.category};risk={triage.risk_level}", started)]}

    @staticmethod
    def _route_after_triage(state: TicketState) -> str:
        """Prevent a high-risk ticket from reaching retrieval or generation nodes."""
        if state["triage"].risk_level is RiskLevel.HIGH:
            return "quality_agent"
        return "customer_context_agent"

    def _lookup_customer_context(self, state: TicketState):
        started = perf_counter()
        snapshot = self.customer_tool.lookup(state["ticket"].customer_id)
        if snapshot is None:
            return {"trace": [self._trace("customer_context_agent", "customer_context=not_found", started, tool_mode="readonly")]}
        evidence = Evidence(
            source_id=f"CRM-{snapshot.customer_id}",
            title="CRM 客户画像（只读）",
            excerpt=f"客户等级：{snapshot.tier}；待处理工单数：{snapshot.open_ticket_count}。{snapshot.summary}",
            score=1.0,
            source_version=snapshot.source_version,
        )
        return {"evidence": [evidence], "trace": [self._trace("customer_context_agent", f"customer_context={snapshot.customer_id}", started, tool_mode="readonly")]}

    def _lookup_order_context(self, state: TicketState):
        started = perf_counter()
        snapshot = self.order_tool.lookup_from_text(state["ticket"].content)
        if snapshot is None:
            return {"trace": [self._trace("order_context_agent", "order_context=not_requested", started)]}
        evidence = Evidence(
            source_id=f"ORDER-{snapshot.order_id}",
            title="订单系统只读查询",
            excerpt=f"订单号：{snapshot.order_id}；支付状态：{snapshot.payment_status}；退款状态：{snapshot.refund_status}。{snapshot.summary}",
            score=1.0,
            source_version=snapshot.source_version,
        )
        return {"evidence": [evidence], "trace": [self._trace("order_context_agent", f"order_context={snapshot.order_id}", started, tool_mode="readonly")]}

    def _retrieve(self, state: TicketState):
        self._emit_progress(state["ticket"].id, "retrieving", "正在检索可引用的知识库证据")
        started = perf_counter()
        knowledge_evidence, retrieval_mode = self.knowledge.retrieve(state["ticket"].content, state["triage"].category)
        evidence = state.get("evidence", []) + knowledge_evidence
        return {"evidence": evidence, "trace": [self._trace("knowledge_agent", f"evidence={len(evidence)}", started, retrieval_mode=retrieval_mode)]}

    def _draft(self, state: TicketState):
        self._emit_progress(state["ticket"].id, "generating", "正在基于证据生成客服回复草稿")
        started = perf_counter()
        draft = self.workflow._draft(state["ticket"], state["triage"], state["evidence"])
        mode = draft.generation_mode if draft else "none"
        model = getattr(self.workflow.reply_writer, "model", "template")
        return {"draft": draft, "trace": [self._trace("resolution_agent", f"draft={draft is not None};mode={mode}", started, generation_mode=mode, model=model, prompt_version=prompt_version_for(state["ticket"]))]}

    def _quality(self, state: TicketState):
        self._emit_progress(state["ticket"].id, "checking", "正在执行安全与人工审核质检")
        started = perf_counter()
        quality = self.policy.check(state["triage"], state.get("draft"), state.get("evidence", []))
        return {"quality": quality, "trace": [self._trace("quality_agent", f"outcome={quality.outcome}", started)]}

    @staticmethod
    def _trace(agent: str, message: str, started: float, **metadata: str) -> dict:
        return {"agent": agent, "message": message, "duration_ms": round((perf_counter() - started) * 1000), "metadata": metadata}

    def _emit_progress(self, ticket_id: str, stage: str, message: str) -> None:
        with self._progress_lock:
            callback = self._progress_callbacks.get(ticket_id)
        if callback:
            callback(stage, message)

    def run_with_progress(self, ticket: Ticket, thread_id: str, progress_callback: Callable[[str, str], None]) -> WorkflowResult:
        """Run one graph while safely exposing its actual active node to the UI."""
        with self._progress_lock:
            self._progress_callbacks[ticket.id] = progress_callback
        try:
            return self.run(ticket, thread_id)
        finally:
            with self._progress_lock:
                self._progress_callbacks.pop(ticket.id, None)

    def run(self, ticket: Ticket, thread_id: str) -> WorkflowResult:
        state = self.graph.invoke({"ticket": ticket, "trace": []}, {"configurable": {"thread_id": thread_id}})
        return WorkflowResult(
            ticket=state["ticket"],
            status=state["quality"].outcome,
            triage=state["triage"],
            evidence=state.get("evidence", []),
            draft=state.get("draft"),
            quality=state["quality"],
            trace=[
                TraceEvent(**entry)
                for entry in state["trace"]
            ],
        )
