"""The first SupportFlow graph: a finite, auditable collaboration of four agents."""

from __future__ import annotations

from supportflow.domain import RiskLevel, ResolutionDraft, Ticket, TicketCategory, TraceEvent, TriageResult, WorkflowResult
from supportflow.generation import GeneratedReply, ReplyWriter, TemplateReplyWriter
from supportflow.knowledge import KnowledgeGrounder
from supportflow.policy import PolicyGate


class SupportFlowWorkflow:
    """Coordinates routing, retrieval, drafting, and quality checks for one ticket."""

    def __init__(
        self,
        knowledge: KnowledgeGrounder | None = None,
        policy: PolicyGate | None = None,
        reply_writer: ReplyWriter | None = None,
    ):
        self.knowledge = knowledge or KnowledgeGrounder()
        self.policy = policy or PolicyGate()
        self.reply_writer = reply_writer or TemplateReplyWriter()
        self.reply_cache: dict[tuple, GeneratedReply] = {}

    def clear_reply_cache(self) -> None:
        """Invalidate drafts when the active knowledge set changes."""
        self.reply_cache.clear()

    def run(self, ticket: Ticket) -> WorkflowResult:
        triage = self.policy.triage(ticket.content)
        trace = [TraceEvent("triage_agent", f"分类为 {triage.category}; 风险为 {triage.risk_level}")]
        if triage.risk_level is RiskLevel.HIGH:
            evidence: list = []
            draft = None
            trace.append(TraceEvent("routing_policy", "高风险工单跳过检索与生成，直接进入人工升级决策"))
            quality = self.policy.check(triage, draft, evidence)
            trace.append(TraceEvent("quality_agent", "; ".join(quality.reasons)))
            return WorkflowResult(ticket, quality.outcome, triage, evidence, draft, quality, trace)
        evidence = self.knowledge.find_evidence(ticket.content, triage.category)
        trace.append(TraceEvent("knowledge_agent", f"检索到 {len(evidence)} 条证据"))
        draft = self._draft(ticket, triage, evidence)
        trace.append(TraceEvent("resolution_agent", f"已生成处理草稿；模式={draft.generation_mode}" if draft else "因证据不足未生成草稿"))
        quality = self.policy.check(triage, draft, evidence)
        trace.append(TraceEvent("quality_agent", "; ".join(quality.reasons)))
        return WorkflowResult(ticket, quality.outcome, triage, evidence, draft, quality, trace)

    def _draft(self, ticket: Ticket, triage: TriageResult, evidence: list) -> ResolutionDraft | None:
        if not evidence:
            return None
        if triage.category is TicketCategory.REFUND_POLICY:
            action = "收集订单信息并提交人工审核"
        elif triage.category is TicketCategory.ACCOUNT_ACCESS:
            action = "请求必要的身份核验信息"
        else:
            action = "收集故障复现信息"
        cache_key = (
            ticket.subject,
            ticket.content,
            triage.category.value,
            tuple((item.source_id, item.source_version) for item in evidence),
        )
        cached = self.reply_cache.get(cache_key)
        if cached:
            generated = GeneratedReply(cached.reply, f"{cached.mode}_cache")
        else:
            generated = self.reply_writer.generate(ticket, triage, evidence)
            if generated.mode != "template_fallback":
                self.reply_cache[cache_key] = generated
        return ResolutionDraft(generated.reply, action, min(0.9, evidence[0].score), True, generated.mode)
