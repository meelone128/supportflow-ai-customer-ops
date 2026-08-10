"""Deterministic guardrails that run independently of an LLM."""

from __future__ import annotations

from supportflow.domain import Evidence, QualityDecision, ResolutionDraft, RiskLevel, TicketCategory, TicketStatus, TriageResult


class PolicyGate:
    _english_high_risk_terms = ("complaint", "privacy", "data leak", "compensation", "lawyer", "legal action", "chargeback", "fraud")
    _english_refund_terms = ("refund", "return", "after-sales", "after sales")
    _english_account_terms = ("password", "access denied", "locked account", "login", "sign in", "sign-in", "credentials")
    _english_product_terms = ("error", "bug", "not working", "unavailable", "crash", "technical issue", "software issue")
    _high_risk_terms = ("投诉", "隐私", "泄露", "赔偿", "律师", "fraud")
    _prohibited_commitments = (
        "退款已成功",
        "已经退款",
        "已为您退款",
        "已发放补偿",
        "已修改权限",
        "已重置密码",
    )
    _sensitive_requests = ("请输入密码", "提供密码", "发送验证码", "银行卡密码")

    def triage(self, content: str) -> TriageResult:
        lowered = content.lower()
        if any(term in lowered for term in self._english_high_risk_terms):
            return TriageResult(TicketCategory.UNKNOWN, "high", RiskLevel.HIGH)
        if any(term in lowered for term in self._english_refund_terms):
            return TriageResult(TicketCategory.REFUND_POLICY, "normal", RiskLevel.MEDIUM)
        if any(term in lowered for term in self._english_account_terms):
            return TriageResult(TicketCategory.ACCOUNT_ACCESS, "normal", RiskLevel.MEDIUM)
        if any(term in lowered for term in self._english_product_terms):
            return TriageResult(TicketCategory.PRODUCT_ISSUE, "normal", RiskLevel.LOW)
        if any(term in lowered for term in self._high_risk_terms):
            return TriageResult(TicketCategory.UNKNOWN, "high", RiskLevel.HIGH)
        if any(term in lowered for term in ("退款", "退货", "售后", "refund")):
            return TriageResult(TicketCategory.REFUND_POLICY, "normal", RiskLevel.MEDIUM)
        if any(term in lowered for term in ("登录", "账号", "权限", "password", "access")):
            return TriageResult(TicketCategory.ACCOUNT_ACCESS, "normal", RiskLevel.MEDIUM)
        if any(term in lowered for term in ("报错", "异常", "故障", "不可用", "error", "bug")):
            return TriageResult(TicketCategory.PRODUCT_ISSUE, "normal", RiskLevel.LOW)
        return TriageResult(TicketCategory.UNKNOWN, "normal", RiskLevel.MEDIUM)

    def check(self, triage: TriageResult, draft: ResolutionDraft | None, evidence: list[Evidence]) -> QualityDecision:
        reasons: list[str] = []
        if triage.risk_level is RiskLevel.HIGH:
            reasons.append("高风险工单必须升级人工")
        if not evidence:
            reasons.append("缺少可引用的知识库证据")
        if draft is None:
            reasons.append("未生成可审核的回复草稿")
        else:
            if triage.category is TicketCategory.REFUND_POLICY and "人工审核" not in draft.reply:
                reasons.append("退款类草稿必须明确人工审核边界")
            if any(claim in draft.reply for claim in self._prohibited_commitments):
                reasons.append("草稿包含模型无权做出的执行或补偿承诺")
            if any(request in draft.reply for request in self._sensitive_requests):
                reasons.append("草稿请求了不应收集的敏感信息")
        if reasons:
            return QualityDecision(TicketStatus.ESCALATED, tuple(reasons))
        return QualityDecision(TicketStatus.PENDING_APPROVAL, ("已通过自动质检，等待人工确认",))
