"""Reply-generation seam: OpenAI in the app, deterministic templates in tests/fallbacks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from supportflow.domain import Evidence, Ticket, TicketCategory, TriageResult


PROMPT_VERSION = "support-reply-v1"


def prompt_version_for(ticket: Ticket) -> str:
    return "support-reply-v2" if ticket.experiment_arm == "treatment" else PROMPT_VERSION


def _apply_prompt_variant(reply: str, ticket: Ticket) -> str:
    if ticket.experiment_arm == "treatment":
        return f"{reply}\n\n为便于后续人工核验，请补充订单编号或相关申请时间。"
    return reply


def _is_english_ticket(ticket: Ticket) -> bool:
    text = f"{ticket.subject} {ticket.content}"
    latin_words = re.findall(r"[a-zA-Z]{2,}", text)
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_words) >= 2 and not chinese_characters


@dataclass(frozen=True)
class GeneratedReply:
    reply: str
    mode: str


class ReplyWriter(Protocol):
    def generate(self, ticket: Ticket, triage: TriageResult, evidence: list[Evidence]) -> GeneratedReply: ...


class TemplateReplyWriter:
    """The deterministic safe baseline used in unit tests and provider failures."""

    def generate(self, ticket: Ticket, triage: TriageResult, evidence: list[Evidence]) -> GeneratedReply:
        del evidence
        if _is_english_ticket(ticket):
            if triage.category is TicketCategory.REFUND_POLICY:
                return GeneratedReply(
                    "We received your after-sales request. Refund eligibility can be checked against the relevant policy; refund execution and any compensation commitment require human review (人工审核).",
                    "template",
                )
            if triage.category is TicketCategory.ACCOUNT_ACCESS:
                return GeneratedReply(
                    "Please first verify the account status and access permissions. To protect account security, do not share passwords or other sensitive details before identity verification is complete.",
                    "template",
                )
            return GeneratedReply(
                "Please provide the error message, steps to reproduce, and the time of the issue. If it cannot be reproduced, the case will be escalated to technical support.",
                "template",
            )
        if triage.category is TicketCategory.REFUND_POLICY:
            return GeneratedReply(
                "我们已收到您的售后咨询。退款资格可依据相关政策核验；退款执行和任何补偿承诺将由人工审核后处理。",
                "template",
            )
        if triage.category is TicketCategory.ACCOUNT_ACCESS:
            return GeneratedReply(
                "建议先确认账号状态与访问权限。为保护账户安全，请勿在未完成身份核验前提供敏感信息。",
                "template",
            )
        reply = "建议补充报错信息、操作步骤和发生时间；若问题无法复现，将升级技术支持进一步排查。"
        return GeneratedReply(_apply_prompt_variant(reply, ticket), "template")


class OpenAIReplyWriter:
    """Evidence-grounded response drafting using the OpenAI Responses API."""

    def __init__(self, api_key: str, model: str | None = None, fallback: ReplyWriter | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=float(os.getenv("SUPPORTFLOW_LLM_TIMEOUT_SECONDS", "15")), max_retries=0)
        self.model = model or os.getenv("SUPPORTFLOW_OPENAI_MODEL", "gpt-5.6")
        self.fallback = fallback or TemplateReplyWriter()

    def generate(self, ticket: Ticket, triage: TriageResult, evidence: list[Evidence]) -> GeneratedReply:
        prompt = _prompt(ticket, triage, evidence)
        try:
            response = self.client.responses.create(model=self.model, input=prompt)
            reply = response.output_text.strip()
            if not reply:
                raise ValueError("OpenAI returned an empty reply")
            return GeneratedReply(reply, "openai")
        except Exception:
            fallback = self.fallback.generate(ticket, triage, evidence)
            return GeneratedReply(fallback.reply, "template_fallback")


class DeepSeekReplyWriter:
    """DeepSeek's OpenAI-compatible Chat Completions adapter."""

    def __init__(self, api_key: str, model: str | None = None, fallback: ReplyWriter | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=float(os.getenv("SUPPORTFLOW_LLM_TIMEOUT_SECONDS", "15")), max_retries=0)
        self.model = model or os.getenv("SUPPORTFLOW_DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.fallback = fallback or TemplateReplyWriter()

    def generate(self, ticket: Ticket, triage: TriageResult, evidence: list[Evidence]) -> GeneratedReply:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You draft safe, evidence-grounded enterprise customer support replies."},
                    {"role": "user", "content": _prompt(ticket, triage, evidence)},
                ],
                max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            )
            reply = (response.choices[0].message.content or "").strip()
            if not reply:
                raise ValueError("DeepSeek returned an empty reply")
            return GeneratedReply(reply, "deepseek")
        except Exception:
            fallback = self.fallback.generate(ticket, triage, evidence)
            return GeneratedReply(fallback.reply, "template_fallback")


def _prompt(ticket: Ticket, triage: TriageResult, evidence: list[Evidence]) -> str:
    evidence_text = "\n\n".join(
        f"[{item.source_id}] {item.title}\n{item.excerpt}" for item in evidence
    )
    experiment_instruction = "- End with one concise clarification request that helps a human reviewer verify the case." if ticket.experiment_arm == "treatment" else ""
    language_instruction = "Reply in English." if _is_english_ticket(ticket) else "Reply in Chinese."
    return f"""Role: enterprise customer-support reply drafter.

Goal: draft one concise Chinese reply to the customer. It will be reviewed by a human before any action.

Ticket subject: {ticket.subject}
Ticket content: {ticket.content}
Ticket category: {triage.category.value}

Retrieved evidence (the only source of factual claims):
{evidence_text}

Constraints:
- Use only the evidence above; if it is insufficient, say that human verification is needed.
- Do not claim that a refund, compensation, permission change, or account operation has been completed.
- For refund requests, explicitly say that execution and compensation require human review.
- Do not request passwords or reveal account details.
{experiment_instruction}
- {language_instruction}
- Return only the customer-facing reply, without headings, citations, analysis, or markdown.
"""


def create_default_reply_writer() -> ReplyWriter:
    """Load local development config without exposing its value to the application trace."""
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".dev.env", override=False)
    load_dotenv(project_root / ".env", override=False)
    tracing_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not tracing_key:
        # Keep local runs private. Do not send failed trace
        # uploads when a developer has not configured LangSmith credentials.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or (openai_key if openai_key and openai_key.startswith("dpsk-") else None)
    if deepseek_key:
        return DeepSeekReplyWriter(deepseek_key)
    return OpenAIReplyWriter(openai_key) if openai_key else TemplateReplyWriter()
