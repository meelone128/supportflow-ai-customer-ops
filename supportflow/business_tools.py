"""Read-only business-system tool adapters used by the agent graph."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    payment_status: str
    refund_status: str
    summary: str
    source_version: str = "live_readonly"


class OrderContextTool(Protocol):
    def lookup_from_text(self, content: str) -> OrderSnapshot | None: ...


@dataclass(frozen=True)
class CustomerSnapshot:
    customer_id: str
    tier: str
    open_ticket_count: int
    summary: str
    source_version: str = "live_readonly"


class CustomerContextTool(Protocol):
    def lookup(self, customer_id: str) -> CustomerSnapshot | None: ...


class ExternalReadOnlyCustomerTool:
    """Adapter for the least-privilege customer profile endpoint in a CRM."""

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 2, request: Callable[[Request, float], bytes] | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.request = request or self._request

    @staticmethod
    def _request(request: Request, timeout_seconds: float) -> bytes:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: URL is deployment configuration
            return response.read(65536)

    def lookup(self, customer_id: str) -> CustomerSnapshot | None:
        request = Request(
            f"{self.base_url}/customers/{quote(customer_id, safe='')}/support-context",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            payload = json.loads(self.request(request, self.timeout_seconds))
            if not isinstance(payload, dict):
                return None
            tier, open_ticket_count, summary = payload.get("tier"), payload.get("open_ticket_count"), payload.get("summary")
            if not isinstance(tier, str) or not isinstance(open_ticket_count, int) or not isinstance(summary, str):
                return None
            return CustomerSnapshot(customer_id, tier, open_ticket_count, summary, source_version="external_readonly")
        except Exception:
            return None


class ReadOnlyCustomerTool:
    """Demo CRM seam. It intentionally has no mutation surface."""

    def __init__(self, customers: dict[str, CustomerSnapshot] | None = None):
        self.customers = customers or {
            "demo-customer": CustomerSnapshot("demo-customer", "普通客户", 0, "暂无待处理的同类历史工单。"),
            "web:C-10086": CustomerSnapshot("web:C-10086", "会员客户", 1, "近 30 天有 1 张退款相关工单，当前无已完成退款承诺。"),
        }

    @classmethod
    def from_environment(cls) -> CustomerContextTool:
        base_url = os.getenv("SUPPORTFLOW_CRM_API_BASE_URL", "").strip()
        token = os.getenv("SUPPORTFLOW_CRM_API_TOKEN", "").strip()
        if base_url and token:
            return ExternalReadOnlyCustomerTool(base_url, token, timeout_seconds=float(os.getenv("SUPPORTFLOW_CRM_API_TIMEOUT_SECONDS", "2")))
        return cls()

    def lookup(self, customer_id: str) -> CustomerSnapshot | None:
        return self.customers.get(customer_id)


class ExternalReadOnlyOrderTool:
    """Adapter for a support-safe HTTP endpoint owned by an order system."""

    _order_pattern = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)

    def __init__(self, base_url: str, token: str, timeout_seconds: float = 2, request: Callable[[Request, float], bytes] | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.request = request or self._request

    @staticmethod
    def _request(request: Request, timeout_seconds: float) -> bytes:
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310: URL is deployment configuration
            return response.read(65536)

    def lookup_from_text(self, content: str) -> OrderSnapshot | None:
        matched = self._order_pattern.search(content)
        if not matched:
            return None
        order_id = matched.group(0).upper()
        request = Request(
            f"{self.base_url}/orders/{quote(order_id)}/support-context",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            payload = json.loads(self.request(request, self.timeout_seconds))
            if not isinstance(payload, dict):
                return None
            values = [payload.get(key) for key in ("payment_status", "refund_status", "summary")]
            if not all(isinstance(value, str) and value.strip() for value in values):
                return None
            return OrderSnapshot(order_id, values[0], values[1], values[2], source_version="external_readonly")
        except Exception:
            # The workflow can still rely on RAG and human review. Never replace a
            # failed production lookup with demo data under the same order ID.
            return None


class ReadOnlyOrderTool:
    """Demo adapter and factory for the read-only order-context tool seam."""

    _order_pattern = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)

    def __init__(self, orders: dict[str, OrderSnapshot] | None = None):
        self.orders = orders or {
            "ORD-1001": OrderSnapshot("ORD-1001", "支付成功", "退款审核中", "订单已支付，退款申请正在人工审核，尚未产生到账承诺。"),
            "ORD-1002": OrderSnapshot("ORD-1002", "支付成功", "未发起退款", "订单已支付，当前未发现退款申请记录。"),
        }

    @classmethod
    def from_environment(cls) -> OrderContextTool:
        base_url = os.getenv("SUPPORTFLOW_ORDER_API_BASE_URL", "").strip()
        token = os.getenv("SUPPORTFLOW_ORDER_API_TOKEN", "").strip()
        if base_url and token:
            return ExternalReadOnlyOrderTool(
                base_url,
                token,
                timeout_seconds=float(os.getenv("SUPPORTFLOW_ORDER_API_TIMEOUT_SECONDS", "2")),
            )
        return cls()

    def lookup_from_text(self, content: str) -> OrderSnapshot | None:
        matched = self._order_pattern.search(content)
        if not matched:
            return None
        return self.orders.get(matched.group(0).upper())
