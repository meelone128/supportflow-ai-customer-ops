import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient

from supportflow.api import create_app
from supportflow.domain import Ticket
from supportflow.generation import TemplateReplyWriter
from supportflow.knowledge import KnowledgeGrounder
from supportflow.langgraph_runtime import SupportFlowGraph
from supportflow.storage import TicketStore, create_ticket_store


class SupportFlowApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        workflow = SupportFlowGraph(knowledge=KnowledgeGrounder(), reply_writer=TemplateReplyWriter())
        self.client = TestClient(create_app(
            TicketStore(f"{self.temporary_directory.name}/tickets.sqlite3"),
            workflow,
            knowledge_directory=Path(self.temporary_directory.name) / "knowledge",
            experiment_project_root=Path(self.temporary_directory.name),
            run_async=False,
        ))
        self.client.headers.update({"X-Actor-Role": "administrator"})

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_ticket_can_be_created_then_human_approved(self):
        created = self.client.post("/tickets", json={"subject": "退款", "content": "购买后可以退款吗？"})
        self.assertEqual(created.status_code, 201)
        ticket_id = created.json()["ticket"]["id"]
        self.assertEqual(created.json()["status"], "pending_approval")
        decided = self.client.post(
            f"/tickets/{ticket_id}/approval",
            json={"decision": "approve", "edited_reply": "人工确认后的回复"},
        )
        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json()["status"], "resolved")
        restored = self.client.get(f"/tickets/{ticket_id}")
        self.assertEqual(restored.json()["approval"]["decision"], "approve")
        self.assertEqual(restored.json()["draft"]["reply"], "人工确认后的回复")
        reloaded_store = TicketStore(f"{self.temporary_directory.name}/tickets.sqlite3")
        self.assertEqual(reloaded_store.get(ticket_id)["status"], "resolved")

    def test_idempotency_key_returns_the_original_ticket_without_creating_a_duplicate(self):
        headers = {"Idempotency-Key": "demo-retry-0001"}
        first = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}, headers=headers)
        second = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}, headers=headers)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["ticket"]["id"], first.json()["ticket"]["id"])
        self.assertEqual(len(self.client.get("/tickets").json()), 1)
        self.assertEqual(second.json()["job"]["idempotency_key"], "demo-retry-0001")

    def test_jwt_authentication_enforces_roles_and_records_the_actor_identity(self):
        plain_client = TestClient(self.client.app)
        self.assertEqual(plain_client.get("/tickets").status_code, 401)
        support_token = self.client.post("/auth/token", json={"username": "support", "password": "supportflow-demo"}).json()["access_token"]
        supervisor_token = self.client.post("/auth/token", json={"username": "supervisor", "password": "supportflow-demo"}).json()["access_token"]
        support_headers = {"Authorization": f"Bearer {support_token}"}
        supervisor_headers = {"Authorization": f"Bearer {supervisor_token}"}
        self.assertEqual(plain_client.get("/auth/me", headers=supervisor_headers).json(), {"actor_id": "supervisor", "role": "supervisor"})
        created = plain_client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}, headers=support_headers).json()
        self.assertEqual(plain_client.get("/metrics", headers=support_headers).status_code, 403)
        self.assertEqual(plain_client.get("/metrics", headers=supervisor_headers).status_code, 200)
        approved = plain_client.post(f"/tickets/{created['ticket']['id']}/approval", json={"decision": "approve"}, headers=supervisor_headers).json()
        self.assertEqual(approved["approval"]["actor_id"], "supervisor")
        self.assertEqual(approved["approval"]["actor_role"], "supervisor")

    def test_experiment_center_is_supervisor_only_and_marks_missing_reports(self):
        plain_client = TestClient(self.client.app)
        support_token = self.client.post("/auth/token", json={"username": "support", "password": "supportflow-demo"}).json()["access_token"]
        supervisor_token = self.client.post("/auth/token", json={"username": "supervisor", "password": "supportflow-demo"}).json()["access_token"]
        self.assertEqual(plain_client.get("/experiments/center", headers={"Authorization": f"Bearer {support_token}"}).status_code, 403)
        response = plain_client.get("/experiments/center", headers={"Authorization": f"Bearer {supervisor_token}"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["knowledge_update"]["available"])

    def test_channel_event_is_normalized_and_idempotent(self):
        event = {
            "external_event_id": "mail-001", "sender_id": "customer-42",
            "subject": "退款咨询", "content": "购买后可以退款吗？",
        }
        headers = {"X-Integration-Key": "supportflow-integration-demo"}
        first = self.client.post("/channels/email/events", json=event, headers=headers)
        second = self.client.post("/channels/email/events", json=event, headers=headers)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["ticket"]["id"], second.json()["ticket"]["id"])
        self.assertEqual(first.json()["ticket"]["customer_id"], "email:customer-42")
        self.assertEqual(second.json()["job"]["source"], "email")
        self.assertEqual(second.json()["job"]["submitted_by"], "integration:email")
        self.assertEqual(self.client.post("/channels/wecom/events", json=event, headers={"X-Integration-Key": "wrong"}).status_code, 401)
        self.assertEqual(self.client.post("/channels/slack/events", json=event, headers=headers).status_code, 404)

    def test_customer_portal_creates_a_web_ticket_and_hides_unapproved_draft(self):
        created = self.client.post("/customer/tickets", json={
            "customer_id": "C-10086", "subject": "退款咨询", "content": "购买后可以退款吗？",
        })
        self.assertEqual(created.status_code, 202)
        public_ticket = created.json()
        self.assertEqual(public_ticket["status"], "pending_approval")
        self.assertNotIn("reply", public_ticket)
        ticket_id = public_ticket["ticket_id"]
        access_token = public_ticket["access_token"]
        stored = self.client.get(f"/tickets/{ticket_id}").json()
        self.assertEqual(stored["ticket"]["customer_id"], "web:C-10086")
        self.assertEqual(stored["job"]["source"], "web")
        self.assertEqual(self.client.get(f"/customer/tickets/{ticket_id}", params={"access_token": "x" * 24}).status_code, 404)
        self.client.post(f"/tickets/{ticket_id}/approval", json={"decision": "approve", "edited_reply": "已为您确认退款规则。"})
        resolved = self.client.get(f"/customer/tickets/{ticket_id}", params={"access_token": access_token}).json()
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["reply"], "已为您确认退款规则。")

    def test_async_ticket_is_persisted_as_processing_before_a_slow_workflow_finishes(self):
        started, release = Event(), Event()

        class BlockingReplyWriter(TemplateReplyWriter):
            def generate(self, ticket, triage, evidence):
                started.set()
                release.wait(timeout=5)
                return super().generate(ticket, triage, evidence)

        client = TestClient(create_app(
            TicketStore(f"{self.temporary_directory.name}/async.sqlite3"),
            SupportFlowGraph(knowledge=KnowledgeGrounder(), reply_writer=BlockingReplyWriter()),
            knowledge_directory=Path(self.temporary_directory.name) / "async-knowledge",
            run_async=True,
        ))
        client.headers.update({"X-Actor-Role": "administrator"})
        created = client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"})
        self.assertEqual(created.status_code, 202)
        self.assertEqual(created.json()["status"], "processing")
        self.assertTrue(started.wait(timeout=1))
        processing = client.get("/tickets/T-0001").json()
        self.assertEqual(processing["status"], "processing")
        self.assertEqual(processing["progress"]["stage"], "generating")
        release.set()
        deadline = monotonic() + 2
        while monotonic() < deadline and client.get("/tickets/T-0001").json()["status"] == "processing":
            sleep(0.02)
        self.assertEqual(client.get("/tickets/T-0001").json()["status"], "pending_approval")

    def test_startup_recovers_a_durable_processing_ticket(self):
        store = TicketStore(f"{self.temporary_directory.name}/recovery.sqlite3")
        ticket = Ticket("T-0001", "退款咨询", "购买后可以退款吗？")
        store.save_processing(asdict(ticket))
        app = create_app(
            store,
            SupportFlowGraph(knowledge=KnowledgeGrounder(), reply_writer=TemplateReplyWriter()),
            knowledge_directory=Path(self.temporary_directory.name) / "recovery-knowledge",
            run_async=True,
        )
        with TestClient(app) as client:
            client.headers.update({"X-Actor-Role": "administrator"})
            deadline = monotonic() + 2
            while monotonic() < deadline and client.get("/tickets/T-0001").json()["status"] == "processing":
                sleep(0.02)
            recovered = client.get("/tickets/T-0001").json()
        self.assertEqual(recovered["status"], "pending_approval")
        self.assertEqual(recovered["job"]["state"], "completed")
        self.assertEqual(recovered["job"]["attempts"], 1)
        self.assertEqual(recovered["job"]["recovery_count"], 1)

    def test_workbench_and_ticket_queue_can_be_loaded(self):
        self.assertEqual(self.client.get("/workbench").status_code, 200)
        created = self.client.post("/tickets", json={"subject": "功能报错", "content": "系统报错，功能不可用"})
        queued = self.client.get("/tickets", params={"status": "pending_approval"})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(queued.json()[0]["ticket"]["id"], created.json()["ticket"]["id"])

    def test_store_factory_keeps_sqlite_as_the_default_without_a_database_url(self):
        store = create_ticket_store(f"{self.temporary_directory.name}/factory.sqlite3")
        self.assertIsInstance(store, TicketStore)
        self.assertEqual(store.next_id(), "T-0001")

    def test_ticket_list_can_filter_by_category_and_risk(self):
        self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"})
        self.client.post("/tickets", json={"subject": "功能报错", "content": "系统报错，功能不可用"})
        refunds = self.client.get("/tickets", params={"category": "refund_policy", "risk_level": "medium"}).json()
        self.assertEqual(len(refunds), 1)
        self.assertEqual(refunds[0]["triage"]["category"], "refund_policy")

    def test_ticket_list_can_search_id_subject_content_and_customer(self):
        created = self.client.post("/tickets", json={
            "subject": "订单同步报错", "content": "同步订单时出现系统错误", "customer_id": "customer-search-001",
        }).json()
        self.assertEqual(self.client.get("/tickets", params={"q": "订单同步"}).json()[0]["ticket"]["id"], created["ticket"]["id"])
        self.assertEqual(self.client.get("/tickets", params={"q": "customer-search"}).json()[0]["ticket"]["id"], created["ticket"]["id"])
        self.assertEqual(self.client.get("/tickets", params={"q": created["ticket"]["id"]}).json()[0]["ticket"]["id"], created["ticket"]["id"])

    def test_metrics_expose_safety_and_human_review_signals(self):
        reviewable = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗"}).json()
        self.client.post(
            f"/tickets/{reviewable['ticket']['id']}/approval",
            json={"decision": "approve", "edited_reply": "人工确认后的回复"},
        )
        self.client.post("/tickets", json={"subject": "隐私投诉", "content": "我要投诉你们泄露了隐私"})
        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["ticket_count"], 2)
        self.assertEqual(metrics["escalation_rate"], 0.5)
        self.assertEqual(metrics["evidence_coverage"], 0.5)
        self.assertEqual(metrics["human_edit_rate"], 1.0)
        self.assertIn("p95", metrics["latency_ms"])
        self.assertIn("template", metrics["model_version_counts"])

    def test_health_and_operational_alerts_expose_safe_runtime_signals(self):
        self.assertEqual(self.client.get("/healthz").json()["status"], "ok")
        self.client.post("/tickets", json={"subject": "隐私投诉", "content": "我要投诉你们泄露了隐私"})
        alerts = self.client.get("/operational-alerts").json()
        self.assertIn("low_evidence_coverage", [alert["code"] for alert in alerts])

    def test_independent_human_annotations_have_a_queue_and_agreement_metric(self):
        ticket = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}).json()
        ticket_id = ticket["ticket"]["id"]
        support_headers = {"X-Actor-Role": "customer_support"}
        self.assertEqual(self.client.get("/annotations/queue", headers=support_headers).json()[0]["ticket"]["id"], ticket_id)
        first = self.client.post(f"/tickets/{ticket_id}/annotations", json={"label": "helpful", "note": "证据与边界清晰"}, headers=support_headers)
        second = self.client.post(f"/tickets/{ticket_id}/annotations", json={"label": "helpful"})
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(self.client.get("/annotations/queue", headers=support_headers).json(), [])
        quality = self.client.get("/annotations/quality").json()
        self.assertEqual(quality["annotation_count"], 2)
        self.assertEqual(quality["multi_annotated_ticket_count"], 1)
        self.assertEqual(quality["pairwise_agreement"], 1.0)

    def test_reviewer_feedback_is_persisted_and_aggregated(self):
        created = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}).json()
        ticket_id = created["ticket"]["id"]
        recorded = self.client.post(
            f"/tickets/{ticket_id}/feedback",
            json={"label": "needs_edit", "note": "需要更明确说明审核时效"},
        )
        self.assertEqual(recorded.status_code, 200)
        self.assertEqual(recorded.json()["feedback"]["label"], "needs_edit")
        self.assertEqual(recorded.json()["knowledge_gap"]["status"], "open")
        gaps = self.client.get("/knowledge/gaps").json()
        self.assertEqual(gaps[0]["count"], 1)
        self.assertEqual(gaps[0]["category"], "refund_policy")
        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["feedback_count"], 1)
        self.assertEqual(metrics["negative_feedback_rate"], 1.0)
        self.assertEqual(metrics["knowledge_gap_count"], 1)
        published = self.client.post("/knowledge/documents", json={
            "source_id": "KB-FEEDBACK-001", "title": "Refund review timeline", "category": "refund_policy",
            "content": "退款审核时效说明：收到完整订单信息后将在两个工作日内完成人工核验，并同步处理进度。",
        }).json()
        self.assertEqual(published["knowledge_gaps_marked_for_verification"], 1)
        self.assertEqual(self.client.get("/knowledge/gaps").json()[0]["status"], "verification")

    def test_knowledge_gap_outcome_compares_pre_and_post_feedback_cohorts(self):
        for index in range(3):
            ticket = self.client.post("/tickets", json={"subject": "退款咨询", "content": f"购买后可以退款吗？样本{index}"}).json()
            self.client.post(f"/tickets/{ticket['ticket']['id']}/feedback", json={"label": "needs_edit", "note": "补充审核时效"})
        self.client.post("/knowledge/documents", json={
            "source_id": "KB-OUTCOME-001", "title": "Refund review outcome", "category": "refund_policy",
            "content": "退款审核时效补充：收到完整订单信息后两个工作日内完成人工核验，并同步进度。",
        })
        for index in range(3):
            ticket = self.client.post("/tickets", json={"subject": "退款审核时效", "content": f"收到完整订单信息后，退款人工审核需要几个工作日？验证样本{index}"}).json()
            self.client.post(f"/tickets/{ticket['ticket']['id']}/feedback", json={"label": "helpful", "note": "时效说明清晰"})
        outcome = self.client.get("/knowledge/gap-outcomes").json()[0]
        self.assertEqual(outcome["verdict"], "ready_for_comparison")
        self.assertEqual(outcome["baseline"]["feedback_count"], 3)
        self.assertEqual(outcome["after"]["feedback_count"], 3)
        self.assertEqual(outcome["baseline"]["negative_feedback_rate"], 1.0)
        self.assertEqual(outcome["after"]["negative_feedback_rate"], 0.0)
        self.assertEqual(outcome["assessment"]["status"], "improved")
        self.assertEqual(outcome["assessment"]["negative_feedback_rate_delta"], -1.0)
        self.assertGreaterEqual(outcome["after"]["evaluated_source_citation_rate"], 0.5)

    def test_knowledge_gap_outcome_withholds_attribution_when_new_source_is_not_cited(self):
        for index in range(3):
            ticket = self.client.post("/tickets", json={"subject": "退款咨询", "content": f"购买后可以退款吗？归因前样本{index}"}).json()
            self.client.post(f"/tickets/{ticket['ticket']['id']}/feedback", json={"label": "needs_edit"})
        self.client.post("/knowledge/documents", json={
            "source_id": "KB-ATTRIBUTION-001", "title": "Refund review attribution", "category": "refund_policy",
            "content": "退款审核时效补充：收到完整订单信息后两个工作日内完成审核并同步进度。",
        })
        for index in range(3):
            ticket = self.client.post("/tickets", json={"subject": "退款咨询", "content": f"购买后可以退款吗？归因后样本{index}"}).json()
            self.client.post(f"/tickets/{ticket['ticket']['id']}/feedback", json={"label": "helpful"})
        outcome = self.client.get("/knowledge/gap-outcomes").json()[0]
        self.assertEqual(outcome["verdict"], "insufficient_attribution")
        self.assertEqual(outcome["assessment"]["status"], "insufficient_attribution")
        self.assertLess(outcome["after"]["evaluated_source_citation_rate"], 0.5)

    def test_roles_limit_medium_risk_approval_and_knowledge_publication(self):
        ticket = self.client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"}).json()
        staff_headers = {"X-Actor-Role": "customer_support"}
        denied = self.client.post(f"/tickets/{ticket['ticket']['id']}/approval", json={"decision": "approve"}, headers=staff_headers)
        self.assertEqual(denied.status_code, 403)
        allowed = self.client.post(f"/tickets/{ticket['ticket']['id']}/approval", json={"decision": "approve"}, headers={"X-Actor-Role": "supervisor"})
        self.assertEqual(allowed.status_code, 200)
        publish_denied = self.client.post("/knowledge/documents", json={
            "source_id": "KB-RBAC-001", "title": "RBAC policy", "category": "product_issue", "content": "This document is long enough to satisfy validation.",
        }, headers={"X-Actor-Role": "supervisor"})
        self.assertEqual(publish_denied.status_code, 403)

    def test_failed_ticket_can_be_resumed_without_losing_its_original_request(self):
        class FlakyWorkflow:
            def __init__(self):
                self.calls = 0
                self.working = SupportFlowGraph(knowledge=KnowledgeGrounder(), reply_writer=TemplateReplyWriter())

            def run(self, ticket, thread_id):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary provider issue")
                return self.working.run(ticket, thread_id)

        workflow = FlakyWorkflow()
        client = TestClient(create_app(
            TicketStore(f"{self.temporary_directory.name}/resume.sqlite3"),
            workflow,
            knowledge_directory=Path(self.temporary_directory.name) / "resume-knowledge",
            run_async=False,
        ))
        client.headers.update({"X-Actor-Role": "administrator"})
        failed = client.post("/tickets", json={"subject": "退款咨询", "content": "购买后可以退款吗？"})
        self.assertEqual(failed.status_code, 201)
        self.assertEqual(failed.json()["status"], "failed")
        self.assertTrue(failed.json()["failure"]["resumable"])
        resumed = client.post("/tickets/T-0001/resume", headers={"X-Actor-Role": "supervisor"})
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(resumed.json()["status"], "pending_approval")
        self.assertEqual(resumed.json()["ticket"]["subject"], "退款咨询")

    def test_operator_can_publish_and_reindex_a_knowledge_document(self):
        published = self.client.post("/knowledge/documents", json={
            "source_id": "KB-ORDER-SYNC-001",
            "title": "订单同步故障处理",
            "category": "product_issue",
            "content": "订单同步失败时，先收集订单编号、报错截图和发生时间，再升级订单服务团队排查。",
        })
        self.assertEqual(published.status_code, 201)
        self.assertGreaterEqual(published.json()["indexed_document_count"], 5)
        documents = self.client.get("/knowledge/documents").json()
        self.assertEqual(documents[0]["source_id"], "KB-ORDER-SYNC-001")
        ticket = self.client.post("/tickets", json={
            "subject": "订单同步报错", "content": "订单同步失败，系统报错怎么办？",
        }).json()
        self.assertIn("KB-ORDER-SYNC-001", [item["source_id"] for item in ticket["evidence"]])

    def test_knowledge_publication_creates_versioned_history(self):
        first = self.client.post("/knowledge/documents", json={
            "source_id": "KB-VERSION-001", "title": "Versioned policy", "category": "refund_policy", "content": "Version one explains the refund review process in sufficient detail.",
        })
        self.assertEqual(first.json()["version"], "v1")
        second = self.client.post("/knowledge/documents", json={
            "source_id": "KB-VERSION-001", "title": "Versioned policy", "category": "refund_policy", "content": "Version two updates the refund review process in sufficient detail.",
        })
        self.assertEqual(second.json()["version"], "v2")
        versions = self.client.get("/knowledge/documents/KB-VERSION-001/versions", headers={"X-Actor-Role": "supervisor"}).json()
        self.assertEqual([item["version"] for item in versions], ["v2", "v1"])
        ticket = self.client.post("/tickets", json={"subject": "refund review", "content": "refund review process"}).json()
        evidence = next(item for item in ticket["evidence"] if item["source_id"] == "KB-VERSION-001")
        self.assertEqual(evidence["source_version"], "v2")

    def test_administrator_can_restore_a_knowledge_version_with_audit_record(self):
        first = {"source_id": "KB-RESTORE-001", "title": "Restore policy", "category": "refund_policy", "content": "First policy wording with a clear two-day review timeline."}
        second = first | {"content": "Second policy wording that must be reverted after a review."}
        self.client.post("/knowledge/documents", json=first)
        self.client.post("/knowledge/documents", json=second)
        restored = self.client.post("/knowledge/documents/KB-RESTORE-001/restore", json={"version": "v1"})
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json()["version"], "v3")
        self.assertEqual(restored.json()["restored_from"], "v1")
        versions = self.client.get("/knowledge/documents/KB-RESTORE-001/versions").json()
        self.assertEqual([item["version"] for item in versions], ["v3", "v2", "v1"])
        audit = self.client.get("/knowledge/audit").json()
        self.assertEqual(audit[0]["action"], "restore")
        self.assertEqual(audit[0]["details"]["restored_from"], "v1")
