import os
import json
import unittest

from supportflow.domain import Ticket, TicketCategory, TicketStatus
from supportflow.business_tools import ExternalReadOnlyOrderTool, ReadOnlyCustomerTool, ReadOnlyOrderTool
from supportflow.evaluation import RetrievalCase, evaluate_retrieval, run_default_workflow_evaluation, run_retrieval_comparison
from supportflow.experiments import PromptExperiment
from supportflow.generation import DeepSeekReplyWriter, GeneratedReply, TemplateReplyWriter
from supportflow.knowledge import BailianEmbeddingProvider, KnowledgeGrounder, create_default_knowledge_grounder
from supportflow.public_dataset import TARGET_INTENT, build_public_corpus, records_from_rows
from supportflow.public_dataset_experiment import compare_public_knowledge_versions
from supportflow.kaggle_ticket_benchmark import audit_tickets, benchmark_tickets_from_rows
from supportflow.langgraph_runtime import SupportFlowGraph
from supportflow.workflow import SupportFlowWorkflow


class SupportFlowWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = SupportFlowWorkflow(reply_writer=TemplateReplyWriter())

    @staticmethod
    def graph():
        return SupportFlowGraph(knowledge=KnowledgeGrounder(), reply_writer=TemplateReplyWriter())

    def test_refund_question_requires_human_approval_and_cites_policy(self):
        result = self.workflow.run(Ticket("T-001", "退款咨询", "购买后可以退款吗？"))
        self.assertEqual(result.triage.category, TicketCategory.REFUND_POLICY)
        self.assertEqual(result.status, TicketStatus.PENDING_APPROVAL)
        self.assertEqual(result.evidence[0].source_id, "KB-REFUND-001")
        self.assertIn("人工审核", result.draft.reply)

    def test_high_risk_ticket_is_escalated_without_a_reply_draft(self):
        result = self.workflow.run(Ticket("T-002", "隐私投诉", "我要投诉你们泄露了我的隐私"))
        self.assertEqual(result.status, TicketStatus.ESCALATED)
        self.assertIsNone(result.draft)
        self.assertIn("高风险工单必须升级人工", result.quality.reasons)

    def test_langgraph_high_risk_ticket_skips_retrieval_and_generation(self):
        result = self.graph().run(
            Ticket("T-005", "隐私投诉", "我要投诉你们泄露了我的隐私"),
            "thread-005",
        )

        self.assertEqual(result.status, TicketStatus.ESCALATED)
        self.assertEqual([event.agent for event in result.trace], ["triage_agent", "quality_agent"])

    def test_identical_ticket_reuses_a_version_aware_reply_cache(self):
        class CountingWriter:
            def __init__(self):
                self.calls = 0

            def generate(self, ticket, triage, evidence):
                self.calls += 1
                return GeneratedReply("基于证据的测试草稿", "deepseek")

        writer = CountingWriter()
        workflow = SupportFlowWorkflow(reply_writer=writer)
        ticket = Ticket("T-CACHE", "退款咨询", "购买后可以退款吗？")
        first = workflow.run(ticket)
        second = workflow.run(ticket)
        self.assertEqual(writer.calls, 1)
        self.assertEqual(first.draft.generation_mode, "deepseek")
        self.assertEqual(second.draft.generation_mode, "deepseek_cache")

    def test_product_issue_generates_a_grounded_draft(self):
        result = self.workflow.run(Ticket("T-003", "功能报错", "系统报错，功能不可用"))
        self.assertEqual(result.triage.category, TicketCategory.PRODUCT_ISSUE)
        self.assertEqual(result.status, TicketStatus.PENDING_APPROVAL)
        self.assertIn("报错信息", result.draft.reply)

    def test_langgraph_runtime_persists_a_ticket_thread(self):
        graph = self.graph()
        result = graph.run(Ticket("T-004", "权限问题", "账号没有权限，无法登录"), "thread-004")

        self.assertEqual(result.triage.category, TicketCategory.ACCOUNT_ACCESS)
        self.assertEqual(result.status, TicketStatus.PENDING_APPROVAL)
        self.assertEqual([event.agent for event in result.trace], [
            "triage_agent", "customer_context_agent", "order_context_agent", "knowledge_agent", "resolution_agent", "quality_agent",
        ])

    def test_customer_context_agent_adds_readonly_crm_evidence_without_mutation(self):
        graph = self.graph()
        result = graph.run(Ticket("T-CRM-001", "退款咨询", "购买后可以退款吗？", customer_id="web:C-10086"), thread_id="T-CRM-001")
        crm_evidence = next(item for item in result.evidence if item.source_id == "CRM-web:C-10086")
        self.assertEqual(crm_evidence.source_version, "live_readonly")
        self.assertIn("会员客户", crm_evidence.excerpt)
        self.assertFalse(hasattr(ReadOnlyCustomerTool(), "update"))

    def test_order_context_agent_adds_readonly_order_evidence_without_exposing_mutations(self):
        graph = self.graph()
        result = graph.run(Ticket("T-ORDER-001", "退款进度", "订单 ORD-1001 的退款进度是什么？"), thread_id="T-ORDER-001")
        order_evidence = next(item for item in result.evidence if item.source_id == "ORDER-ORD-1001")
        self.assertEqual(order_evidence.source_version, "live_readonly")
        self.assertIn("退款审核中", order_evidence.excerpt)
        self.assertFalse(hasattr(ReadOnlyOrderTool(), "refund"))

    def test_external_order_adapter_only_calls_the_support_context_endpoint(self):
        captured = {}

        def fake_request(request, timeout_seconds):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout_seconds
            return b'{"payment_status":"paid","refund_status":"reviewing","summary":"Manual review is pending."}'

        tool = ExternalReadOnlyOrderTool("https://orders.example.test/v1", "demo-token", timeout_seconds=1.5, request=fake_request)
        snapshot = tool.lookup_from_text("Please check order ORD-4321")
        self.assertEqual(snapshot.source_version, "external_readonly")
        self.assertEqual(captured, {
            "url": "https://orders.example.test/v1/orders/ORD-4321/support-context",
            "method": "GET",
            "authorization": "Bearer demo-token",
            "timeout": 1.5,
        })
        self.assertFalse(hasattr(tool, "refund"))

    def test_prompt_experiment_assignment_is_stable_and_records_the_treatment_prompt_version(self):
        experiment = PromptExperiment("reply-clarity-v2")
        arm = experiment.assign("T-EXPERIMENT-001")
        self.assertEqual(experiment.assign("T-EXPERIMENT-001"), arm)
        ticket = Ticket("T-EXPERIMENT-001", "产品故障", "系统报错，功能不可用", experiment_id=experiment.experiment_id, experiment_arm="treatment")
        result = self.graph().run(ticket, thread_id=ticket.id)
        resolution_trace = next(item for item in result.trace if item.agent == "resolution_agent")
        self.assertEqual(resolution_trace.metadata["prompt_version"], "support-reply-v2")
        self.assertIn("人工核验", result.draft.reply)

    def test_vector_retrieval_evaluation_has_expected_policy_hits(self):
        report = evaluate_retrieval(KnowledgeGrounder(), [
            RetrievalCase("购买后可以退款吗", TicketCategory.REFUND_POLICY, "KB-REFUND-001"),
            RetrievalCase("退款审核后多久到账", TicketCategory.REFUND_POLICY, "KB-REFUND-002"),
            RetrievalCase("账号没有权限登录", TicketCategory.ACCOUNT_ACCESS, "KB-ACCOUNT-001"),
            RetrievalCase("功能报错无法使用", TicketCategory.PRODUCT_ISSUE, "KB-PRODUCT-001"),
        ])
        self.assertEqual(report.hit_at_1, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)

    def test_deepseek_adapter_uses_the_compatible_endpoint(self):
        writer = DeepSeekReplyWriter("dpsk-test-key")
        self.assertEqual(str(writer.client.base_url), "https://api.deepseek.com")

    def test_bailian_embedding_adapter_uses_the_configured_compatible_endpoint(self):
        provider = BailianEmbeddingProvider("sk-test-key", "https://example.com/compatible-mode/v1")
        self.assertEqual(str(provider.client.base_url), "https://example.com/compatible-mode/v1/")

    def test_bailian_accepts_a_full_compatible_url_in_workspace_setting(self):
        previous = {name: os.environ.get(name) for name in (
            "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_WORKSPACE_ID",
            "SUPPORTFLOW_RETRIEVAL_MODE",
        )}
        try:
            os.environ["DASHSCOPE_API_KEY"] = "sk-test-key"
            os.environ["SUPPORTFLOW_RETRIEVAL_MODE"] = "semantic"
            os.environ.pop("DASHSCOPE_BASE_URL", None)
            os.environ["DASHSCOPE_WORKSPACE_ID"] = "https://example.com/compatible-mode/v1"
            grounder = create_default_knowledge_grounder()
            self.assertEqual(
                str(grounder.semantic_index.provider.client.base_url),
                "https://example.com/compatible-mode/v1/",
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_product_evaluation_set_passes_routing_retrieval_and_safety_gates(self):
        report = run_default_workflow_evaluation()
        self.assertGreaterEqual(report.cases, 30)
        self.assertEqual(report.category_accuracy, 1.0)
        self.assertEqual(report.status_accuracy, 1.0)
        self.assertEqual(report.evidence_hit_rate, 1.0)
        self.assertEqual(report.safety_case_pass_rate, 1.0)

    def test_local_retrieval_experiment_reports_quality_latency_and_mode(self):
        report = run_retrieval_comparison()[0]
        self.assertEqual(report.name, "local_tfidf")
        self.assertGreaterEqual(report.cases, 19)
        self.assertEqual(report.hit_at_1, 1.0)
        self.assertEqual(report.retrieval_mode_counts, {"local_tfidf": report.cases})

    def test_unsafe_model_commitment_is_blocked_by_quality_agent(self):
        class UnsafeWriter:
            def generate(self, ticket, triage, evidence):
                return GeneratedReply("您的退款已成功，我们已经退款。", "deepseek")

        result = SupportFlowWorkflow(reply_writer=UnsafeWriter()).run(
            Ticket("T-006", "退款咨询", "购买后可以退款吗"),
        )
        self.assertEqual(result.status, TicketStatus.ESCALATED)
        self.assertIn("草稿包含模型无权做出的执行或补偿承诺", result.quality.reasons)

    def test_public_corpus_keeps_blind_test_out_of_both_knowledge_versions(self):
        rows = [
            {"instruction": f"Where is my refund {index}?", "response": "Check the refund status.", "category": "refund", "intent": TARGET_INTENT}
            for index in range(5)
        ] + [{"instruction": "Can I return an order?", "response": "Follow the return policy.", "category": "refund", "intent": "get_refund"}]
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            manifest = build_public_corpus(records_from_rows(rows), Path(directory), test_size=2, addition_size=2)
            test_ids = {json.loads(line)["record_id"] for line in (Path(directory) / "blind_test_track_refund.jsonl").read_text().splitlines()}
            knowledge_ids = {
                json.loads(line)["record_id"]
                for name in ("knowledge_v1.jsonl", "knowledge_v2_additions.jsonl")
                for line in (Path(directory) / name).read_text().splitlines()
            }
            comparison = compare_public_knowledge_versions(Path(directory))
        self.assertEqual(manifest.leakage_check, "passed")
        self.assertFalse(test_ids & knowledge_ids)
        self.assertGreaterEqual(comparison.v2_target_intent_at_1, comparison.v1_target_intent_at_1)

    def test_kaggle_ticket_audit_keeps_pii_out_and_reports_human_review_capture(self):
        rows = [{
            "Ticket ID": "12", "Ticket Subject": "Account access", "Ticket Description": "I cannot log in to my account.",
            "Ticket Type": "Technical", "Ticket Priority": "High", "Ticket Status": "Open", "Ticket Channel": "Email",
            "Resolution": "Reset the password", "Customer Satisfaction Rating": "3",
            "Customer Name": "Must not be retained", "Customer Email": "private@example.test",
        }]
        tickets = benchmark_tickets_from_rows(rows)
        report = audit_tickets(tickets)
        self.assertEqual(tickets[0].ticket_id, "KAGGLE-12")
        self.assertFalse(hasattr(tickets[0], "customer_email"))
        self.assertEqual(report.high_priority_cases, 1)
        self.assertEqual(report.high_priority_sent_to_human_review, 1)


    def test_english_account_ticket_is_grounded_and_keeps_human_approval(self):
        result = self.workflow.run(Ticket("T-EN-ACCOUNT", "Login denied", "My account access is denied after login and I need help."))
        self.assertEqual(result.triage.category, TicketCategory.ACCOUNT_ACCESS)
        self.assertEqual(result.status, TicketStatus.PENDING_APPROVAL)
        self.assertTrue(any(item.source_id == "KB-ACCOUNT-EN-001" for item in result.evidence))
        self.assertIn("do not share passwords", result.draft.reply.lower())

    def test_english_refund_ticket_is_grounded_and_explicitly_requires_human_review(self):
        result = self.workflow.run(Ticket("T-EN-REFUND", "Refund status", "Where can I track my refund status?"))
        self.assertEqual(result.triage.category, TicketCategory.REFUND_POLICY)
        self.assertEqual(result.status, TicketStatus.PENDING_APPROVAL)
        self.assertTrue(any(item.source_id == "KB-REFUND-EN-001" for item in result.evidence))
        self.assertIn("human review", result.draft.reply.lower())

    def test_english_privacy_complaint_still_skips_retrieval_and_generation(self):
        result = self.workflow.run(Ticket("T-EN-PRIVACY", "Privacy complaint", "I want to complain about a privacy data leak."))
        self.assertEqual(result.status, TicketStatus.ESCALATED)
        self.assertIsNone(result.draft)


if __name__ == "__main__":
    unittest.main()
