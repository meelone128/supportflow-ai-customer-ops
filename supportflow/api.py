"""Local HTTP interface for the first SupportFlow workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import hmac
import os
from pathlib import Path
import secrets
from tempfile import gettempdir
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

from supportflow.auth import Actor, AuthenticationError, JwtAuthenticator, ROLES
from supportflow.dispatch import create_task_dispatcher
from supportflow.domain import Ticket, TicketCategory, TicketStatus
from supportflow.execution import TicketExecutor
from supportflow.experiments import PromptExperiment
from supportflow.experiment_center import ExperimentCenter
from supportflow.knowledge import ManagedKnowledgeRepository, create_default_knowledge_grounder
from supportflow.langgraph_runtime import SupportFlowGraph
from supportflow.metrics import calculate_metrics
from supportflow.observability import calculate_operational_alerts
from supportflow.storage import TicketStore, create_ticket_store


ActorRole = Literal["customer_support", "supervisor", "administrator"]


def require_role(actor_role: ActorRole, *allowed: ActorRole) -> None:
    if actor_role not in allowed:
        raise HTTPException(status_code=403, detail="This action is not permitted for the current role")



class CreateTicketRequest(BaseModel):
    subject: str = Field(min_length=1)
    content: str = Field(min_length=1)
    customer_id: str = "demo-customer"


class CustomerPortalTicketRequest(BaseModel):
    customer_id: str = Field(min_length=2, max_length=80)
    subject: str = Field(default="客户咨询", min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4000)


class AuthTokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=200)


class ChannelEventRequest(BaseModel):
    external_event_id: str = Field(min_length=1, max_length=128)
    sender_id: str = Field(min_length=1, max_length=120)
    subject: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    decision: str = Field(pattern="^(approve|escalate)$")
    edited_reply: str | None = None


class FeedbackRequest(BaseModel):
    label: Literal["helpful", "needs_edit", "incorrect", "unsafe"]
    note: str | None = Field(default=None, max_length=500)


class AnnotationRequest(BaseModel):
    label: Literal["helpful", "needs_edit", "incorrect", "unsafe"]
    note: str | None = Field(default=None, max_length=500)


class CreateKnowledgeDocumentRequest(BaseModel):
    source_id: str = Field(pattern="^[A-Z0-9-]{3,64}$")
    title: str = Field(min_length=1, max_length=120)
    category: TicketCategory
    content: str = Field(min_length=20)


class RestoreKnowledgeDocumentRequest(BaseModel):
    version: str = Field(pattern=r"^v[1-9][0-9]*$")


def create_app(
    store: TicketStore | None = None,
    workflow: SupportFlowGraph | None = None,
    knowledge_directory: Path | None = None,
    experiment_project_root: Path | None = None,
    run_async: bool = True,
) -> FastAPI:
    data_dir = Path(os.environ.get("SUPPORTFLOW_DATA_DIR", Path(gettempdir()) / "supportflow"))
    store = store or create_ticket_store(data_dir / "supportflow.sqlite3", os.getenv("SUPPORTFLOW_DATABASE_URL"))
    repository = ManagedKnowledgeRepository(knowledge_directory or data_dir / "knowledge")
    workflow = workflow or SupportFlowGraph(knowledge=create_default_knowledge_grounder(managed_directory=repository.directory))
    executor = TicketExecutor(store, workflow)
    dispatcher = create_task_dispatcher(executor) if run_async else None
    prompt_experiment = PromptExperiment.from_environment()
    submission_lock = Lock()
    authenticator = JwtAuthenticator.from_environment()
    allow_demo_role_header = os.getenv("SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER", "true").lower() == "true"
    integration_key = os.getenv("SUPPORTFLOW_INTEGRATION_KEY", "supportflow-integration-demo")
    experiment_center = ExperimentCenter(experiment_project_root or Path(__file__).parent.parent)

    def current_actor(authorization: str | None, x_actor_role: str | None) -> Actor:
        if authorization:
            try:
                return authenticator.verify(authorization)
            except AuthenticationError as error:
                raise HTTPException(status_code=401, detail=str(error)) from error
        if allow_demo_role_header and x_actor_role in ROLES:
            return Actor(f"demo-{x_actor_role}", x_actor_role)
        raise HTTPException(status_code=401, detail="Bearer token required")

    def queue_ticket(ticket: Ticket, previous: dict | None = None, reset_attempts: bool = False, idempotency_key: str | None = None, source: str = "workbench", submitted_by: str = "unknown", customer_access_token_hash: str | None = None) -> dict:
        processing = store.save_processing(asdict(ticket), previous=previous, reset_attempts=reset_attempts, idempotency_key=idempotency_key, source=source, submitted_by=submitted_by)
        if customer_access_token_hash:
            processing["customer_portal"] = {"access_token_hash": customer_access_token_hash}
            store.save(processing)
        if dispatcher is None:
            raise RuntimeError("A dispatcher is required for asynchronous ticket submission")
        dispatcher.dispatch(ticket.id)
        return processing

    def recover_incomplete_tickets() -> None:
        if not run_async:
            return
        for payload in store.list(status=TicketStatus.PROCESSING.value):
            if dispatcher is not None:
                dispatcher.dispatch(payload["ticket"]["id"], recovered_from_restart=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        recover_incomplete_tickets()
        yield

    app = FastAPI(title="SupportFlow", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        store.list(status=TicketStatus.PROCESSING.value)
        return {"status": "ok", "task_backend": os.getenv("SUPPORTFLOW_TASK_BACKEND", "local")}

    @app.post("/auth/token")
    def issue_token(request: AuthTokenRequest):
        try:
            token = authenticator.issue(request.username, request.password)
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        return {"access_token": token, "token_type": "bearer", "expires_in_minutes": authenticator.ttl_minutes}

    @app.get("/auth/me")
    def authenticated_profile(authorization: str | None = Header(default=None)):
        actor = current_actor(authorization, None)
        return {"actor_id": actor.actor_id, "role": actor.role}

    def submit_ticket(request: CreateTicketRequest, idempotency_key: str | None, source: str, submitted_by: str, customer_access_token_hash: str | None = None):
        with submission_lock:
            if idempotency_key and (existing := store.get_by_idempotency_key(idempotency_key)):
                return JSONResponse(content=existing, status_code=200)
            ticket_id = store.next_id()
            experiment_arm = prompt_experiment.assign(ticket_id) if prompt_experiment else None
            ticket = Ticket(id=ticket_id, experiment_id=prompt_experiment.experiment_id if prompt_experiment else None, experiment_arm=experiment_arm, **request.model_dump())
            if run_async:
                return JSONResponse(content=queue_ticket(ticket, idempotency_key=idempotency_key, source=source, submitted_by=submitted_by, customer_access_token_hash=customer_access_token_hash), status_code=202)
            processing = store.save_processing(asdict(ticket), idempotency_key=idempotency_key, source=source, submitted_by=submitted_by)
            if customer_access_token_hash:
                processing["customer_portal"] = {"access_token_hash": customer_access_token_hash}
                store.save(processing)
            return executor.execute(ticket.id)

    def customer_ticket_view(payload: dict) -> dict:
        """Expose only customer-safe progress; internal traces and drafts stay in the workbench."""
        status = payload["status"]
        messages = {
            TicketStatus.PROCESSING.value: "正在分析您的问题，请稍候。",
            TicketStatus.PENDING_APPROVAL.value: "已生成处理建议，正在等待客服专员确认。",
            TicketStatus.ESCALATED.value: "该问题已转交客服专员处理，我们会尽快回复。",
            TicketStatus.RESOLVED.value: "您的问题已处理完成。",
            TicketStatus.FAILED.value: "当前处理暂时异常，客服专员会继续跟进。",
        }
        response = {"ticket_id": payload["ticket"]["id"], "status": status, "message": messages[status]}
        if status == TicketStatus.RESOLVED.value and (reply := (payload.get("draft") or {}).get("reply")):
            response["reply"] = reply
        return response

    @app.post("/tickets", status_code=201)
    def create_ticket(request: CreateTicketRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128), authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        normalized_key = idempotency_key.strip() if idempotency_key else None
        return submit_ticket(request, normalized_key, "workbench", actor.actor_id)

    @app.post("/customer/tickets", status_code=202)
    def create_customer_ticket(request: CustomerPortalTicketRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128)):
        customer_id = f"web:{request.customer_id.strip()}"
        ticket = CreateTicketRequest(subject=request.subject, content=request.content, customer_id=customer_id)
        normalized_key = idempotency_key.strip() if idempotency_key else None
        access_token = (
            hmac.new(integration_key.encode(), f"customer-portal:{normalized_key}".encode(), hashlib.sha256).hexdigest()
            if normalized_key else secrets.token_urlsafe(24)
        )
        access_token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        result = submit_ticket(ticket, normalized_key, "web", f"customer:{request.customer_id.strip()}", access_token_hash)
        payload = result.body if isinstance(result, JSONResponse) else result
        if isinstance(payload, bytes):
            import json
            payload = json.loads(payload)
        return customer_ticket_view(payload) | {"access_token": access_token}

    @app.get("/customer/tickets/{ticket_id}")
    def get_customer_ticket(ticket_id: str, access_token: str = Query(min_length=20, max_length=200)):
        payload = store.get(ticket_id)
        expected_hash = (payload or {}).get("customer_portal", {}).get("access_token_hash", "")
        received_hash = hashlib.sha256(access_token.encode()).hexdigest()
        if payload is None or not expected_hash or not hmac.compare_digest(expected_hash, received_hash):
            raise HTTPException(status_code=404, detail="Ticket not found")
        return customer_ticket_view(payload)

    @app.post("/channels/{channel}/events", status_code=202)
    def ingest_channel_event(channel: str, event: ChannelEventRequest, x_integration_key: str | None = Header(default=None)):
        if channel not in {"web", "email", "wecom"}:
            raise HTTPException(status_code=404, detail="Unsupported channel")
        if not x_integration_key or not hmac.compare_digest(x_integration_key, integration_key):
            raise HTTPException(status_code=401, detail="Invalid integration key")
        request = CreateTicketRequest(subject=event.subject, content=event.content, customer_id=f"{channel}:{event.sender_id}")
        return submit_ticket(request, f"channel:{channel}:{event.external_event_id}", channel, f"integration:{channel}")

    @app.get("/tickets")
    def list_tickets(
        status: str | None = Query(default=None),
        category: TicketCategory | None = Query(default=None),
        risk_level: str | None = Query(default=None, pattern="^(low|medium|high)$"),
        q: str | None = Query(default=None, min_length=1, max_length=120),
        authorization: str | None = Header(default=None),
        x_actor_role: str | None = Header(default=None),
    ):
        current_actor(authorization, x_actor_role)
        tickets = store.list(status=status)
        if category:
            tickets = [ticket for ticket in tickets if ticket["triage"]["category"] == category.value]
        if risk_level:
            tickets = [ticket for ticket in tickets if ticket["triage"]["risk_level"] == risk_level]
        if q:
            query = q.casefold()
            tickets = [
                ticket for ticket in tickets
                if query in ticket["ticket"]["id"].casefold()
                or query in ticket["ticket"]["subject"].casefold()
                or query in ticket["ticket"]["content"].casefold()
                or query in ticket["ticket"]["customer_id"].casefold()
            ]
        return tickets

    @app.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: str, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        current_actor(authorization, x_actor_role)
        payload = store.get(ticket_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return payload

    @app.get("/metrics")
    def metrics(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return calculate_metrics(store.list())

    @app.get("/operational-alerts")
    def operational_alerts(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return calculate_operational_alerts(calculate_metrics(store.list()))

    @app.get("/experiments/prompt")
    def prompt_experiment_results(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        if prompt_experiment is None:
            return {"enabled": False, "results": None}
        return {"enabled": True, "results": store.prompt_experiment_results(prompt_experiment.experiment_id)}

    @app.get("/experiments/center")
    def experiment_center_snapshot(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return experiment_center.snapshot()

    @app.get("/knowledge/documents")
    def list_knowledge_documents(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        current_actor(authorization, x_actor_role)
        return [asdict(document) | {"category": document.category.value} for document in repository.list_documents()]

    @app.post("/knowledge/documents", status_code=201)
    def publish_knowledge_document(request: CreateKnowledgeDocumentRequest, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "administrator")
        document = repository.save(**request.model_dump())
        workflow.refresh_knowledge(create_default_knowledge_grounder(managed_directory=repository.directory))
        store.record_knowledge_audit("publish", document.source_id, document.version, actor.actor_id, actor.role)
        marked_for_verification = store.mark_knowledge_gaps_for_verification(request.category.value, request.source_id)
        return {"source_id": request.source_id, "version": document.version, "indexed_document_count": len(workflow.knowledge.index.chunks), "knowledge_gaps_marked_for_verification": marked_for_verification}

    @app.get("/knowledge/documents/{source_id}/versions")
    def list_knowledge_versions(source_id: str, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return [asdict(document) | {"category": document.category.value} for document in repository.list_versions(source_id)]

    @app.post("/knowledge/documents/{source_id}/restore")
    def restore_knowledge_version(source_id: str, request: RestoreKnowledgeDocumentRequest, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "administrator")
        try:
            restored, target = repository.restore(source_id, request.version)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        workflow.refresh_knowledge(create_default_knowledge_grounder(managed_directory=repository.directory))
        store.record_knowledge_audit("restore", restored.source_id, restored.version, actor.actor_id, actor.role, {"restored_from": target.version})
        return {"source_id": restored.source_id, "version": restored.version, "restored_from": target.version, "indexed_document_count": len(workflow.knowledge.index.chunks)}

    @app.get("/knowledge/audit")
    def list_knowledge_audit(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return store.list_knowledge_audit()

    @app.get("/knowledge/gaps")
    def list_knowledge_gaps(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return store.list_knowledge_gaps()

    @app.get("/knowledge/gap-outcomes")
    def list_knowledge_gap_outcomes(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return store.list_knowledge_gap_outcomes()

    @app.post("/tickets/{ticket_id}/approval")
    def decide_ticket(ticket_id: str, request: ApprovalRequest, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        payload = store.get(ticket_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if payload["status"] != TicketStatus.PENDING_APPROVAL.value:
            raise HTTPException(status_code=409, detail="Ticket is not awaiting approval")
        if request.decision == "approve":
            allowed = ("customer_support", "supervisor", "administrator") if payload["triage"]["risk_level"] == "low" else ("supervisor", "administrator")
            require_role(actor.role, *allowed)
        return store.decide(ticket_id, request.decision, request.edited_reply, actor.role, actor.actor_id)

    @app.post("/tickets/{ticket_id}/feedback")
    def record_ticket_feedback(ticket_id: str, request: FeedbackRequest, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        payload = store.record_feedback(ticket_id, request.label, request.note, actor.role, actor.actor_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return payload

    @app.get("/annotations/queue")
    def annotation_queue(limit: int = Query(default=20, ge=1, le=100), authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        return store.list_annotation_queue(actor.actor_id, limit)

    @app.post("/tickets/{ticket_id}/annotations", status_code=201)
    def annotate_ticket(ticket_id: str, request: AnnotationRequest, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        annotation = store.record_annotation(ticket_id, request.label, request.note, actor.actor_id, actor.role)
        if annotation is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return annotation

    @app.get("/annotations/quality")
    def annotation_quality(authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        return store.annotation_quality()

    @app.post("/tickets/{ticket_id}/resume")
    def resume_ticket(ticket_id: str, authorization: str | None = Header(default=None), x_actor_role: str | None = Header(default=None)):
        actor = current_actor(authorization, x_actor_role)
        require_role(actor.role, "supervisor", "administrator")
        previous = store.get(ticket_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if previous["status"] != TicketStatus.FAILED.value:
            raise HTTPException(status_code=409, detail="Only failed tickets can be resumed")
        ticket = Ticket(**previous["ticket"])
        if run_async:
            return JSONResponse(content=queue_ticket(ticket, previous=previous, reset_attempts=True), status_code=202)
        store.save_processing(asdict(ticket), previous=previous, reset_attempts=True)
        return executor.execute(ticket.id)

    @app.get("/workbench", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    def workbench():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/customer", include_in_schema=False)
    def customer_portal():
        return FileResponse(Path(__file__).parent / "static" / "customer.html")

    return app


app = create_app()
