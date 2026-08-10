# SupportFlow

SupportFlow is a production-oriented multi-agent workflow for enterprise support tickets.

## What is implemented

- Triage agent classifies a ticket and assigns a risk level.
- Knowledge agent ingests Markdown policy documents and retrieves cited evidence through a local TF-IDF vector index.
- Resolution agent creates a reply draft only; it cannot perform actions.
- Quality agent checks risk, evidence, and policy constraints.
- Medium-risk tickets enter human approval; high-risk tickets are escalated.
- SQLite persists the ticket, agent trace, evidence, reply draft, approval decision, and reviewer feedback.
- Demo RBAC uses the `X-Actor-Role` header: `customer_support` can approve low-risk drafts, `supervisor` can approve medium-risk work and view metrics, and only `administrator` can publish knowledge.
- Operator-published knowledge is versioned. Updating a `source_id` archives the prior Markdown as `vN`, makes `vN+1` current for retrieval, and exposes history through `GET /knowledge/documents/{source_id}/versions`.
- Every production graph trace records per-agent latency plus retrieval mode, generation mode, model version, and prompt version; `/metrics` aggregates P50/P95 latency and retriever/model usage.
- Background work is durably recorded in SQLite with queue/running/completed states, attempt counts, restart-recovery counts, and a bounded three-attempt policy. On service startup, unfinished tickets are reclaimed; after the limit, the original ticket remains available for an explicit supervisor retry.
- A read-only order-context agent recognizes supported order IDs and adds payment/refund status as auditable evidence. The tool surface intentionally has no refund, cancellation, or account-mutation method, so the model cannot perform business actions.
- The order-context seam can call a real order service when configured. It only invokes `GET /orders/{order_id}/support-context`, validates the support-safe response shape, and never falls back to demo order data if the external lookup fails.
- The CRM-context seam reads a minimal customer support profile before retrieval. It only invokes `GET /customers/{customer_id}/support-context` and exposes customer tier, open-ticket count, and a support-safe summary as auditable evidence; it has no customer-update or account-mutation capability.
- Publishing knowledge for an open feedback-driven gap starts a verification period instead of closing the gap. Supervisors can compare same-category ticket cohorts from before and after that timestamp through `GET /knowledge/gap-outcomes`; the product withholds an optimization verdict until each cohort has at least three feedback samples and enough post-publication tickets actually cite the evaluated source.
- Knowledge versions are immutable in practice: restoring a historical version creates a new current version, refreshes retrieval, and records an actor-attributed audit event instead of overwriting document history.

The workflow is implemented as a LangGraph state graph. Policy routing is deterministic; retrieval is a local sparse-vector RAG implementation that needs no API key or Docker for a demo. `KnowledgeGrounder` is the seam where Qdrant or a hybrid retriever can later be substituted.

When `OPENAI_API_KEY` is set in `.dev.env` or `.env`, the production graph uses the OpenAI Responses API to draft the reply from retrieved evidence. When `DEEPSEEK_API_KEY` is set (or the existing `OPENAI_API_KEY` starts with `dpsk-`), it instead uses DeepSeek's OpenAI-compatible Chat Completions API with `deepseek-v4-flash`. Tests inject the deterministic template writer, and any provider failure automatically returns that safe template instead. This keeps test runs free of API calls and makes the degraded mode explicit in `draft.generation_mode`.

SupportFlow defaults to local TF-IDF retrieval because the current small, keyword-heavy corpus is faster and more accurate in the recorded evaluation. To explicitly enable semantic retrieval, set `SUPPORTFLOW_RETRIEVAL_MODE=semantic` together with `DASHSCOPE_API_KEY` and either `DASHSCOPE_BASE_URL` or `DASHSCOPE_WORKSPACE_ID`, plus `DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4`. `DASHSCOPE_WORKSPACE_ID` accepts either a bare workspace ID or the complete OpenAI-compatible URL copied from Bailian. Vector and LLM calls have bounded timeouts and automatically fall back to the local index or safe template, so a provider outage cannot leave a ticket request hanging.

## Run locally

From the repository root (Python 3.11+):

```powershell
python -m unittest discover -s supportflow\tests -v
python -m supportflow.run_evals
python -m supportflow.run_retrieval_experiment
python -m uvicorn supportflow.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API.

By default, the SQLite file is stored in the system temporary directory. Set `SUPPORTFLOW_DATA_DIR` to a mounted or durable directory when deploying.

For PostgreSQL, set `SUPPORTFLOW_DATABASE_URL` to a full connection URL such as `postgresql://user:strong-password@db-host:5432/supportflow`. This is opt-in: without it, SupportFlow remains on SQLite for local development. The PostgreSQL adapter preserves the same ticket, audit, annotation, idempotency, and recovery contract as SQLite; install the `psycopg` production dependency before enabling it.

## Run with Docker

The Docker setup starts the SupportFlow API, Redis broker, and a separate Celery worker; the workbench is available on `http://127.0.0.1:8765/workbench` and SQLite data plus operator-managed knowledge persist in `./data/supportflow`.

```powershell
# Optional: keep provider keys and local credentials in .dev.env, never commit this file.
docker compose --env-file .dev.env up --build -d
docker compose ps
docker compose logs -f supportflow
```

Stop the local deployment with `docker compose down`. Before any non-local deployment, set a unique `SUPPORTFLOW_JWT_SECRET`, `SUPPORTFLOW_INTEGRATION_KEY`, and `SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER=false`. The Compose file intentionally provides safe local defaults only.

`SUPPORTFLOW_TASK_BACKEND=local` is the default when starting Python directly and uses a local background thread for the smallest demo path. Docker Compose sets `SUPPORTFLOW_TASK_BACKEND=celery`: the API first persists a processing ticket, then sends only its ticket ID to Redis; the worker claims and executes that durable ticket. This keeps retries, service restarts, and horizontal worker scaling out of the HTTP process.

## Deploy a portfolio demo to Render

The repository includes a [`render.yaml`](../render.yaml) Blueprint. Creating a Blueprint from the repository provisions one public web service and one managed PostgreSQL database in Singapore. It uses the Docker image in this repository, passes Render's injected `PORT`, and reads the database URL from the managed database rather than committing credentials.

This is intentionally a **demo deployment**: it uses `SUPPORTFLOW_TASK_BACKEND=local` so a single web service can be shown without paying for a background worker. The production-shaped API + Redis + Celery worker topology remains in `docker-compose.yml`; do not describe the public demo as horizontally scaled asynchronous processing.

The Blueprint generates a random operator password rather than using the local default. Before sharing an operator-workbench demo, either retrieve that generated value securely or add explicit accounts using `SUPPORTFLOW_AUTH_USERS_JSON` in Render's environment settings; keep `SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER=false`. Never set provider keys, CRM credentials, or `.dev.env` values in Git.

## Deploy a no-card portfolio demo to Vercel

The repository also includes `api/index.py`, `requirements.txt`, and `vercel.json` for a Vercel FastAPI deployment. Set the following values in Vercel Project Settings rather than in Git:

- `SUPPORTFLOW_TASK_BACKEND=inline`
- `SUPPORTFLOW_DATA_DIR=/tmp/supportflow`
- `SUPPORTFLOW_RETRIEVAL_MODE=local`
- `SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER=false`
- a unique `SUPPORTFLOW_JWT_SECRET`, `SUPPORTFLOW_INTEGRATION_KEY`, and `SUPPORTFLOW_DEMO_PASSWORD`

`inline` is a serverless adapter: it completes a ticket inside the HTTP request instead of starting a background thread. Vercel's filesystem is temporary, so ticket history and published knowledge can reset between function instances. This is deliberately a portfolio demo deployment, not a replacement for the Docker + Redis/Celery + PostgreSQL production topology.

## Authentication

Sensitive API routes accept an `Authorization: Bearer <JWT>` token. Obtain one through `POST /auth/token`; the token contains the user identity and role, and approval/feedback audit records persist both values. For a real deployment, set `SUPPORTFLOW_JWT_SECRET` to a long random secret, supply accounts through `SUPPORTFLOW_AUTH_USERS_JSON` (a username-to-password-and-role JSON object), and set `SUPPORTFLOW_ALLOW_DEMO_ROLE_HEADER=false`.

The local workbench retains `X-Actor-Role` only while the explicit demo-header switch is enabled (default: `true` for this local project). The fallback demo accounts (`support`, `supervisor`, `admin`) exist solely for local verification and share the configurable `SUPPORTFLOW_DEMO_PASSWORD`; they must not be used in deployment.

Channel integrations use a separate `SUPPORTFLOW_INTEGRATION_KEY`; set it to a unique secret before connecting a real email or enterprise-messaging webhook. The built-in `supportflow-integration-demo` value is local-demo-only.

To connect an order system, configure `SUPPORTFLOW_ORDER_API_BASE_URL`, `SUPPORTFLOW_ORDER_API_TOKEN`, and optionally `SUPPORTFLOW_ORDER_API_TIMEOUT_SECONDS` (default `2`). The external endpoint must return JSON with non-empty `payment_status`, `refund_status`, and `summary` fields. Without both URL and token, SupportFlow stays in the local demo-order adapter. The integration is intentionally read-only; real refunds and account changes belong in separately authorized business workflows.

To connect a CRM, configure `SUPPORTFLOW_CRM_API_BASE_URL`, `SUPPORTFLOW_CRM_API_TOKEN`, and optionally `SUPPORTFLOW_CRM_API_TIMEOUT_SECONDS` (default `2`). The CRM endpoint must return a support-safe JSON object with `tier` (string), `open_ticket_count` (integer), and `summary` (string). The Agent calls only `GET /customers/{customer_id}/support-context`; do not give it write credentials for customer records, order changes, or refunds.

## API flow

1. `POST /tickets` durably queues a new ticket and returns `202 processing`; the four-agent workflow runs in the background. Clients may pass an `Idempotency-Key` header, so a network retry returns the original ticket instead of creating a duplicate.
2. `GET /tickets/{ticket_id}` loads the stored workflow result and audit trail.
3. `POST /tickets/{ticket_id}/approval` records an operator's approve or escalate decision.
4. `POST /tickets/{ticket_id}/feedback` captures reviewer labels (`helpful`, `needs_edit`, `incorrect`, `unsafe`) and an optional note.
5. `POST /tickets/{ticket_id}/resume` reruns a persisted failed ticket with its original input after a temporary dependency problem is corrected.
6. `GET /metrics` reports escalation rate, evidence coverage, human edit rate, and negative-feedback rate from the persisted audit trail.
7. `POST /knowledge/documents` lets an operator publish a Markdown-backed policy document and rebuilds the in-memory index without changing agent code.
8. `POST /channels/{channel}/events` accepts normalized `web`, `email`, or `wecom` events. Set `X-Integration-Key` and provide an `external_event_id`; it becomes the durable idempotency key, so vendor webhook retries do not create duplicate tickets.
9. `GET /knowledge/gap-outcomes` compares feedback, human-edit, and escalation metrics across the pre- and post-publication cohorts for knowledge gaps in verification. It is restricted to supervisors and administrators and returns `insufficient_sample` until both cohorts have enough feedback.
10. `POST /knowledge/documents/{source_id}/restore` restores a historical `vN` as a new version; `GET /knowledge/audit` exposes publish and restore audit events to supervisors and administrators.
11. `GET /healthz` is a dependency-safe container health probe. `GET /operational-alerts` turns persisted metrics into supervisor-visible alerts for failed tickets, low evidence coverage, negative feedback, high latency, and retriever fallback.
12. `GET /annotations/queue` returns the current annotator's unlabelled completed tickets. `POST /tickets/{ticket_id}/annotations` saves an independent human quality label, while `GET /annotations/quality` reports label counts and pairwise agreement for supervisors.

## Observability

Ticket completion and failure events are emitted as secret-safe JSON logs for container log collection. They contain only operational metadata such as ticket ID, category, final status, trace count, and error type; customer content, credentials, and provider keys are never logged. Docker Compose uses `/healthz` as the API health check. Alert thresholds are intentionally visible in `supportflow/observability.py` so operators can review and tune them instead of treating them as hidden model behavior.

## Human evaluation and experiments

Business feedback and evaluation labels serve different purposes. Reviewer feedback can create a knowledge gap; independent annotations do not change the workflow or knowledge base and are used to evaluate a proposed model, prompt, retriever, or knowledge-version experiment. Each annotator receives a personal queue and can update only their own label. For reliable experiment comparison, use at least two annotators on an overlap sample and inspect `pairwise_agreement` before treating label aggregates as a release decision.

The first supported A/B experiment compares `support-reply-v1` against `support-reply-v2`. Enable it deliberately with `SUPPORTFLOW_PROMPT_EXPERIMENT_ID=reply-clarity-v2`. A stable SHA-256 assignment of experiment ID plus ticket ID sends each new ticket to `control` or `treatment`; both the arm and prompt version are persisted with the ticket. V2 asks for one concise clarification that helps a human reviewer verify the case, while retaining the same evidence and safety constraints. Supervisors inspect `GET /experiments/prompt`; the endpoint withholds a comparison verdict until each arm has at least three independent annotations.

## Evaluation gate

`supportflow/evals/ticket_cases.json` is a versioned 30-case regression set for routing, evidence retrieval, no-evidence fallback, and safety escalation. Run `python -m supportflow.run_evals` before changing prompts, retrieval, policies, or models. It uses the deterministic writer, so this quality gate has no model cost and fails when category accuracy, status accuracy, evidence coverage, or safety pass rate falls below the defined threshold. See [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) for the case design and gate thresholds.

GitHub Actions runs `make quality` on pull requests and pushes to the default branch. The workflow intentionally excludes DeepSeek and Bailian calls; external-model experiments remain explicit, opt-in commands.

Run `python -m supportflow.run_retrieval_experiment` for the local TF-IDF baseline. Add `--with-bailian` only when you intend to send the demo knowledge corpus and the 19 retrieval evaluation queries to configured Bailian embeddings; the JSON report distinguishes real semantic retrieval from automatic local fallback.

The first recorded experiment is [`evals/reports/retrieval-comparison-2026-08-08.md`](evals/reports/retrieval-comparison-2026-08-08.md). It documents why the local baseline currently wins on the small, keyword-heavy corpus; semantic retrieval remains configurable rather than being assumed superior.

## Interview framing

This is not a “several prompts talking to each other” demo. The product boundary is risk-controlled ticket resolution: state transitions are explicit, every reply has retrievable evidence, the model has no direct action permission, and human decisions are persisted for audit and future evaluation.

For a demo script, resume bullets, architecture diagram, and interview answers grounded in the current implementation, read [`docs/INTERVIEW_PLAYBOOK.md`](docs/INTERVIEW_PLAYBOOK.md).
