"""Offline, repeatable V1/V2 retrieval comparison for the public corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from supportflow.domain import TicketCategory
from supportflow.knowledge import KnowledgeChunk, LocalVectorIndex
from supportflow.public_dataset import PublicSupportRecord, TARGET_INTENT


@dataclass(frozen=True)
class PublicKnowledgeReport:
    corpus: str
    blind_test_cases: int
    v1_target_intent_at_1: float
    v2_target_intent_at_1: float
    delta: float
    metric_definition: str


def _load_jsonl(path: Path) -> list[PublicSupportRecord]:
    return [PublicSupportRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(records: list[PublicSupportRecord]) -> tuple[LocalVectorIndex, dict[str, str]]:
    chunks = [
        KnowledgeChunk(
            source_id=record.record_id,
            title=f"Public support example: {record.intent}",
            category=TicketCategory.REFUND_POLICY,
            content=f"Intent: {record.intent}\nCustomer: {record.instruction}\nReference: {record.response}",
            version="public-corpus",
        )
        for record in records
    ]
    return LocalVectorIndex(chunks), {record.record_id: record.intent for record in records}


def _target_intent_at_1(index: LocalVectorIndex, intents: dict[str, str], cases: list[PublicSupportRecord]) -> float:
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        results = index.search(case.instruction, TicketCategory.REFUND_POLICY, limit=1)
        if results and intents[results[0][0].source_id] == TARGET_INTENT:
            hits += 1
    return round(hits / len(cases), 3)


def compare_public_knowledge_versions(corpus_directory: Path) -> PublicKnowledgeReport:
    v1 = _load_jsonl(corpus_directory / "knowledge_v1.jsonl")
    additions = _load_jsonl(corpus_directory / "knowledge_v2_additions.jsonl")
    blind_test = _load_jsonl(corpus_directory / "blind_test_track_refund.jsonl")
    v1_index, v1_intents = _index(v1)
    v2_index, v2_intents = _index(v1 + additions)
    v1_score = _target_intent_at_1(v1_index, v1_intents, blind_test)
    v2_score = _target_intent_at_1(v2_index, v2_intents, blind_test)
    report = PublicKnowledgeReport(
        corpus=str(corpus_directory),
        blind_test_cases=len(blind_test),
        v1_target_intent_at_1=v1_score,
        v2_target_intent_at_1=v2_score,
        delta=round(v2_score - v1_score, 3),
        metric_definition="For blind track_refund queries, whether top-1 retrieved public knowledge has intent=track_refund.",
    )
    (corpus_directory / "retrieval_comparison.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
