"""Build a traceable public-data experiment corpus without mixing it into demo tickets.

The resulting files are intentionally generated locally rather than committed with
the repository: public datasets can change and their source license must be
with each experiment run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Iterable, Mapping


SOURCE_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SOURCE_LICENSE = "cdla-sharing-1.0"
TARGET_INTENT = "track_refund"


@dataclass(frozen=True)
class PublicSupportRecord:
    record_id: str
    instruction: str
    response: str
    category: str
    intent: str
    source_dataset: str = SOURCE_DATASET
    source_license: str = SOURCE_LICENSE


@dataclass(frozen=True)
class PublicCorpusManifest:
    source_dataset: str
    source_license: str
    seed: str
    withheld_intent: str
    knowledge_v1_records: int
    knowledge_v2_additions: int
    blind_test_records: int
    leakage_check: str


_AFTER_SALES_TERMS = (
    "refund", "return", "order", "delivery", "shipping", "payment", "cancel",
    "password", "account", "invoice", "track",
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _record_id(instruction: str, response: str, category: str, intent: str) -> str:
    material = "\n".join((instruction, response, category, intent)).encode("utf-8")
    return "BITEXT-" + sha256(material).hexdigest()[:16].upper()


def records_from_rows(rows: Iterable[Mapping[str, object]]) -> list[PublicSupportRecord]:
    """Normalize only rows with the fields required for a reproducible experiment."""
    records: list[PublicSupportRecord] = []
    seen: set[str] = set()
    for row in rows:
        instruction = _clean(row.get("instruction"))
        response = _clean(row.get("response"))
        category = _clean(row.get("category")).lower()
        intent = _clean(row.get("intent")).lower()
        searchable = " ".join((instruction, response, category, intent)).lower()
        if not instruction or not response or not any(term in searchable for term in _AFTER_SALES_TERMS):
            continue
        record_id = _record_id(instruction, response, category, intent)
        if record_id not in seen:
            records.append(PublicSupportRecord(record_id, instruction, response, category, intent))
            seen.add(record_id)
    return records


def _stable_order(records: Iterable[PublicSupportRecord], seed: str) -> list[PublicSupportRecord]:
    return sorted(records, key=lambda item: sha256(f"{seed}:{item.record_id}".encode()).hexdigest())


def build_public_corpus(
    records: Iterable[PublicSupportRecord],
    output_directory: Path,
    *,
    seed: str = "supportflow-public-corpus-v1",
    test_size: int = 30,
    addition_size: int = 40,
) -> PublicCorpusManifest:
    """Create a controlled V1/V2 knowledge comparison with a held-out intent."""
    output_directory.mkdir(parents=True, exist_ok=True)
    normalized = list(records)
    target = [item for item in normalized if item.intent == TARGET_INTENT]
    if len(target) < test_size + 1:
        raise ValueError(f"Need at least {test_size + 1} {TARGET_INTENT!r} records; found {len(target)}")

    ordered_target = _stable_order(target, seed)
    blind_test = ordered_target[:test_size]
    additions = ordered_target[test_size : test_size + addition_size]
    if not additions:
        raise ValueError("No non-test records remain for the V2 knowledge addition")
    test_ids = {item.record_id for item in blind_test}
    knowledge_v1 = [item for item in normalized if item.intent != TARGET_INTENT]
    if test_ids & {item.record_id for item in knowledge_v1 + additions}:
        raise RuntimeError("Public corpus leakage check failed")

    def write_jsonl(name: str, values: list[PublicSupportRecord]) -> None:
        (output_directory / name).write_text(
            "\n".join(json.dumps(asdict(value), ensure_ascii=False) for value in values) + ("\n" if values else ""),
            encoding="utf-8",
        )

    write_jsonl("knowledge_v1.jsonl", knowledge_v1)
    write_jsonl("knowledge_v2_additions.jsonl", additions)
    write_jsonl("blind_test_track_refund.jsonl", blind_test)
    manifest = PublicCorpusManifest(SOURCE_DATASET, SOURCE_LICENSE, seed, TARGET_INTENT, len(knowledge_v1), len(additions), len(blind_test), "passed")
    (output_directory / "manifest.json").write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bitext_rows() -> Iterable[Mapping[str, object]]:
    """Load the public source lazily so product runtime never needs ``datasets``."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Missing optional dependency. Run: python -m pip install -r supportflow/requirements-data.txt") from error
    dataset = load_dataset(SOURCE_DATASET)
    return dataset["train"] if "train" in dataset else next(iter(dataset.values()))


def safe_filename(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")
