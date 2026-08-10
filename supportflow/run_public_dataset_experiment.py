"""CLI: compare public corpus knowledge V1 against V2."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from supportflow.public_dataset_experiment import compare_public_knowledge_versions


if __name__ == "__main__":
    report = compare_public_knowledge_versions(Path("supportflow/evals/public_corpus"))
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
