"""CLI: download public Bitext data and create SupportFlow's isolated experiment corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from supportflow.public_dataset import build_public_corpus, load_bitext_rows, records_from_rows
from supportflow.public_dataset_experiment import compare_public_knowledge_versions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a leakage-checked public customer-support corpus")
    parser.add_argument("--output", type=Path, default=Path("supportflow/evals/public_corpus"))
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--addition-size", type=int, default=40)
    arguments = parser.parse_args()
    manifest = build_public_corpus(
        records_from_rows(load_bitext_rows()), arguments.output,
        test_size=arguments.test_size, addition_size=arguments.addition_size,
    )
    report = compare_public_knowledge_versions(arguments.output)
    print(json.dumps({"manifest": asdict(manifest), "comparison": asdict(report)}, ensure_ascii=False, indent=2))
