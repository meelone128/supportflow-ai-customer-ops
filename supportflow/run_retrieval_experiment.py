"""Compare local and optional Bailian retrieval on the versioned evaluation set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from supportflow.evaluation import run_retrieval_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-bailian", action="store_true", help="send the demo corpus and evaluation queries to configured Bailian embeddings")
    arguments = parser.parse_args()
    print(json.dumps([asdict(report) for report in run_retrieval_comparison(arguments.with_bailian)], ensure_ascii=False, indent=2))
