"""Command-line entry point for the deterministic SupportFlow regression suite."""

from __future__ import annotations

from dataclasses import asdict
import json

from supportflow.evaluation import assert_workflow_regression_gate, run_default_workflow_evaluation


if __name__ == "__main__":
    report = run_default_workflow_evaluation()
    assert_workflow_regression_gate(report)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
