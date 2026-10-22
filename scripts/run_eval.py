"""CLI runner for the evaluation harness.

Works fully offline (no API key) via ``--offline`` using the deterministic
heuristic stub, which is how it runs in CI on every pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.evaluate import (
    EvalReport,
    regression,
    run_evaluation,
)
from assistant.testing import HeuristicLLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AI Email Assistant evaluation harness")
    parser.add_argument("--dataset", default="data/eval_dataset.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--fail-under", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="force the heuristic stub (no API key)")
    parser.add_argument("--allow-regression", action="store_true")
    args = parser.parse_args(argv)

    llm = None
    if args.offline:
        llm = HeuristicLLM()

    report = run_evaluation(
        dataset_path=args.dataset,
        llm=llm,
        concurrency=args.concurrency,
    )

    print(report.to_markdown())
    print("")
    print("JSON:")
    print(json.dumps(report.to_dict(), indent=2))

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(report.to_markdown(), encoding="utf-8")

    if args.baseline:
        baseline = EvalReport(**json.loads(Path(args.baseline).read_text(encoding="utf-8")))
        regs = regression(baseline, report)
        if regs and not args.allow_regression:
            print(f"REGRESSION DETECTED: {regs}")
            return 1

    if args.fail_under:
        metric, _, threshold = args.fail_under.partition(":")
        threshold = float(threshold)
        value = report.metrics.get(metric)
        if value is None:
            print(f"unknown metric {metric}")
            return 1
        if value < threshold:
            print(f"FAIL-UNDER: {metric} = {value:.4f} < {threshold}")
            return 1

    print("Evaluation complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
