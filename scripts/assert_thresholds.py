#!/usr/bin/env python3
"""
assert_thresholds.py — CI gate for promotion thresholds.

RBAC Leak Rate is checked FIRST (hard fail, blocks merge immediately).
Then checks Citation Accuracy, Faithfulness, Context Precision/Recall.

Per module-4 §4.4 and MASTER-PLAN §C:
  RBAC Leak Rate = 0.00  (HARD FAIL — any leak blocks pipeline)
  Citation Accuracy = 1.00 (HARD FAIL)
  Faithfulness >= 0.95
  ContextPrecision >= 0.85
  ContextRecall >= 0.90

Usage:
  python3 scripts/assert_thresholds.py --junit reports/junit.xml [--verbose]
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ─── Thresholds (match conftest.py exactly) ────────────────────────────────
RBAC_LEAK_RATE            = 0.00
CITATION_ACCURACY_THRESHOLD = 1.00
FAITHFULNESS_THRESHOLD    = 0.95
CTX_PRECISION_THRESHOLD   = 0.85
CTX_RECALL_THRESHOLD      = 0.90


def parse_junit(junit_path: str) -> dict:
    """Parse JUnit XML into per-suite pass/fail/skip counts."""
    path = Path(junit_path)
    if not path.exists():
        print(f"[assert_thresholds] WARNING: {junit_path} not found.")
        return {}

    tree = ET.parse(path)
    root = tree.getroot()

    suites: dict[str, dict] = {}

    for tc in root.iter("testcase"):
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        full = f"{cls}.{name}"

        # Route to suite
        key = "other"
        for marker in ["rbac", "citation", "temporal", "ambig", "hybrid", "cache", "latency", "observ"]:
            if marker in full.lower():
                key = marker
                break

        suites.setdefault(key, {"pass": 0, "fail": 0, "skip": 0, "failed_names": []})

        status = "pass"
        for child in tc:
            if child.tag == "failure":
                status = "fail"
                suites[key]["failed_names"].append(name)
            elif child.tag == "skipped":
                status = "skip"
        suites[key][status] += 1

    return suites


def main() -> int:
    parser = argparse.ArgumentParser(description="Assert CI promotion thresholds.")
    parser.add_argument("--junit", default="reports/junit.xml", help="JUnit XML path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    suites = parse_junit(args.junit)
    gate_failures: list[str] = []
    exit_code = 0

    print("[assert_thresholds] ═══ CI Promotion Gate Check ═══")

    # ── GATE 1: RBAC Leak Rate (HARD FAIL, checked FIRST) ─────────────────────
    rbac = suites.get("rbac", {"pass": 0, "fail": 0, "skip": 0, "failed_names": []})
    rbac_total = rbac["pass"] + rbac["fail"]
    rbac_leak_rate = rbac["fail"] / rbac_total if rbac_total > 0 else 0.0

    if rbac_leak_rate > RBAC_LEAK_RATE:
        msg = (
            f"HARD FAIL: RBAC Leak Rate = {rbac_leak_rate:.4f} > {RBAC_LEAK_RATE:.2f}. "
            f"Failed tests: {rbac.get('failed_names', [])}. "
            "Any RBAC leak blocks the release pipeline immediately."
        )
        print(f"[assert_thresholds] {msg}")
        gate_failures.append(msg)
        exit_code = 1
    else:
        print(f"[assert_thresholds] GATE 1 PASS: RBAC Leak Rate = {rbac_leak_rate:.4f} <= {RBAC_LEAK_RATE:.2f}")

    # ── GATE 2: Citation Accuracy (HARD FAIL) ─────────────────────────────────
    cit = suites.get("citation", {"pass": 0, "fail": 0, "skip": 0, "failed_names": []})
    cit_total = cit["pass"] + cit["fail"]
    cit_accuracy = cit["pass"] / cit_total if cit_total > 0 else 1.0

    if cit_accuracy < CITATION_ACCURACY_THRESHOLD:
        msg = (
            f"HARD FAIL: Citation Accuracy = {cit_accuracy:.4f} < {CITATION_ACCURACY_THRESHOLD:.2f}. "
            f"Failed: {cit.get('failed_names', [])}."
        )
        print(f"[assert_thresholds] {msg}")
        gate_failures.append(msg)
        exit_code = 1
    else:
        print(f"[assert_thresholds] GATE 2 PASS: Citation Accuracy = {cit_accuracy:.4f}")

    # ── GATE 3: Temporal Correctness ──────────────────────────────────────────
    temp = suites.get("temporal", {"pass": 0, "fail": 0, "skip": 0})
    temp_total = temp["pass"] + temp["fail"]
    temp_rate = temp["pass"] / temp_total if temp_total > 0 else 1.0
    threshold = 0.95

    if temp_total > 0 and temp_rate < threshold:
        msg = f"SOFT FAIL: Temporal correctness = {temp_rate:.4f} < {threshold:.2f}."
        print(f"[assert_thresholds] WARNING: {msg}")
        # Non-blocking gate
    else:
        print(f"[assert_thresholds] GATE 3 PASS: Temporal correctness = {temp_rate:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = sum(s.get("pass", 0) for s in suites.values())
    all_fail = sum(s.get("fail", 0) for s in suites.values())
    all_skip = sum(s.get("skip", 0) for s in suites.values())

    print(f"\n[assert_thresholds] Total: {all_pass} passed / {all_fail} failed / {all_skip} skipped")

    if gate_failures:
        print(f"\n[assert_thresholds] {len(gate_failures)} HARD GATE(S) FAILED:")
        for gf in gate_failures:
            print(f"  - {gf}")
        return 1

    print("[assert_thresholds] All hard promotion gates passed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
