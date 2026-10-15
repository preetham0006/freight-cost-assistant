from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_inputs
from src.note_retriever import NoteRetriever

ROOT = Path(__file__).resolve().parents[1]

def main():
    _, notes = load_inputs(ROOT / "data" / "shipment_records.csv", ROOT / "data" / "context_notes.csv")
    retriever = NoteRetriever(notes, index_dir=ROOT / "vector_store")
    cases = json.loads((ROOT / "tests" / "evaluation_cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        note = retriever.retrieve(case["route"], pd.Timestamp(case["week_of"]))
        actual = note["note_id"] if note else None
        passed = actual == case["expected_note_id"]
        results.append({**case, "actual_note_id": actual, "passed": passed})
    report = {
        "case_count": len(results),
        "passed": sum(bool(r["passed"]) for r in results),
        "failed": sum(not bool(r["passed"]) for r in results),
        "accuracy": round(sum(bool(r["passed"]) for r in results) / len(results), 4) if results else 1.0,
        "retrieval_backend": retriever.backend,
        "warning": retriever.warning,
        "results": results,
        "interpretation": "Small hand-labelled smoke/evidence set; not a statistically representative benchmark.",
    }
    path = ROOT / "logs" / "evaluation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["failed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
