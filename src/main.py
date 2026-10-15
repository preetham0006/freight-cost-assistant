from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from .data_loader import load_inputs
from .cost_analyzer import aggregate_weekly_costs, add_comparison_baselines
from .anomaly_detector import detect_anomalies, ANOMALY_THRESHOLD_PCT
from .note_retriever import NoteRetriever
from .explanation_engine import GeminiExplainer

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass


def run_pipeline(data_dir: Path = ROOT / "data",
                 output_path: Path = ROOT / "outputs" / "shipping_cost_report.csv",
                 use_gemini: bool | None = None) -> pd.DataFrame:
    _load_dotenv()
    if use_gemini is not None:
        os.environ["ENABLE_GEMINI"] = "true" if use_gemini else "false"
    shipments, notes = load_inputs(data_dir / "shipment_records.csv", data_dir / "context_notes.csv")
    weekly = aggregate_weekly_costs(shipments)
    comparisons = add_comparison_baselines(weekly)
    retriever = NoteRetriever(notes, index_dir=ROOT / "vector_store",
                              model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    alerts = detect_anomalies(comparisons, notes, retriever=retriever)

    explainer = GeminiExplainer()
    if explainer.enabled:
        # The LLM only rewrites reasons for deterministic, already-validated positive evidence.
        # It cannot change the flag, note ID, numerical comparisons, or whether evidence is valid.
        for idx, row in alerts.iterrows():
            if row["flagged"] != "No (justified)":
                continue
            matched = notes[notes["note_id"].astype(str) == str(row["matched_note_id"])].iloc[0]
            note = {"note_id": str(matched["note_id"]), "note": str(matched["note"]),
                    "date": pd.Timestamp(matched["date"])}
            alerts.at[idx, "reason"] = explainer.explain_validated_note(
                str(row["route"]), str(row["week_of"]), note
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(output_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    llm_usage = explainer.usage.as_dict()
    run_log = {
        "shipment_records": int(len(shipments)),
        "context_notes": int(len(notes)),
        "route_week_observations": int(len(weekly)),
        "alerts_written": int(len(alerts)),
        "anomaly_threshold_pct": ANOMALY_THRESHOLD_PCT,
        "retrieval_backend": retriever.backend,
        "retrieval_warning": retriever.warning,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "llm": llm_usage,
        "output_sha256": digest,
        "core_verdicts_deterministic": True,
        "note": "LLM, when enabled, may change reason wording only; it cannot change numbers, flags, or matched note IDs.",
    }
    log_path = ROOT / "logs" / "run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
    print(f"Loaded {len(shipments):,} shipments and {len(notes)} notes.")
    print(f"Computed {len(weekly):,} route-week observations.")
    print(f"Retrieval backend: {retriever.backend}")
    if retriever.warning:
        print(f"Retrieval note: {retriever.warning}")
    print(f"Wrote {len(alerts):,} alert rows to: {output_path}")
    print(f"Output SHA-256: {digest}")
    print(f"Gemini calls: {llm_usage['calls']}; input tokens: {llm_usage['input_tokens']}; output tokens: {llm_usage['output_tokens']}; estimated USD cost: {llm_usage['estimated_cost_usd']}")
    return alerts


def main():
    parser = argparse.ArgumentParser(description="FreightTiger shipping cost anomaly assistant")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "shipping_cost_report.csv")
    parser.add_argument("--use-gemini", action="store_true", help="Enable Gemini explanations (requires GEMINI_API_KEY).")
    parser.add_argument("--no-gemini", action="store_true", help="Force deterministic no-LLM mode.")
    args = parser.parse_args()
    use_gemini = True if args.use_gemini else (False if args.no_gemini else None)
    run_pipeline(args.data_dir, args.output, use_gemini=use_gemini)


if __name__ == "__main__":
    main()
