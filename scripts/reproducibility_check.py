from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import tempfile
import pandas as pd
from src.main import ROOT, run_pipeline

CORE_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]

def main():
    outputs = []
    hashes = []
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        for index in range(1, 4):
            target = temp_path / f"run_{index}.csv"
            frame = run_pipeline(ROOT / "data", target, use_gemini=False)
            outputs.append(frame[CORE_COLUMNS].to_csv(index=False, lineterminator="\n"))
            hashes.append(hashlib.sha256(target.read_bytes()).hexdigest())
    core_equal = all(item == outputs[0] for item in outputs[1:])
    report = {
        "runs": 3,
        "same_input_files": True,
        "gemini_disabled_for_reproducibility_check": True,
        "core_flags_numbers_and_note_ids_identical": core_equal,
        "full_csv_sha256_by_run": hashes,
        "full_csv_byte_identical": len(set(hashes)) == 1,
        "note": "The case brief permits explanation wording to vary; core numerical fields, flags, verdicts, and cited note IDs are compared. Gemini is disabled for this check.",
    }
    report_path = ROOT / "logs" / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not core_equal:
        raise SystemExit("Core reproducibility check failed")

if __name__ == "__main__":
    main()
