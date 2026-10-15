from pathlib import Path
import pandas as pd
from src.cost_analyzer import aggregate_weekly_costs, add_comparison_baselines
from src.note_retriever import retrieve_supporting_note
from src.main import ROOT, run_pipeline
from src.data_loader import load_inputs


def test_weighted_weekly_cost_and_monday_week():
    shipments = pd.DataFrame({
        "origin": ["A", "A"], "destination": ["B", "B"], "route_type": ["Short", "Short"],
        "shipment_date": pd.to_datetime(["2025-01-06", "2025-01-12"]),
        "quantity_tonnes": [10, 20], "distance_km": [100, 100], "freight_cost_inr": [1000, 4000],
    })
    result = aggregate_weekly_costs(shipments)
    assert len(result) == 1
    assert result.loc[0, "week_of"] == pd.Timestamp("2025-01-06")
    assert result.loc[0, "cost_per_tonne_km"] == 5000 / 3000


def test_historical_baseline_excludes_current_week():
    weekly = pd.DataFrame({
        "origin": ["A"] * 3, "destination": ["B"] * 3, "route_type": ["Short"] * 3,
        "week_of": pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-20"]),
        "cost_per_tonne_km": [2.0, 4.0, 100.0],
    })
    result = add_comparison_baselines(weekly)
    assert pd.isna(result.loc[0, "own_history_avg"])
    assert result.loc[1, "own_history_avg"] == 2.0
    assert result.loc[2, "own_history_avg"] == 3.0


def test_note_must_match_route_and_week():
    notes = pd.DataFrame([
        {"note_id": "N1", "date": pd.Timestamp("2025-01-20"), "applies_to": "A-B",
         "note": "A temporary surcharge was applied due to high demand and limited truck availability."},
        {"note_id": "N2", "date": pd.Timestamp("2025-01-20"), "applies_to": "C-D",
         "note": "A temporary surcharge was applied due to high demand and limited truck availability."},
    ])
    found = retrieve_supporting_note(notes, "A-B", pd.Timestamp("2025-01-20"))
    assert found and found["note_id"] == "N1"
    assert retrieve_supporting_note(notes, "A-B", pd.Timestamp("2025-02-03")) is None


def test_real_input_pipeline_writes_exact_columns(tmp_path):
    data_dir = ROOT / "data"
    output = tmp_path / "out.csv"
    result = run_pipeline(data_dir, output, use_gemini=False)
    expected = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                "vs_similar_routes", "flagged", "matched_note_id", "reason"]
    assert result.columns.tolist() == expected
    assert output.exists()
    assert set(result["flagged"].unique()).issubset({"Yes", "No (justified)"})
    justified = result[result["flagged"] == "No (justified)"]
    assert justified["matched_note_id"].ne("").all()
    assert (result.loc[result["flagged"] == "Yes", "matched_note_id"] == "").all()
