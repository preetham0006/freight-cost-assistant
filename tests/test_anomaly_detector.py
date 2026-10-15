import pandas as pd
from src.anomaly_detector import detect_anomalies
from src.note_retriever import NoteRetriever


def test_anomaly_rule_flags_either_baseline_and_preserves_output_contract(tmp_path):
    comparisons = pd.DataFrame([
        {"origin": "A", "destination": "B", "route_type": "Short", "week_of": pd.Timestamp("2025-01-20"),
         "cost_per_tonne_km": 10.0, "vs_own_history_pct": 5.0, "vs_similar_routes_pct": 25.0},
        {"origin": "C", "destination": "D", "route_type": "Short", "week_of": pd.Timestamp("2025-01-20"),
         "cost_per_tonne_km": 5.0, "vs_own_history_pct": 10.0, "vs_similar_routes_pct": 10.0},
    ])
    notes = pd.DataFrame(columns=["note_id", "date", "applies_to", "note"])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    result = detect_anomalies(comparisons, notes, threshold_pct=20, retriever=retriever)
    assert len(result) == 1
    assert result.iloc[0]["route"] == "A-B"
    assert result.iloc[0]["flagged"] == "Yes"
    assert result.iloc[0]["matched_note_id"] == ""
    assert result.columns.tolist() == ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                                       "vs_similar_routes", "flagged", "matched_note_id", "reason"]


def test_valid_note_justifies_only_when_route_and_week_match(tmp_path):
    comparisons = pd.DataFrame([
        {"origin": "A", "destination": "B", "route_type": "Short", "week_of": pd.Timestamp("2025-01-20"),
         "cost_per_tonne_km": 10.0, "vs_own_history_pct": 25.0, "vs_similar_routes_pct": 10.0},
    ])
    notes = pd.DataFrame([{"note_id": "N1", "date": pd.Timestamp("2025-01-20"), "applies_to": "A-B",
                           "note": "A temporary surcharge was applied due to high demand and limited truck availability."}])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    result = detect_anomalies(comparisons, notes, threshold_pct=20, retriever=retriever)
    assert result.iloc[0]["flagged"] == "No (justified)"
    assert result.iloc[0]["matched_note_id"] == "N1"
    assert "N1" in result.iloc[0]["reason"]
