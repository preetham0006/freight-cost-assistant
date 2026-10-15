import pandas as pd
from src.note_retriever import NoteRetriever, retrieve_supporting_note


def _notes(rows):
    return pd.DataFrame(rows, columns=["note_id", "date", "applies_to", "note"])


def test_correct_route_and_date_note_is_returned(tmp_path):
    notes = _notes([
        ("N1", "2025-01-20", "A-B", "A temporary surcharge was applied due to high demand and limited truck availability."),
        ("N2", "2025-01-20", "C-D", "A temporary surcharge was applied due to high demand and limited truck availability."),
    ])
    notes["date"] = pd.to_datetime(notes["date"])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    found = retriever.retrieve("A-B", pd.Timestamp("2025-01-20"))
    assert found and found["note_id"] == "N1"
    assert retriever.retrieve("A-B", pd.Timestamp("2025-02-03")) is None


def test_all_routes_note_can_support_route_during_valid_week(tmp_path):
    notes = _notes([("N3", "2025-05-05", "All Routes", "Diesel prices rose nationwide starting this week, pushing up transportation costs across all routes by roughly 5-7%.")])
    notes["date"] = pd.to_datetime(notes["date"])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    assert retriever.retrieve("A-B", pd.Timestamp("2025-05-05"))["note_id"] == "N3"


def test_negative_or_unrelated_note_never_justifies_increase(tmp_path):
    notes = _notes([
        ("N6", "2025-09-22", "All Routes", "Demand remained stable with no major disruptions reported."),
        ("N7", "2024-05-20", "A-B", "Improved road conditions after resurfacing work was completed."),
    ])
    notes["date"] = pd.to_datetime(notes["date"])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    assert retriever.retrieve("A-B", pd.Timestamp("2025-09-15")) is None
    assert retriever.retrieve("A-B", pd.Timestamp("2024-05-20")) is None


def test_compatibility_function(tmp_path):
    notes = _notes([("N1", "2025-01-20", "A-B", "A temporary surcharge was applied due to high demand and limited truck availability.")])
    notes["date"] = pd.to_datetime(notes["date"])
    retriever = NoteRetriever(notes, index_dir=tmp_path)
    assert retrieve_supporting_note(notes, "A-B", pd.Timestamp("2025-01-20"), retriever)["note_id"] == "N1"
