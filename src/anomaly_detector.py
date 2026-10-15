"""Deterministic alert rules and evidence-grounded verdicts."""
from __future__ import annotations
import pandas as pd
from .note_retriever import NoteRetriever, route_label

# The brief does not define a numeric threshold. This is a documented, configurable assumption.
ANOMALY_THRESHOLD_PCT = 20.0
OUTPUT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history",
                  "vs_similar_routes", "flagged", "matched_note_id", "reason"]


def _comparison_text(value, label: str) -> str:
    if pd.isna(value):
        return f"N/A (no available {label} baseline)"
    return f"{value:+.1f}% vs {label}"


def _reason_for_justified(note: dict) -> str:
    return (f"Matches note {note['note_id']} dated {pd.Timestamp(note['date']).date()}: "
            f"{note['note'].strip()} The cost rise has a clear explanation.")


def _reason_for_unexplained(retriever: NoteRetriever, route: str, week_of: pd.Timestamp) -> str:
    # A nearby non-supporting note can be mentioned as context, but is never assigned
    # to matched_note_id and never used to justify the cost increase.
    closest = retriever.closest_context_note(route, week_of)
    if closest:
        text = closest["note"].strip()
        return (f"The closest contextual note ({closest['note_id']}, {closest['date'].date()}) says: {text} "
                "It does not provide valid route- and time-specific evidence for this cost rise. "
                "No genuine justification was found; flagged for human review.")
    return ("No matching note found for this route or date range. Cost rise looks unexplained "
            "and is worth a human review.")


def detect_anomalies(comparisons: pd.DataFrame, notes: pd.DataFrame,
                     threshold_pct: float = ANOMALY_THRESHOLD_PCT,
                     retriever: NoteRetriever | None = None) -> pd.DataFrame:
    """Emit route-weeks whose historical OR peer increase reaches threshold.

    The case study leaves the threshold unspecified; 20% is an explicit configurable
    engineering assumption. Deterministic code owns flags and matched_note_id.
    """
    retriever = retriever or NoteRetriever(notes)
    records = []
    for _, row in comparisons.iterrows():
        own = row["vs_own_history_pct"]
        peer = row["vs_similar_routes_pct"]
        own_rising = pd.notna(own) and own >= threshold_pct
        peer_rising = pd.notna(peer) and peer >= threshold_pct
        if not (own_rising or peer_rising):
            continue

        route = route_label(row["origin"], row["destination"])
        week = pd.Timestamp(row["week_of"]).normalize()
        note = retriever.retrieve(route, week)
        if note:
            flag = "No (justified)"
            matched_note_id = note["note_id"]
            reason = _reason_for_justified(note)
        else:
            flag = "Yes"
            matched_note_id = ""
            reason = _reason_for_unexplained(retriever, route, week)

        records.append({
            "route": route,
            "week_of": week.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
            "vs_own_history": _comparison_text(own, "this route's past average"),
            "vs_similar_routes": _comparison_text(peer, "similar-length routes this week"),
            "flagged": flag,
            "matched_note_id": matched_note_id,
            "reason": reason,
        })
    output = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if not output.empty:
        output = output.sort_values(["week_of", "route"]).reset_index(drop=True)
    return output
