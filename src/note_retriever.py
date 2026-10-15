"""Route/date-constrained evidence retrieval backed by FAISS or TF-IDF.

Similarity only ranks notes after metadata and evidence rules are applied; it never
makes a note valid by itself.
"""
from __future__ import annotations
import re
from datetime import timedelta
from pathlib import Path
import pandas as pd
from .vector_store import NoteVectorStore, DEFAULT_MODEL

POSITIVE_PATTERNS = (
    r"higher trip costs", r"higher costs", r"costs across all routes",
    r"pushing up transportation costs", r"surcharge", r"rate increase",
    r"increased transportation costs", r"trip costs", r"diesel prices rose",
    r"toll (?:fee|charge|rate)s? (?:increased|rose|changed)", r"fuel prices? (?:increased|rose|went up)",
    r"limited truck availability", r"forced longer detours",
)
NEGATIVE_PATTERNS = (
    r"costs were not significantly affected", r"not part of this dataset",
    r"no major disruptions", r"no significant disruptions", r"remained normal",
    r"without a rate change", r"costs were unaffected", r"costs were not affected",
    r"returned to normal", r"improved road conditions", r"costs? (?:were )?absorbed .* without a rate change",
)


def route_label(origin: str, destination: str) -> str:
    return f"{origin}-{destination}"


def effective_window(note: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Use explicit event dates when present; otherwise use the note's week only."""
    text = str(note["note"])
    base = pd.Timestamp(note["date"]).normalize()
    match = re.search(r"from\s+([A-Za-z]{3,9})\s+(\d{1,2})\s+to\s+([A-Za-z]{3,9})\s+(\d{1,2})",
                      text, flags=re.IGNORECASE)
    if match:
        sm, sd, em, ed = match.groups()
        start = pd.to_datetime(f"{sm} {sd} {base.year}", errors="coerce")
        end = pd.to_datetime(f"{em} {ed} {base.year}", errors="coerce")
        if pd.notna(start) and pd.notna(end):
            if end < start:
                end = end.replace(year=end.year + 1)
            return start.normalize(), end.normalize()
    start = base.to_period("W-SUN").start_time
    return start, start + timedelta(days=6)


def applies_to_route(note: pd.Series, route: str) -> bool:
    applies = str(note["applies_to"]).strip().casefold()
    return applies == "all routes" or applies == route.casefold()


def is_positive_evidence(note_text: str) -> bool:
    text = note_text.casefold()
    if any(re.search(pattern, text) for pattern in NEGATIVE_PATTERNS):
        return False
    return any(re.search(pattern, text) for pattern in POSITIVE_PATTERNS)


class NoteRetriever:
    def __init__(self, notes: pd.DataFrame, index_dir: str | Path | None = None,
                 model_name: str = DEFAULT_MODEL):
        self.notes = notes.reset_index(drop=True).copy()
        self.vector_store = NoteVectorStore(self.notes, index_dir=index_dir, model_name=model_name)

    @property
    def backend(self) -> str:
        return self.vector_store.backend

    @property
    def warning(self) -> str | None:
        return self.vector_store.warning

    def candidates_for(self, route: str, week_of: pd.Timestamp) -> list[int]:
        week_start = pd.Timestamp(week_of).normalize()
        week_end = week_start + timedelta(days=6)
        valid = []
        for idx, note in self.notes.iterrows():
            if not applies_to_route(note, route):
                continue
            start, end = effective_window(note)
            if start > week_end or end < week_start:
                continue
            if not is_positive_evidence(str(note["note"])):
                continue
            valid.append(int(idx))
        return valid

    def retrieve(self, route: str, week_of: pd.Timestamp) -> dict | None:
        week_start = pd.Timestamp(week_of).normalize()
        valid_indices = self.candidates_for(route, week_start)
        if not valid_indices:
            return None
        query = (f"Route {route}; week starting {week_start.date()}; unusual freight cost increase; "
                 "transportation cost drivers such as fuel, toll, surcharge, disruption, detour, truck availability")
        scores = self.vector_store.scores(query)
        best_idx = max(valid_indices, key=lambda idx: (scores.get(idx, 0.0), -idx))
        best = self.notes.iloc[best_idx]
        return {"note_id": str(best["note_id"]), "note": str(best["note"]),
                "date": pd.Timestamp(best["date"]), "similarity": float(scores.get(best_idx, 0.0))}

    def closest_context_note(self, route: str, week_of: pd.Timestamp) -> dict | None:
        """Return closest route/global note for explanatory context, never as justification."""
        week_start = pd.Timestamp(week_of).normalize()
        pool = []
        for _, note in self.notes.iterrows():
            if not applies_to_route(note, route):
                continue
            note_date = pd.Timestamp(note["date"]).normalize()
            distance = abs((note_date - week_start).days)
            pool.append((distance, str(note["note_id"]), note))
        if not pool:
            return None
        _, _, note = min(pool, key=lambda item: (item[0], item[1]))
        return {"note_id": str(note["note_id"]), "note": str(note["note"]),
                "date": pd.Timestamp(note["date"])}


def retrieve_supporting_note(notes: pd.DataFrame, route: str, week_of: pd.Timestamp,
                             retriever: NoteRetriever | None = None) -> dict | None:
    """Compatibility helper used by tests and scripts."""
    return (retriever or NoteRetriever(notes)).retrieve(route, week_of)
