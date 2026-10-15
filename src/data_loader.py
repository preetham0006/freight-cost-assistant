from pathlib import Path
import pandas as pd

SHIPMENT_REQUIRED = {
    "shipment_id", "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date"
}
NOTE_REQUIRED = {"note_id", "date", "applies_to", "note"}


def load_inputs(shipment_path: str | Path, notes_path: str | Path):
    """Load inputs and fail early when required fields or values are invalid."""
    shipments = pd.read_csv(shipment_path)
    notes = pd.read_csv(notes_path)

    missing_shipments = SHIPMENT_REQUIRED - set(shipments.columns)
    missing_notes = NOTE_REQUIRED - set(notes.columns)
    if missing_shipments:
        raise ValueError(f"Shipment file is missing columns: {sorted(missing_shipments)}")
    if missing_notes:
        raise ValueError(f"Notes file is missing columns: {sorted(missing_notes)}")

    shipments["shipment_date"] = pd.to_datetime(shipments["shipment_date"], errors="raise")
    notes["date"] = pd.to_datetime(notes["date"], errors="raise")
    for column in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        shipments[column] = pd.to_numeric(shipments[column], errors="raise")

    if shipments[["origin", "destination", "route_type", "shipment_date"]].isna().any().any():
        raise ValueError("Shipment route/date fields cannot be empty.")
    if (shipments["quantity_tonnes"] <= 0).any():
        raise ValueError("quantity_tonnes must be positive.")
    if (shipments["distance_km"] <= 0).any():
        raise ValueError("distance_km must be positive.")
    if (shipments["freight_cost_inr"] < 0).any():
        raise ValueError("freight_cost_inr cannot be negative.")
    if notes[["note_id", "date", "applies_to", "note"]].isna().any().any():
        raise ValueError("Context notes must have note_id, date, applies_to, and note values.")
    return shipments, notes
