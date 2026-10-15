import pandas as pd
import pytest
from src.cost_analyzer import aggregate_weekly_costs, add_comparison_baselines


def _weekly(rows):
    return pd.DataFrame(rows, columns=["origin", "destination", "route_type", "week_of", "cost_per_tonne_km"])


def test_weighted_weekly_cost_and_monday_week():
    shipments = pd.DataFrame({
        "origin": ["A", "A"], "destination": ["B", "B"], "route_type": ["Short", "Short"],
        "shipment_date": pd.to_datetime(["2025-01-06", "2025-01-12"]),
        "quantity_tonnes": [10, 20], "distance_km": [100, 100], "freight_cost_inr": [1000, 4000],
    })
    result = aggregate_weekly_costs(shipments)
    assert len(result) == 1
    assert result.loc[0, "week_of"] == pd.Timestamp("2025-01-06")
    assert result.loc[0, "cost_per_tonne_km"] == pytest.approx(5000 / 3000)


def test_historical_baseline_excludes_current_week_and_uses_prior_observations():
    weekly = _weekly([
        ("A", "B", "Short", "2025-01-06", 2.0),
        ("A", "B", "Short", "2025-01-13", 4.0),
        ("A", "B", "Short", "2025-01-20", 100.0),
    ])
    weekly["week_of"] = pd.to_datetime(weekly["week_of"])
    result = add_comparison_baselines(weekly)
    assert pd.isna(result.loc[0, "own_history_avg"])
    assert result.loc[1, "own_history_avg"] == pytest.approx(2.0)
    assert result.loc[2, "own_history_avg"] == pytest.approx(3.0)


def test_historical_baseline_uses_at_most_eight_previous_observations():
    rows = [("A", "B", "Short", pd.Timestamp("2025-01-06") + pd.Timedelta(days=7*i), float(i+1)) for i in range(10)]
    weekly = _weekly(rows)
    weekly["week_of"] = pd.to_datetime(weekly["week_of"])
    result = add_comparison_baselines(weekly)
    # For observation 10, previous eight costs are 2..9, average 5.5; current 10 excluded.
    assert result.iloc[9]["own_history_avg"] == pytest.approx(5.5)


def test_peer_average_excludes_current_route_and_uses_same_type_and_week():
    weekly = _weekly([
        ("A", "B", "Short", "2025-01-06", 10.0),
        ("C", "D", "Short", "2025-01-06", 4.0),
        ("E", "F", "Short", "2025-01-06", 6.0),
        ("G", "H", "Long", "2025-01-06", 100.0),
    ])
    weekly["week_of"] = pd.to_datetime(weekly["week_of"])
    result = add_comparison_baselines(weekly)
    assert result.loc[0, "peer_avg"] == pytest.approx(5.0)
    assert result.loc[1, "peer_avg"] == pytest.approx(8.0)


def test_peer_baseline_missing_when_no_comparable_route():
    weekly = _weekly([("A", "B", "Short", "2025-01-06", 10.0)])
    weekly["week_of"] = pd.to_datetime(weekly["week_of"])
    result = add_comparison_baselines(weekly)
    assert pd.isna(result.loc[0, "peer_avg"])
    assert pd.isna(result.loc[0, "vs_similar_routes_pct"])
