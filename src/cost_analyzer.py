import pandas as pd

ROUTE_KEYS = ["origin", "destination", "route_type"]


def aggregate_weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Monday-Sunday route weeks using weighted cost / tonne-km."""
    data = shipments.copy()
    data["week_of"] = data["shipment_date"].dt.to_period("W-SUN").dt.start_time
    data["tonne_km"] = data["quantity_tonnes"] * data["distance_km"]
    grouped = data.groupby(ROUTE_KEYS + ["week_of"], as_index=False, sort=True).agg(
        total_freight_cost=("freight_cost_inr", "sum"),
        total_tonne_km=("tonne_km", "sum"),
    )
    grouped["cost_per_tonne_km"] = grouped["total_freight_cost"] / grouped["total_tonne_km"]
    return grouped.sort_values(ROUTE_KEYS + ["week_of"]).reset_index(drop=True)


def add_comparison_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 8-observation history and same-week same-type peer baselines."""
    result = weekly.copy().sort_values(ROUTE_KEYS + ["week_of"]).reset_index(drop=True)
    result["own_history_avg"] = (
        result.groupby(ROUTE_KEYS, sort=False)["cost_per_tonne_km"]
        .transform(lambda s: s.shift(1).rolling(window=8, min_periods=1).mean())
    )

    # A peer baseline is the mean of every other route of the same route_type in
    # the same week. First aggregate each route-week, then remove the current route.
    type_week_sum = result.groupby(["route_type", "week_of"])["cost_per_tonne_km"].transform("sum")
    type_week_count = result.groupby(["route_type", "week_of"])["cost_per_tonne_km"].transform("count")
    result["peer_avg"] = (type_week_sum - result["cost_per_tonne_km"]) / (type_week_count - 1)
    result.loc[type_week_count <= 1, "peer_avg"] = float("nan")

    result["vs_own_history_pct"] = _pct_change(result["cost_per_tonne_km"], result["own_history_avg"])
    result["vs_similar_routes_pct"] = _pct_change(result["cost_per_tonne_km"], result["peer_avg"])
    return result


def _pct_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
    baseline = baseline.astype(float)
    values = (current - baseline) / baseline * 100.0
    values = values.where(baseline.notna() & baseline.ne(0))
    return values
