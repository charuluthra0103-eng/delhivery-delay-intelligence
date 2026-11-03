"""Part A - cleaning, modelling table, checks, delay evidence and correlations."""
import os
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "before" counts, measured after removing duplicates and before any filling
dedup = raw.drop_duplicates()
spell_before = dedup["service_type"].nunique()
missing_dist = int(dedup["distance_km"].isna().sum())
missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
sentinel = int((dedup["parcel_weight_kg"] > 100).sum())
missing_value = int(dedup["declared_value"].isna().sum())
missing_pay = int(dedup["payment_mode"].isna().sum())

clean, medians = clean_shipments(raw)
tbl = build_features(clean, hubs)
tbl["is_delayed"] = (tbl["delivery_hours"] > tbl["promised_hours"]).astype(int)
tbl.to_csv("data/modelling_table.csv", index=False)

q1, q3 = tbl["delivery_hours"].quantile(0.25), tbl["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = int(((tbl["delivery_hours"] < lo) | (tbl["delivery_hours"] > hi)).sum())
nulls = int(tbl[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())

checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": missing_dist,
    "missing_parcel_weight_before": missing_weight,
    "sentinel_weight_rows": sentinel,
    "missing_declared_value_before": missing_value,
    "missing_payment_mode_before": missing_pay,
    "service_type_spellings_before": spell_before,
    "service_type_spellings_after": tbl["service_type"].nunique(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(tbl),
    "nulls_in_features_and_target": nulls,
    "delayed_count": int(tbl["is_delayed"].sum()),
    "delay_rate": tbl["is_delayed"].mean(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": outliers,
    "declared_value_mean_inr": tbl["declared_value_inr"].mean(),
    "declared_value_median_inr": tbl["declared_value_inr"].median(),
    "log_declared_value_mean": tbl["log_declared_value"].mean(),
    "log_declared_value_median": tbl["log_declared_value"].median(),
}
pd.DataFrame({"metric": list(checks), "value": list(checks.values())}).round(4).to_csv(
    "outputs/part_a_checks.csv", index=False)

# delay evidence by group
tbl["distance_band"] = pd.cut(tbl["distance_km"], bins=[0, 200, 400, 600, 2000],
                              labels=["0-200", "201-400", "401-600", "601+"], right=True)
rows = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = tbl.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                            delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    for grp, r in g.iterrows():
        rows.append({"dimension": dim, "group": str(grp), "shipments": int(r["shipments"]),
                     "delayed": int(r["delayed"]), "delay_rate": r["delay_rate"], "mean_hours": r["mean_hours"]})
pd.DataFrame(rows).round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# correlations of numeric columns of the modelling table (distance_band is not numeric)
num = tbl.drop(columns=["distance_band"]).select_dtypes(include="number")
num.corr().round(4).to_csv("outputs/part_a_correlations.csv")
print("Part A done:", len(tbl), "rows")
