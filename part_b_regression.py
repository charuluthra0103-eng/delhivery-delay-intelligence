"""Part B - delivery-hours regression."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaling without leakage
scaler = StandardScaler().fit(X_train)
tr_s, te_s = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({"metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
              "value": [len(X_train), len(X_val), len(X_test),
                        np.abs(tr_s.mean(axis=0)).max(), np.abs(te_s.mean(axis=0)).max()]}
             ).to_csv("outputs/part_b_checks.csv", index=False, float_format="%.10g")
# (train value is ~1e-16; written at full precision so the < 1e-9 check is visible)


def metrics(model_name, split, y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(y_true, pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}


def rmse(y_true, pred):
    return float(np.sqrt(mean_squared_error(y_true, pred)))


rows = []
# 3. baseline
rows.append(metrics("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean())))

# 4. tree depth diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
dep_rows = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    dep_rows.append({"max_depth": "unlimited" if d is None else d,
                     "train_rmse": rmse(y_train, t.predict(X_train)), "val_rmse": rmse(y_val, t.predict(X_val))})
dep = pd.DataFrame(dep_rows)
dep.round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)
best_i = dep["val_rmse"].round(10).idxmin()          # first (smallest depth) on ties; unlimited is last
best_depth = depths[best_i]

# 5. four candidates
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    r = metrics(name, "val", y_val, m.predict(X_val))
    rows.append(r)
    val_rmse[name] = r["RMSE"]

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, p), "val_r2": r2_score(y_val, p)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validation of the two best, final model, single test evaluation
ranked = sorted(val_rmse, key=val_rmse.get)
top2 = ranked[:2]
X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)       # one shared object
cv_rows = []
for name in top2:
    s = cross_val_score(models[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final = ranked[0]
rows.append(metrics(final, "test", y_test, models[final].predict(X_test)))   # scored once
pd.DataFrame(rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print("validation ranking:", ranked, "| final:", final, "| best depth:", best_depth)
