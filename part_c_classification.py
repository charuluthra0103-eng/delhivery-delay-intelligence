"""Part C - delay classifier and alert threshold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cm_row(pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "accuracy": (tn + tp) / n,
            "precision": prec, "recall": rec, "f1": f1}


# 2. baseline and three classifiers (0.5 rule)
rows = [{"model": "blind_all_on_time", **cm_row(np.zeros(len(y_val), dtype=int), y_val)}]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **cm_row(lr.predict(X_val), y_val)})
best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1 + 1e-12:
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_d}", **cm_row(tree.predict(X_val), y_val)})
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **cm_row(rf.predict(X_val), y_val)})
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature scores
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

sk_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    p.fit(X_train, y_train)
    sk_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
pd.DataFrame(sk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning on 7 selected features (selector fitted on training rows only)
K = 7
sel = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = sel.transform(X_train), sel.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
dflt = RandomForestClassifier(random_state=42).fit(Xtr7, y_train)
cands = {
    "grid_search": (gs.best_params_, 27, 5, gs.best_score_, f1_score(y_val, gs.best_estimator_.predict(Xva7))),
    "randomized_search": (rs.best_params_, 9, 5, rs.best_score_, f1_score(y_val, rs.best_estimator_.predict(Xva7))),
    "default_forest_same_7_features": ({"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}, 1, 0, np.nan,
                                       f1_score(y_val, dflt.predict(Xva7))),
}
tun = pd.DataFrame([{"search": n, "candidates": c, "cv_folds": cv, "total_fits": 1 if n.startswith("default") else c * cv,
                     "best_params": str({k: (None if v is None else int(v)) for k, v in p.items()}),
                     "best_cv_accuracy": acc, "val_f1": f1} for n, (p, c, cv, acc, f1) in cands.items()])
tun = tun.round(4)
tun.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. choose final configuration (ties -> default, then grid, then randomized)
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen = max(order, key=lambda n: (float(tun.loc[tun.search == n, "val_f1"].iloc[0]), -order.index(n)))
params = cands[chosen][0]
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, n_estimators=int(params["n_estimators"]),
                                                   max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
                                                   min_samples_leaf=int(params["min_samples_leaf"])))])
final.fit(X_train, y_train)

# 6. alert threshold from validation probabilities
pv = final.predict_proba(X_val)[:, 1]
th_rows = [{"threshold": t, **cm_row((pv >= t).astype(int), y_val)} for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
th = pd.DataFrame(th_rows)
th.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
ok = th[th["recall"].round(4) >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
pt = final.predict_proba(X_test)[:, 1]
pd.DataFrame([{"threshold": threshold, **cm_row((pt >= threshold).astype(int), y_test)}]).round(4).to_csv(
    "outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": int(params["n_estimators"]),
               "max_depth": "unlimited" if params["max_depth"] is None else int(params["max_depth"]),
               "min_samples_leaf": int(params["min_samples_leaf"]), "threshold": threshold, "chosen_from": chosen}]
             ).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, params, "threshold", threshold)
