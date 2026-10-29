"""AI-Powered Network Intrusion Detection System (simulated flow data)."""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

RNG = np.random.default_rng(42)
N_PER_CLASS = 2400  # 4 classes x 2400 = 9,600 flows
FEATURES = ["duration", "src_bytes", "dst_bytes", "packets", "syn_count",
            "ack_count", "unique_dst_ports", "failed_logins", "conn_rate"]


def make_class(label, n, _depth=0):
    r = RNG
    if label == "normal":
        d = dict(duration=r.gamma(2, 3, n), src_bytes=r.lognormal(6.5, 1.0, n),
                 dst_bytes=r.lognormal(7.5, 1.2, n), packets=r.poisson(25, n),
                 syn_count=r.poisson(1.2, n), ack_count=r.poisson(22, n),
                 unique_dst_ports=r.poisson(2, n) + 1, failed_logins=r.poisson(0.1, n),
                 conn_rate=r.gamma(2, 2, n))
    elif label == "port_scan":
        d = dict(duration=r.gamma(1, 0.5, n), src_bytes=r.lognormal(4.5, 0.5, n),
                 dst_bytes=r.lognormal(3.5, 0.8, n), packets=r.poisson(60, n),
                 syn_count=r.poisson(45, n), ack_count=r.poisson(3, n),
                 unique_dst_ports=r.poisson(150, n) + 20, failed_logins=r.poisson(0.1, n),
                 conn_rate=r.gamma(6, 5, n))
    elif label == "syn_flood":
        d = dict(duration=r.gamma(1, 0.3, n), src_bytes=r.lognormal(4.0, 0.4, n),
                 dst_bytes=r.lognormal(2.5, 0.6, n), packets=r.poisson(400, n),
                 syn_count=r.poisson(380, n), ack_count=r.poisson(5, n),
                 unique_dst_ports=r.poisson(2, n) + 1, failed_logins=r.poisson(0.1, n),
                 conn_rate=r.gamma(8, 25, n))
    else:  # brute_force
        d = dict(duration=r.gamma(3, 4, n), src_bytes=r.lognormal(6.0, 0.7, n),
                 dst_bytes=r.lognormal(6.0, 0.8, n), packets=r.poisson(70, n),
                 syn_count=r.poisson(3, n), ack_count=r.poisson(60, n),
                 unique_dst_ports=r.poisson(1, n) + 1, failed_logins=r.poisson(12, n) + 3,
                 conn_rate=r.gamma(3, 4, n))
    df = pd.DataFrame(d).astype(float)
    if label != "normal":
        # "Stealthy" attacks: ~3% of flows are blended 10/90 with normal-looking
        # traffic, so classes overlap like they do in the real world.
        stealth = r.random(n) < 0.03
        base = make_class("normal", n)
        for c in FEATURES:
            df.loc[stealth, c] = 0.1 * df.loc[stealth, c] + 0.9 * base.loc[stealth, c]
    df["label"] = label
    return df


def main():
    data = pd.concat([make_class(l, N_PER_CLASS) for l in
                      ["normal", "port_scan", "syn_flood", "brute_force"]], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data.to_csv("flows.csv", index=False)
    print(f"Dataset: {data.shape[0]} flows, {len(FEATURES)} features")
    print(data["label"].value_counts().to_string(), "\n")

    X, y = data[FEATURES], data["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print("5-fold CV accuracy per fold:", np.round(cv, 4))
    print(f"CV mean: {cv.mean():.4f}\n")
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    print("=== Random Forest: held-out test set ===")
    print(classification_report(y_te, pred, digits=3))
    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):\n", labels, "\n", cm, "\n")
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(xticks_rotation=30)
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    print("Feature importance (high = more useful):\n", imp.sort_values(ascending=False).round(3).to_string(), "\n")
    imp.plot(kind="barh", title="Feature importance"); plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120); plt.close()

    # ---- Isolation Forest (normal traffic only) ----
    normal_tr = X_tr[y_tr == "normal"]
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    iso.fit(normal_tr)  # no attack labels used
    is_anom = iso.predict(X_te) == -1
    is_attack = (y_te != "normal").values
    recall = is_anom[is_attack].mean()
    false_alarm = is_anom[~is_attack].mean()
    print("=== Isolation Forest (trained on normal only) ===")
    print(f"Attack detection rate (recall): {recall:.1%}")
    print(f"False alarm rate on normal traffic: {false_alarm:.1%}\n")

    # ---- Save models ----
    joblib.dump(rf, "random_forest.joblib")
    joblib.dump(iso, "isolation_forest.joblib")
    print("Saved: random_forest.joblib, isolation_forest.joblib")

    # ---- Demo: load and use a saved model ----
    model = joblib.load("random_forest.joblib")
    print("Demo prediction on 3 test flows:", model.predict(X_te.iloc[:3]).tolist(),
          "| actual:", y_te.iloc[:3].tolist())


if __name__ == "__main__":
    main()
