"""
utils/data_loader.py
Cached data loading + ML-based Dropout Risk Engine (Random Forest).
Replaces the old heuristic point system with real predict_proba output.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
import shap

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# ── Probability threshold for flagging (lowered from 0.50 to catch more) ──────
RISK_THRESHOLD = 0.35

# ── CSV Loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_patients() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "patients_intake.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_sessions() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "session_logs.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_therapists() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "therapists.csv")
    df = pd.read_csv(path)
    return df


def get_patient_sessions(sessions_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    return (
        sessions_df[sessions_df["patient_id"] == patient_id]
        .sort_values("session_number")
        .reset_index(drop=True)
    )


# ── ML Feature Engineering ────────────────────────────────────────────────────

def encode_gender(gender: str) -> int:
    return 1 if gender == "Male" else 0


def encode_insurance(status: str) -> int:
    if status == "Uninsured":
        return 2
    elif status == "Out-of-Pocket":
        return 1
    return 0


def _build_training_data(sessions_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-session feature matrix for training.
    Each row = one session (with ≥1 prior session for delta calculation).
    Target = did_dropout at that session.
    """
    rows = []
    for pid, grp in sessions_df.groupby("patient_id"):
        grp = grp.sort_values("session_number")
        p = patients_df[patients_df["patient_id"] == pid]
        if p.empty or len(grp) < 2:
            continue
        p = p.iloc[0]

        for i in range(1, len(grp)):
            curr = grp.iloc[i]
            prev = grp.iloc[i - 1]
            srs_drop = prev["srs_overall"] - curr["srs_overall"]
            goals_drop = prev["srs_goals"] - curr["srs_goals"]

            rows.append({
                "age": p["age"],
                "gender": encode_gender(p["gender"]),
                "insurance": encode_insurance(p["insurance_status"]),
                "phq9": p["baseline_phq9"],
                "gad7": p["baseline_gad7"],
                "session_number": curr["session_number"],
                "days_since_last": curr["days_since_last_session"],
                "no_shows": curr["no_show_count"],
                "srs_overall": curr["srs_overall"],
                "srs_goals": curr["srs_goals"],
                "srs_relationship": curr["srs_relationship"],
                "srs_approach": curr["srs_approach"],
                "srs_drop": srs_drop,
                "goals_drop": goals_drop,
                "did_dropout": curr["did_dropout"],
            })

    return pd.DataFrame(rows)


# ── Train Model (cached — runs once) ─────────────────────────────────────────

@st.cache_resource
def train_dropout_model():
    """
    Train a RandomForestClassifier on the full historical dataset.
    Returns (model, metrics_dict).
    Cached so it only trains once per app session.
    """
    patients_df = pd.read_csv(os.path.join(DATA_DIR, "patients_intake.csv"))
    sessions_df = pd.read_csv(os.path.join(DATA_DIR, "session_logs.csv"))

    df = _build_training_data(sessions_df, patients_df)
    feature_cols = [
        "age", "gender", "insurance", "phq9", "gad7",
        "session_number", "days_since_last", "no_shows",
        "srs_overall", "srs_goals", "srs_relationship", "srs_approach",
        "srs_drop", "goals_drop",
    ]
    X = df[feature_cols]
    y = df["did_dropout"]

    # Train with class_weight='balanced' to handle 4.3% minority class
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        max_depth=8,
        min_samples_leaf=3,
    )

    # Cross-validate for metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (y_prob_cv >= RISK_THRESHOLD).astype(int)

    auc = roc_auc_score(y, y_prob_cv)
    report = classification_report(y, y_pred_cv, target_names=["Retained", "Dropout"], output_dict=True)

    # Final fit on full data
    clf.fit(X, y)

    # Feature importances
    importances = dict(zip(feature_cols, clf.feature_importances_))

    metrics = {
        "roc_auc": round(auc, 4),
        "accuracy": round(report["accuracy"], 4),
        "dropout_precision": round(report["Dropout"]["precision"], 4),
        "dropout_recall": round(report["Dropout"]["recall"], 4),
        "dropout_f1": round(report["Dropout"]["f1-score"], 4),
        "training_samples": len(X),
        "dropout_count": int(y.sum()),
        "feature_importances": importances,
        "threshold": RISK_THRESHOLD,
    }

    return clf, metrics


# ── ML-Based Risk Flagging ────────────────────────────────────────────────────

def get_high_risk_patients(sessions_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    ML-powered risk assessment. Replaces old heuristic.
    Uses RandomForest predict_proba for true probabilistic dropout risk.
    Includes Unilateral Graduation safeguard.
    """
    model, metrics = train_dropout_model()
    feature_cols = [
        "age", "gender", "insurance", "phq9", "gad7",
        "session_number", "days_since_last", "no_shows",
        "srs_overall", "srs_goals", "srs_relationship", "srs_approach",
        "srs_drop", "goals_drop",
    ]

    results = []
    for pid, group in sessions_df.groupby("patient_id"):
        group = group.sort_values("session_number")
        if len(group) == 1:
            last = group.iloc[-1]
            prev = last
        else:
            last = group.iloc[-1]
            prev = group.iloc[-2]

        # We will no longer skip Dropped/Graduated here so they can be viewed historically in the Hub.

        # Get patient demographics
        patient_info = patients_df[patients_df["patient_id"] == pid]
        if patient_info.empty:
            continue
        p = patient_info.iloc[0]

        # Engineer features for this patient's latest session
        srs_drop = prev["srs_overall"] - last["srs_overall"]
        goals_drop = prev["srs_goals"] - last["srs_goals"]

        features = pd.DataFrame([{
            "age": p["age"],
            "gender": encode_gender(p["gender"]),
            "insurance": encode_insurance(p["insurance_status"]),
            "phq9": p["baseline_phq9"],
            "gad7": p["baseline_gad7"],
            "session_number": last["session_number"],
            "days_since_last": last["days_since_last_session"],
            "no_shows": last["no_show_count"],
            "srs_overall": last["srs_overall"],
            "srs_goals": last["srs_goals"],
            "srs_relationship": last["srs_relationship"],
            "srs_approach": last["srs_approach"],
            "srs_drop": srs_drop,
            "goals_drop": goals_drop,
        }])

        # Get ML probability
        dropout_prob = model.predict_proba(features[feature_cols])[:, 1][0]
        risk_score = round(dropout_prob * 100, 1)

        # ── Unilateral Graduation Safeguard ───────────────────────────
        # If SRS is high but patient missed sessions, they may be
        # "cured" and simply stopped — NOT a dropout crisis.
        is_graduation_candidate = (
            last["srs_overall"] >= 8.0 and last["no_show_count"] >= 1
        )

        # ── SHAP AI Reasoning ───────────────────────────────────────────
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features[feature_cols])
        # For binary classification RF, shap_values has shape (n_samples, n_features, n_classes)
        # We want the values for the 1st (and only) sample, all features, for class 1 (dropout)
        patient_shap = shap_values[0, :, 1]
        
        feature_impacts = sorted(zip(feature_cols, patient_shap), key=lambda x: -x[1])
        reasons = []
        for feat, impact in feature_impacts[:3]:
            if impact > 0.005:  # Only meaningful positive contributions to dropout risk
                val = features[feat].iloc[0]
                if feat == "srs_drop":
                    reasons.append(f"Recent SRS dropped by {val:.1f} pts")
                elif feat == "goals_drop":
                    reasons.append(f"Goals score dropped {val:.1f} pts")
                elif feat == "srs_relationship":
                    reasons.append(f"Low Relationship score ({val})")
                elif feat == "srs_overall":
                    reasons.append(f"Low overall session rating ({val})")
                elif feat == "no_shows":
                    reasons.append(f"Missed {int(val)} recent sessions")
                elif feat == "phq9":
                    reasons.append(f"High baseline depression (PHQ-9: {val})")
                elif feat == "insurance" and val == 2:
                    reasons.append("Uninsured (financial risk)")
                elif feat == "age" and val < 25:
                    reasons.append(f"Age demographic risk ({val})")
                else:
                    reasons.append(f"Driven by {feat.replace('_', ' ')} trend")
                    
        if not reasons:
            reasons.append("Gradual drift in engagement metrics")

        results.append({
            "patient_id": pid,
            "full_name": p["full_name"],
            "therapist": p["assigned_therapist_name"],
            "session_number": int(last["session_number"]),
            "srs_overall": last["srs_overall"],
            "phone": p.get("phone", ""),
            "risk_score": risk_score if len(group) >= 2 else 0,
            "reasons": reasons if len(group) >= 2 else ["Insufficient data (Requires 2+ sessions)"],
            "risk_reasons": " | ".join(reasons) if len(group) >= 2 else "Insufficient data",
            "latest_feedback": str(last.get("optional_feedback", "")),
            "no_shows": int(last["no_show_count"]),
            "is_flagged": bool(risk_score >= (RISK_THRESHOLD * 100)) if len(group) >= 2 else False,
            "is_graduation_candidate": is_graduation_candidate,
            "latest_session": last,
            "previous_session": prev,
            "status": last["status"],
            "has_enough_data": len(group) >= 2
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("risk_score", ascending=False)
    return df.reset_index(drop=True)


# ── Therapist Retention Rates ─────────────────────────────────────────────────

@st.cache_data
def get_therapist_retention_rates() -> pd.DataFrame:
    """
    Compute historical retention rate per therapist from the session data.
    Returns DataFrame with: therapist_id, name, patients, retained, dropouts, retention_rate
    """
    patients_df = pd.read_csv(os.path.join(DATA_DIR, "patients_intake.csv"))
    sessions_df = pd.read_csv(os.path.join(DATA_DIR, "session_logs.csv"))
    therapists_df = pd.read_csv(os.path.join(DATA_DIR, "therapists.csv"))

    # Get final outcome per patient
    outcomes = (
        sessions_df.groupby("patient_id")
        .agg(did_dropout=("did_dropout", "max"), therapist_id=("therapist_id", "last"))
        .reset_index()
    )

    rates = []
    for _, t in therapists_df.iterrows():
        tid = t["therapist_id"]
        subset = outcomes[outcomes["therapist_id"] == tid]
        total = len(subset)
        dropouts = int(subset["did_dropout"].sum())
        retained = total - dropouts
        rate = (retained / total * 100) if total > 0 else 0

        rates.append({
            "therapist_id": tid,
            "name": t["name"],
            "total_patients": total,
            "retained": retained,
            "dropouts": dropouts,
            "retention_rate": round(rate, 1),
        })

    return pd.DataFrame(rates).sort_values("retention_rate", ascending=False).reset_index(drop=True)


def get_model_metrics() -> dict:
    """Return the cached model's performance metrics for display."""
    _, metrics = train_dropout_model()
    return metrics


def mark_patient_dropout(patient_id: str):
    """
    Manually marks a patient as dropped out by updating their latest session 
    in the session_logs.csv and clearing the Streamlit cache.
    """
    path = os.path.join(DATA_DIR, "session_logs.csv")
    df = pd.read_csv(path)
    
    # Find the latest session for the patient
    patient_sessions = df[df["patient_id"] == patient_id].sort_values("session_number")
    if patient_sessions.empty:
        return
        
    latest_idx = patient_sessions.index[-1]
    
    # Update status to Dropped and did_dropout to 1
    df.at[latest_idx, "status"] = "Dropped"
    df.at[latest_idx, "did_dropout"] = 1
    
    # Save back to CSV
    df.to_csv(path, index=False)
    
    # Clear cache so the dashboard immediately updates
    st.cache_data.clear()
