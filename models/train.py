"""
Trains an XGBoost fraud classifier on the synthetic dataset.
Saves: models/fraud_model.pkl, models/scaler.pkl, models/feature_names.pkl
"""
import pickle, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "data/transactions.csv"
MODEL_DIR = "models"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] % 86400) // 3600
    df["Is_night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)
    df["V1_V3_ratio"] = df["V1"] / (df["V3"].abs() + 1e-5)
    df["High_amount"] = (df["Amount"] > 500).astype(int)
    return df

def train():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    drop_cols = ["Time", "Amount", "Class"]
    X = df.drop(columns=drop_cols)
    y = df["Class"]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=N_legit_ratio(y_train),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_sc, y_train, verbose=False)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("=== Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\nModel saved to {MODEL_DIR}/")
    return auc

def N_legit_ratio(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)

if __name__ == "__main__":
    train()
