import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def infer_problem_type(y: pd.Series) -> str:
    # Heuristique simple: numérique avec beaucoup de valeurs uniques => régression, sinon classification
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique >= 15:
            return "regression"
    return "classification"

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

def build_model(problem_type: str, random_state: int = 42):
    if problem_type == "regression":
        return RandomForestRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1
        )
    return RandomForestClassifier(
        n_estimators=400, random_state=random_state, n_jobs=-1
    )

def evaluate(problem_type: str, y_true, y_pred, y_proba=None) -> dict:
    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    out = {"accuracy": acc, "f1_weighted": f1}

    # ROC AUC seulement si binaire + proba dispo
    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        pass

    return out

def train_tabular(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in columns.")

    df = df.copy()
    y = df[target]
    X = df.drop(columns=[target])

    problem_type = infer_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    pre = build_preprocessor(X_train)
    model = build_model(problem_type, random_state=random_state)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    y_proba = None
    if problem_type == "classification":
        try:
            # proba pour la classe 1 (si binaire)
            proba = pipe.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]
        except Exception:
            y_proba = None

    metrics = evaluate(problem_type, y_test, y_pred, y_proba=y_proba)
    return pipe, {"problem_type": problem_type, "metrics": metrics, "n_rows": int(len(df)), "n_features": int(X.shape[1])}
