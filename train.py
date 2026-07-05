import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def get_renamed_columns(df):
    return df.rename(columns={
        "EKG results": "EKG",
        "Chest pain type": "Chest",
        "Number of vessels fluro": "Vessels",
        "Cholesterol": "Chol",
        "ST depression": "Depr",
        "Slope of ST": "Slope of ST",
    })


def target_to_int(val):
    return 1 if val == "Presence" else 0


train_df = pd.read_csv("train.csv")
train_df = get_renamed_columns(train_df)

train_df["Heart Disease"] = train_df["Heart Disease"].map(target_to_int)

X = train_df.drop(["id", "Heart Disease"], axis=1)
y = train_df["Heart Disease"]

categorical_cols = [
    "Sex",
    "Chest",
    "FBS over 120",
    "EKG",
    "Exercise angina",
    "Slope of ST",
    "Vessels",
    "Thallium",
]

numerical_cols = [
    "Age",
    "BP",
    "Chol",
    "Max HR",
    "Depr",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=3000
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    return_estimator=True,
)

print("AUC mean:", scores["test_score"].mean())
print("AUC std:", scores["test_score"].std())

for i, model in enumerate(scores["estimator"], start=1):
    joblib.dump(model, MODEL_DIR / f"model_fold_{i}.joblib")

pipeline.fit(X, y)
joblib.dump(pipeline, MODEL_DIR / "heart_pipeline.joblib")

print("Saved models.")