import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import heapq
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_renamed_columns(df):
    df = df.rename(columns={"EKG results": "EKG", "Chest pain type": "Chest", "Number of vessels fluro": "Vessels",
                            "Cholesterol": "Chol", "ST depression": "Depr"})
    return df
def f(val):
    if val == "Presence":
        return 1
        
    return 0

def handle_input_data(df):
    scaler = StandardScaler()
    categorical_cols = []
    numerical_cols = []
    for col_name in df.columns:
        df[[col_name]] = scaler.fit_transform(df[[col_name]])    
    for col_name in df.columns:
        if df[col_name].nunique() < 7:
            categorical_cols.append(col_name)   
        else:    
            numerical_cols.append(col_name)
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype("category")
        one_hot_dummies = pd.get_dummies(df[col_name], prefix=col_name)


        df = pd.concat([df, one_hot_dummies], axis=1)
        df.drop(col_name, axis=1, inplace=True)
    print(numerical_cols)
    print(categorical_cols)
    return df

train_df = pd.read_csv("./train.csv")

test_df = pd.read_csv("./test.csv")
train_df = get_renamed_columns(train_df)
test_df = get_renamed_columns(test_df)

train_df["Heart Disease"] = train_df["Heart Disease"].map(lambda val: f(val))

# print(train_df.iloc[:2])


transformed_df = train_df.drop(["id", "Heart Disease"], axis = 1)
transformed_test_df = test_df.drop(["id"], axis = 1)


n_train = len(transformed_df)
combined_df = pd.concat([transformed_df, transformed_test_df], axis=0, ignore_index=True)

combined_df = handle_input_data(combined_df)

# split back into train and test
transformed_df = combined_df.iloc[:n_train].reset_index(drop=True)
transformed_test_df = combined_df.iloc[n_train:].reset_index(drop=True)

X_train, X_val, y_train, y_val = train_test_split(transformed_df, train_df["Heart Disease"], stratify = train_df["Heart Disease"], 
                                                  test_size = 0.2, random_state = 142)

log = LogisticRegression(solver="saga",
                         class_weight = "balanced"
                        )
log.fit(X_train, y_train)
preds = log.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, preds))


# final_log = LogisticRegression(solver="newton-cholesky",
#                                class_weight= "balanced"
#                               ) 
# initial_coeffs = log.coef_[0].copy()  # Store initial coefficients
# final_log.fit(transformed_df, train_df["Heart Disease"])
# submission = final_log.predict_proba(transformed_test_df)[:, 1]

# submission_df = pd.DataFrame({"id": test_df["id"], "Heart Disease": submission})
# submission_df.to_csv("newton_cholesky.csv", index=False)


# coef_df = pd.DataFrame({
#     "feature": transformed_df.columns,
#     "new weight": final_log.coef_[0],
#     "initial weight": initial_coeffs
# })

# print(f"Intercept: {final_log.intercept_[0]:.4f}")
# print(coef_df.to_string(index=False))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    LogisticRegression(solver="saga", class_weight="balanced", max_iter=400, l1_ratio = 0.2),
    transformed_df, train_df["Heart Disease"],
    
    cv=StratifiedKFold(5, shuffle=True, random_state = 22),
    scoring="roc_auc",
)
print(scores.mean(), scores.std())
