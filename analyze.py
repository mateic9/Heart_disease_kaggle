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
    for col_name in df.columns:
        df[[col_name]] = scaler.fit_transform(df[[col_name]])
    for col_name in df.columns:
        if df[col_name].nunique() < 7:
            categorical_cols.append(col_name)   
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype("category")
        one_hot_dummies = pd.get_dummies(df[col_name], prefix=col_name)
        df = pd.concat([df, one_hot_dummies], axis=1)
        df.drop(col_name, axis=1, inplace=True)
    return df

train_df = pd.read_csv("./train.csv")

test_df = pd.read_csv("./test.csv")
train_df = get_renamed_columns(train_df)
test_df = get_renamed_columns(test_df)

train_df["Heart Disease"] = train_df["Heart Disease"].map(lambda val: f(val))

# print(train_df.iloc[:2])


transformed_df = train_df.drop(["id", "Heart Disease"], axis = 1)
transformed_test_df = test_df.drop(["id"], axis = 1)

transformed_df = handle_input_data(transformed_df)
transformed_test_df = handle_input_data(transformed_test_df)

# print(train_df.nunique())


# h = []
# for col1 in transformed_df.columns:
#     for col2 in transformed_df.columns:
#         if col1 != col2:
#             aux_df = transformed_df.loc[:, [col1, col2]]
#             if col1 < col2:
#                 heapq.heappush(h, (abs(aux_df.corr().iloc[0, 1]), col1, col2))

# print(heapq.nlargest(5, h))

# print(train_df.groupby(["Thallium", "Heart Disease"]).count())
# print(train_df.groupby(["Number of vessels fluro", "Heart Disease"]).count())
# print(train_df.groupby(["EKG", "Heart Disease"]).count())

# print(train_df.dtypes)


  




X_train, X_val, y_train, y_val = train_test_split(transformed_df, train_df["Heart Disease"],stratify=train_df["Heart Disease"],test_size=0.2,random_state=42)

def objective_log(trial):
    # trial_l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    # trial_max_iter = trial.suggest_int("max_iter", 5, 500)
    trial_class_weight = trial.suggest_categorical("class_weight", ["balanced", "none"])
    trial_solver = trial.suggest_categorical("solver", ["saga"])
    trial_l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
 
    log = LogisticRegression(solver = trial_solver,
                             l1_ratio = trial_l1_ratio,
                             class_weight=None if trial_class_weight == "none" else trial_class_weight,
                             max_iter = 400
                             )
    
    log.fit(X_train, y_train)
    y_val_proba = log.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, y_val_proba)

def objective_tree(trial):
    
    trial_criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    trial_min_samples_split = trial.suggest_int("min_samples_split", 3, 8)
    choice = trial.suggest_categorical("max_features", ["sqrt", "log2","none"])
    trial_max_depth = trial.suggest_int("max_depth", 7, 12)
    clf = RandomForestClassifier(
                                 criterion = trial_criterion,
                                 min_samples_split = trial_min_samples_split,
                                 max_features = None if choice == "none" else choice,
                                 max_depth = trial_max_depth
                                 )
    clf.fit(X_train, y_train)   
    clf_val_proba = clf.predict_proba(X_val)[:,1] 
    return roc_auc_score(y_val, clf_val_proba)

#OPTUNA 
study = optuna.create_study(direction = 'maximize')
study.optimize(objective_log, n_trials = 150)
print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}")

# test_df["Heart Disease"] = log.predict_proba(transformed_test_df)[:,1]
# test_df.loc[:, ["id", "Heart Disease"]].to_csv("logistic.csv", index = False)



# test_log_pred = log.predict_proba(transformed_test_df)[:, 1]


# print(confusion_matrix(y_val, log_pred))


#BASIC RANDOM FOREST
# clf = RandomForestClassifier(criterion = "gini",
#                              min_samples_split = 9,
#                              max_depth = 12)
# clf.fit(X_train, y_train)
# clf_val_proba = clf.predict_proba(X_val)[:, 1]



# print("Validation ROC-AUC:", roc_auc_score(y_val, clf_val_proba))
# test_df["Heart Disease"] = clf.predict_proba(transformed_test_df)[:,1]
# test_df.loc[:, ["id", "Heart Disease"]].to_csv("tree.csv", index = False)

# Best parameters: {'criterion': 'log_loss', 'min_samples_split': 7, 'max_features': 'sqrt', 'max_depth': 12}
# skf = StratifiedKFold(n_splits = 8, shuffle=True, random_state=42)

# fold_scores = []
# test_proba_sum = np.zeros(len(transformed_test_df))

# for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
#     X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
#     y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

#     clf = RandomForestClassifier(criterion = "log_loss",
#                                  min_samples_split = 8,
#                                  max_features = "sqrt",
#                                  max_depth = 11,
#                                  random_state = 42)
#     clf.fit(X_tr, y_tr)

#     val_proba = clf.predict_proba(X_va)[:, 1]
#     fold_auc = roc_auc_score(y_va, val_proba)
#     fold_scores.append(fold_auc)
#     print(f"Fold {fold_idx} ROC-AUC: {fold_auc:.4f}")

#     test_proba_sum += clf.predict_proba(transformed_test_df)[:, 1]

# mean_auc = np.mean(fold_scores)
# std_auc = np.std(fold_scores)
# print(f"Mean CV ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
# test_df["Heart Disease"] = test_proba_sum / skf.get_n_splits()
# test_df.loc[:, ["id", "Heart Disease"]].to_csv("tree.csv", index=False)

