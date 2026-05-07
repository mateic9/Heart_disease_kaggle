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
    for col_name in df.columns:
        df[[col_name]] = scaler.fit_transform(df[[col_name]])
       
    # df["Chol"] = pd.qcut(df["Chol"], 7, duplicates = "drop" ,labels=False)
    # df["Depr"] = pd.qcut(df["Depr"], 4, duplicates = "drop", labels=False)
    # df["BP"] = pd.qcut(df["BP"], 5, duplicates = "drop", labels=False)
    df = df.apply(lambda x: x.astype('category') if x.dtype == 'object' else x)

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


  

for col_name in transformed_df.columns:
    print(f"{col_name}, {transformed_df[col_name].max() - transformed_df[col_name].min()}")



X_train, X_val, y_train, y_val = train_test_split(transformed_df, train_df["Heart Disease"],stratify=train_df["Heart Disease"],test_size=0.2,random_state=42)

def objective_regression(trial):
    trial_l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    log = LogisticRegression(solver="newton-cholesky" , 
                             penalty = "l2" )
    log.fit(X_train, y_train)
    y_val_proba = log.predict_proba(X_val)[:,1]
    return roc_auc_score(y_val, y_val_proba)

def objective_tree(trial):
    
    trial_criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    trial_min_samples_split = trial.suggest_int("min_samples_split", 6, 10)
    
    trial_max_depth = trial.suggest_int("max_depth", 7, 12)
    clf = RandomForestClassifier(
                                 criterion = trial_criterion,
                                 min_samples_split = trial_min_samples_split,
                                #  max_features = trial_max_features,
                                 max_depth = trial_max_depth
                                 )
    clf.fit(X_train, y_train)   
    clf_val_proba = clf.predict_proba(X_val)[:,1] 
    return roc_auc_score(y_val, clf_val_proba)
# study = optuna.create_study(direction='maximize')
# study.optimize(objective_tree, n_trials=50)
# print(f"Best parameters: {study.best_params}")
# print(f"Best accuracy: {study.best_value}")

# test_df["Heart Disease"] = log.predict_proba(transformed_test_df)[:,1]
# test_df.loc[:, ["id", "Heart Disease"]].to_csv("logistic.csv", index = False)



# test_log_pred = log.predict_proba(transformed_test_df)[:, 1]


# print(confusion_matrix(y_val, log_pred))

clf = RandomForestClassifier(criterion = "gini",
                             min_samples_split = 9,
                             max_depth = 12)
clf.fit(X_train, y_train)
clf_val_proba = clf.predict_proba(X_val)[:,1]


print("Validation ROC-AUC:", roc_auc_score(y_val, clf_val_proba))
test_df["Heart Disease"] = clf.predict_proba(transformed_test_df)[:,1]
test_df.loc[:, ["id", "Heart Disease"]].to_csv("tree.csv", index = False)