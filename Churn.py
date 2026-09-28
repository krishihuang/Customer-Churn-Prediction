import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
'''print(df.head())
print(df.shape)
print(df.info())
print(df.columns)'''

'''print(df["Churn"].value_counts())
print(df["TotalCharges"].unique()[:20])
print((df["TotalCharges"].str.strip() == "").sum())
print(df[df["TotalCharges"].str.strip() == ""][["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])'''

'''print(df[df["TotalCharges"].str.strip() == ""]["tenure"].value_counts())'''
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(),errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

#print(df.groupby("Contract")["Churn"].value_counts(normalize=True))
#print(df.groupby("tenure")["Churn"].value_counts(normalize=True).to_string())

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1,12,24,36,48,60,72],
    labels=["0-12","13-24","25-36","37-48","49-60","61-72"]
)
#print(df.groupby("TenureGroup",observed=True)["Churn"].value_counts(normalize=True))

#print(df.groupby("InternetService")["Churn"].value_counts(normalize=True))
#print(df.groupby("Churn")["MonthlyCharges"].mean())
#print(df.groupby("PaymentMethod")["Churn"].value_counts(normalize=True))
#print(df.groupby("TechSupport")["Churn"].value_counts(normalize=True))
#print(df.groupby("OnlineSecurity")["Churn"].value_counts(normalize=True))
'''print(df.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True))
print(df.groupby("Partner")["Churn"].value_counts(normalize=True))
print(df.groupby("Dependents")["Churn"].value_counts(normalize=True))'''
'''print(df.groupby("gender")["Churn"].value_counts(normalize=True))

df["Churn"].value_counts().plot(kind = "bar")
df.boxplot(column = "MonthlyCharges", by = "Churn")

plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()'''
X = df.drop(columns = ["Churn","TenureGroup","customerID"])
y = df["Churn"].map({"No":0,"Yes":1})
#print(X.shape)
#print(y.shape)
X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.8,random_state=40,stratify=y)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]
categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]
preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
    ]
)
model = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=1000))
    ]
)
model.fit(X_train,y_train)
y_p = model.predict(X_test)
'''print("Accuracy: ",accuracy_score(y_test,y_p))
print(confusion_matrix(y_test,y_p))'''
#print(classification_report(y_test,y_p))
y_prob = model.predict_proba(X_test)[:,1]
#print(y_prob[:10])
auc = roc_auc_score(y_test,y_prob)
#print("ROC-AUC: ",auc)
y_pred40 = (y_prob>=0.40).astype(int)
#print(classification_report(y_test, y_pred40))
rf_model = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("model",RandomForestClassifier(n_estimators=300,random_state=40))
    ]
)
rf_model.fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]
'''print("Accuracy: ",accuracy_score(y_test,rf_pred))
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))'''
thresholds = np.arange(0.20,0.71,0.05)
results = []

for thres in thresholds:
    y_pred_thres = (y_prob>=thres).astype(int)

    precision = precision_score(y_test, y_pred_thres)
    recall = recall_score(y_test, y_pred_thres)
    f1 = f1_score(y_test, y_pred_thres)
    results.append([thres,precision,recall,f1])
result_df = pd.DataFrame(
    results,
    columns =["Threshold","Precision","Recall","F1"]
)
#print(result_df)

'''plt.plot(result_df["Threshold"],result_df["Precision"],label = "Precision")
plt.plot(result_df["Threshold"],result_df["Recall"],label = "Recall")
plt.plot(result_df["Threshold"],result_df["F1"],label = "F1")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision,Recall and F1 vs Threshold")
plt.legend()
plt.show()'''

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=40

)
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})
importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)
#print(importance_df)
model_balanced = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
         )
    ]
)
model_balanced.fit(X_train,y_train)
balanced_pred = model_balanced.predict(X_test)
balanced_prob = model_balanced.predict_proba(X_test)[:,1]
print(classification_report(y_test,balanced_pred))
print("ROC-AUC: ",roc_auc_score(y_test,balanced_prob))
preprocessor_gb = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical_features)

    ]
)
gb_model = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("model",GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=3,random_state=40))
    ]
)
gb_model.fit(X_train,y_train)
gb_pred = gb_model.predict(X_test)
gb_prob = gb_model.predict_proba(X_test)[:,1]
#print(classification_report(y_test, gb_pred))
#print("ROC-AUC:", roc_auc_score(y_test, gb_prob))
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=40
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv = cv,
    scoring = "roc_auc"
)
print(scores)
print("Mean ROC-AUC: ",scores.mean())
rf_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv = cv,
    scoring="roc_auc"
)
print("Random Forest:", rf_scores)
print("Mean ROC-AUC:", rf_scores.mean())
gb_scores = cross_val_score(
    gb_model,
    X_train,
    y_train,
    cv = cv,
    scoring="roc_auc"
)
print("Gradient Boosting:", gb_scores)
print("Mean ROC-AUC:", gb_scores.mean())

param_grid = {
    "model__n_estimators": [100,200],
    "model__learning_rate": [0.03,0.05,0.1],
    "model__max_depth": [2,3]
}
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
grid_search.fit(X_train,y_train)
#print("Best parameters:", grid_search.best_params_)
#print("Best CV ROC-AUC:", grid_search.best_score_)
best_gb = grid_search.best_estimator_
tuned_pred = best_gb.predict(X_test)
tuned_prob = best_gb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, tuned_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, tuned_prob))
thresholds = np.arange(0.20,0.71,0.05)
results_gb = []
for threshold in thresholds:
    pred = (tuned_prob>=threshold).astype(int)

    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    results_gb.append([threshold,precision,recall,f1])

results_gb_df = pd.DataFrame(
    results_gb,
    columns=["Threshold","Precision","Recall","F1"]
)
print(results_gb_df)
best_row = results_gb_df.loc[
    results_gb_df["F1"].idxmax()
]
print(best_row)

final_model = best_gb.named_steps["model"]
final_preprocessor = best_gb.named_steps["preprocessor"]
feature_names = final_preprocessor.get_feature_names_out()
importances = final_model.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature":feature_names,
        "Importance":importances
    }
)
importance_df = importance_df.sort_values("Importance",ascending=False)
print(importance_df.head(15))
joblib.dump(best_gb,"churn_model.pkl")
loaded_model = joblib.load("churn_model.pkl")
print(type(loaded_model))
