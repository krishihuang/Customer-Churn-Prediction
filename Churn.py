
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load and clean the data
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
).fillna(0)


# 2. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=40,
    stratify=y
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# 4. Define feature types
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


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)


# 6. Create candidate models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=40
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=40
        ))
    ])
}


# 7. Cross-validation model comparison
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=40
)

for name, estimator in models.items():
    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    print(f"{name}: {scores.mean():.4f} "
          f"+/- {scores.std():.4f}")


# 8. Tune Gradient Boosting
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3]
}

grid_search = GridSearchCV(
    estimator=models["Gradient Boosting"],
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_gb = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)


# 9. Select threshold using out-of-fold predictions
oof_prob = cross_val_predict(
    best_gb,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

thresholds = np.arange(0.20, 0.71, 0.05)
results = []

for threshold in thresholds:
    pred = (oof_prob >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_train, pred, zero_division=0
        ),
        "F1": f1_score(
            y_train, pred, zero_division=0
        )
    })

results_df = pd.DataFrame(results)
best_index = results_df["F1"].idxmax()
best_threshold = float(results_df.loc[best_index, "Threshold"])

print("\nThreshold results:")
print(results_df.to_string(index=False))
print("\nSelected threshold:", best_threshold)


# 10. Final test evaluation
test_prob = best_gb.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nFinal test results")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(
    y_test,
    test_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))
print("ROC-AUC:", roc_auc_score(y_test, test_prob))


# 11. Feature importance
feature_names = (
    best_gb.named_steps["preprocessor"].get_feature_names_out()
)
importances = best_gb.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))


# 12. Save model and threshold together
joblib.dump(
    {
        "model": best_gb,
        "threshold": best_threshold
    },
    "churn_model.pkl"
)

print("\nModel and threshold saved successfully.")