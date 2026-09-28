
# Customer Churn Prediction

An AI-assisted machine learning project that predicts customer churn using Python and Scikit-learn. It includes exploratory data analysis, model comparison, hyperparameter tuning, SHAP explanations, and an interactive Streamlit application.

## Project Overview

Customer churn occurs when customers stop using a company's services. This project explores how machine learning can help identify customers who may be at risk of leaving.

The application accepts customer details and returns an estimated churn probability, a risk classification, and a SHAP-based explanation of the model's prediction.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Logistic Regression, Random Forest, and Gradient Boosting
- Stratified 5-fold cross-validation
- Hyperparameter tuning using GridSearchCV
- Classification threshold experiments
- Model interpretation using SHAP
- Interactive Streamlit web application

## Dataset

The project uses IBM's Telco Customer Churn sample dataset.

- Customers: 7,043
- Original columns: 21
- Target: `Churn` (`Yes` or `No`)

The target was encoded as:
- `No` = 0
- `Yes` = 1

Dataset source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- SHAP
- Joblib

## Machine Learning Workflow

1. Loaded and inspected the dataset.
2. Cleaned `TotalCharges` and handled its missing values.
3. Conducted EDA to investigate patterns associated with churn.
4. Created training and testing datasets using a stratified split.
5. Used `ColumnTransformer` for numerical scaling and categorical encoding.
6. Compared Logistic Regression, Random Forest, and Gradient Boosting.
7. Used cross-validation and GridSearchCV for model evaluation and tuning.
8. Experimented with different classification thresholds.
9. Used SHAP to interpret individual predictions.
10. Integrated the trained model into a Streamlit application.

## Preliminary Results

Mean 5-fold cross-validation ROC-AUC scores from the initial model comparison:

| Model | Mean ROC-AUC |
|---|---:|
| Logistic Regression | 0.8468 |
| Random Forest | 0.8193 |
| Gradient Boosting | 0.8493 |

Gradient Boosting achieved the highest mean ROC-AUC among these initial configurations, although its performance was close to Logistic Regression.

The tuned Gradient Boosting model achieved a test ROC-AUC of approximately 0.8419 in the current experiment.

**Evaluation note:** The test set was consulted during model and threshold experimentation. These results are exploratory and should not be treated as an unbiased final estimate of generalization performance.

## Running the Application

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Customer Churn Prediction"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

The repository includes the trained model file, so you do not need to retrain it just to run the application.

## Project Structure

```text
Customer Churn Prediction/
├── Churn.py
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- The dataset is a historical sample for a fictional telecommunications company.
- Model predictions describe patterns in the data and do not establish causation.
- The 30% classification threshold is exploratory, not a validated business optimum.
- Model probability calibration and performance on genuinely new customer data have not been established.
- The application is an educational prototype, not a production-validated retention system.

## AI Assistance

AI tools were used as coding assistants for parts of the implementation, explanations, and debugging. The project was developed as a learning exercise, with an emphasis on understanding the ML workflow and interpreting experimental results.