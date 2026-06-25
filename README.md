# Fraud-Detection-System-Using-Machine-Learning
#### End-to-end fraud detection system built with Python, Scikit-Learn, SMOTE, and XGBoost. Includes feature engineering, model evaluation, deployment pipeline, and Streamlit integration for real-time fraud prediction.

## Overview
This project develops an end-to-end Fraud Detection System using Machine Learning to identify potentially fraudulent financial transactions. The solution addresses the challenge of highly imbalanced fraud datasets by combining advanced feature engineering, data preprocessing, SMOTE oversampling, and multiple machine learning algorithms.

The final model is deployed as a reusable pipeline and can be integrated into a Streamlit application for real-time fraud prediction.

# Business Problem

Financial institutions process millions of transactions daily, making it difficult to manually identify fraudulent activities. Traditional rule-based systems often miss fraudulent transactions or generate excessive false alarms.

This project aims to:
1. Detect fraudulent transactions accurately.
2. Reduce false positives.
3. Improve fraud monitoring efficiency.
4. Support real-time decision-making.

# Dataset Features

The dataset contains transaction-level information including:

- Transaction Type
- Transaction Amount
- Sender Account Information
- Recipient Account Information
- Account Balances Before Transaction
- Account Balances After Transaction
- Fraud Indicator (isFraud)
- Flagged Fraud Indicator (isFlaggedFraud)

Target Variable: isFraud
- 0 = Legitimate Transaction
- 1 = Fraudulent Transaction

# Project Workflow

## 1. Data Understanding

Performed:

- Dataset inspection
- Data type validation
- Missing value analysis
- Duplicate record detection
- Class distribution analysis

## 2. Exploratory Data Analysis (EDA)

Explored:

- Fraud distribution
- Transaction type patterns
- Transaction amount behavior
- Account balance relationships
- Existing rule-based fraud flag performance

Finding: Fraudulent transactions represent a very small percentage of all transactions, creating a highly imbalanced classification problem.

## 3. Feature Engineering

Created additional features to improve predictive performance:

Log Transformation
- log_amount = np.log1p(amount). Used to reduce skewness in transaction amounts.

High-Value Transaction Flag
- is_high_amount (Flags transactions above the 99th percentile).

Benefits:
- Captures abnormal transaction behavior
- Preserves extreme-value information
- Improves fraud signal detection

## 4. Data Preprocessing

Implemented using Scikit-Learn's ColumnTransformer.

- Numerical Features: RobustScaler
- Categorical Features: OneHotEncoder

Benefits:

Handles outliers effectively
Encodes categorical variables for machine learning models.

## 5. Class Imbalance Handling

Used: SMOTE (Synthetic Minority Oversampling Technique)

Applied inside an Imbalanced-Learn Pipeline to prevent data leakage.

Benefits:

- Generates synthetic fraud examples
- Improves minority class learning
- Preserves model evaluation integrity

# Machine Learning Models
## Model 1: Logistic Regression

Advantages:

Fast training
Highly interpretable
Strong baseline model

## Model 2: Random Forest

Advantages:

- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance

## Model 3: XGBoost

Advantages:

- High predictive performance
- Robust handling of complex fraud patterns
- Excellent performance on imbalanced datasets

The XGBoost model achieved the best overall performance and was selected as the final deployment model.

# Model Evaluation Metrics
<img width="910" height="337" alt="image" src="https://github.com/user-attachments/assets/38ed59d7-86c6-44ac-b2ca-43d170862638" />



<img width="873" height="437" alt="image" src="https://github.com/user-attachments/assets/c580fcd2-d206-400b-9d68-6081b9668631" />

### Observation:

- Detected many fraud cases.
- Generated excessive false alarms.
- Missed 268 fraudulent transactions.

<img width="921" height="362" alt="image" src="https://github.com/user-attachments/assets/46959771-e7d7-442b-b15d-f0314b2a32e7" />

### Observation:

- Near-perfect fraud detection.
- Only 4 fraud cases missed.
- No false alarms generated.

<img width="890" height="382" alt="image" src="https://github.com/user-attachments/assets/8dc06ab4-85ba-485d-8013-37678dacc231" />

### Observation:

- Extremely strong fraud detection capability.
- Very low number of missed fraud cases.
- Small number of false alerts.





## The models were evaluated using:

- Classification Report
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Precision-Recall AUC
- Confusion Matrix

These metrics provide a balanced assessment of fraud detection effectiveness.

# Feature Importance Analysis

Feature importance was extracted from the trained XGBoost model to identify the most influential variables driving fraud predictions.

Benefits:

- Improves explainability
- Supports business decision-making
- Identifies key fraud risk indicators

# Model Deployment

The final production artifact includes:

{
    "model": pipeline_xgb,
    "threshold": 0.5,
    "features": list(X.columns)
}

Saved using:

joblib.dump()

Output File:

fraud_detection_best_pipeline.joblib

This artifact can be loaded directly into a Streamlit application for real-time fraud prediction.

# Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn
- XGBoost
- Joblib
- Streamlit
