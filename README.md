# MLOps Assignment 1

## Problem Statement

We trained and compared ML models on the Wine dataset using MLOps practices.
MLflow was used for tracking experiments and model registration.

---

## Dataset

Wine dataset from scikit-learn.
It contains 13 features about wine (alcohol, ash, flavanoids, etc.).
Target variable has 3 classes of wine.

---

## Models Trained

Logistic Regression
Random Forest
Support Vector Machine (SVM)

Metrics used: Accuracy, Precision, Recall, F1-score.
Best model: Random Forest.

---

## MLflow

Logged metrics, parameters, and models.
Registered the best model in MLflow registry.

<img width="1627" height="431" alt="image" src="https://github.com/user-attachments/assets/b7b92a9e-0754-4c8b-8938-869b70317e21" />

<img width="1657" height="711" alt="image" src="https://github.com/user-attachments/assets/2d4aca49-1a4c-45bf-874c-3c2cb89975f4" />




---

## How to Run

Step 1: Clone the repository.
Step 2: Install the required libraries from requirements.txt.
Step 3: Run data\_preprocessing.py, model\_run.py, and model\_evaluation.py.
Step 4: Start MLflow UI with the command “mlflow ui” and open it in your browser.


