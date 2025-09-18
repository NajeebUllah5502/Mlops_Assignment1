
import mlflow
import mlflow.sklearn
import os
import pickle
import joblib
import pandas as pd

model_files = [f for f in os.listdir("models") if f.endswith(".pkl") and f != "reports.pkl"]
with open("models/reports.pkl", "rb") as f:
    reports = pickle.load(f)

mlflow.set_experiment("Wine_Classification_Models")
mlflow.set_tracking_uri("http://localhost:5000")  

for i, model_file in enumerate(model_files):
    model_name = model_file.replace(".pkl", "")
    model = joblib.load(f"models/{model_file}")
    
    with mlflow.start_run(run_name=model_name):

        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
        elif model_name == "LogisticRegression":
            mlflow.log_param("max_iter", model.max_iter)
        elif model_name == "SVM":
            mlflow.log_param("kernel", model.kernel)
    
        report = reports[i]
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision", report["precision"])
        mlflow.log_metric("recall", report["recall"])
        mlflow.log_metric("f1_score", report["f1_score"])
        
     
        mlflow.sklearn.log_model(model, "model")
        
