import joblib
import pickle
import os
import mlflow
import mlflow.sklearn
import pandas as pd

with open("models/reports.pkl", "rb") as f:
    reports = pickle.load(f)

df_reports = pd.DataFrame(reports)

best_model_index = df_reports['f1_score'].idxmax()
best_model_name = df_reports.loc[best_model_index, 'model']
best_model_path = f"models/{best_model_name}.pkl"
print(f"Best model: {best_model_name}")


best_model = joblib.load(best_model_path)

mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="model",
    registered_model_name="Best_Wine_Classification_Model"
)

print(f"Model '{best_model_name}' registered ")
