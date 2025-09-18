import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess_data():
    """
    Load and preprocess the Wine Quality dataset
    """
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    
    # Save raw data
    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv('data/raw_wine_data.csv', index=False)
    
    # Preprocessing
    # Convert quality to binary classification (good wine: quality >= 7, bad wine: quality < 7)
    data['quality_binary'] = (data['quality'] >= 7).astype(int)
    
    # Features and target
    X = data.drop(['quality', 'quality_binary'], axis=1)
    y = data['quality_binary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    np.save('data/X_train_scaled.npy', X_train_scaled)
    np.save('data/X_test_scaled.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    print(f"Data preprocessing completed!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")