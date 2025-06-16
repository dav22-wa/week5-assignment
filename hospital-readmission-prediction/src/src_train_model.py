```
"""
Train an XGBoost model for hospital readmission prediction.
Includes hyperparameter tuning and evaluation metrics.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

def load_processed_data():
    """Load preprocessed data and labels."""
    X = pd.read_csv('../data/processed_data.csv').values
    y = pd.read_csv('../data/labels.csv').values.ravel()
    return X, y

def train_xgboost(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """Evaluate model using precision, recall, and confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    return cm, precision, recall, f1

def main():
    """Train and save the model."""
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)
    
    model, best_params = train_xgboost(X_train, y_train)
    print("Best Parameters:", best_params)
    
    evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, '../models/xgboost_model.pkl')

if __name__ == '__main__':
    main()
```