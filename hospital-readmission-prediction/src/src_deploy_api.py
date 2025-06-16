```
"""
Flask API to serve readmission risk predictions.
Loads trained model and preprocessor for inference.
"""
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('../models/xgboost_model.pkl')
preprocessor = joblib.load('../models/preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict readmission risk."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = create_features(df)  # Apply same feature engineering as preprocessing
        X = preprocessor.transform(df)
        prob = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_features(df):
    """Replicate feature engineering for inference."""
    df['prior_admissions'] = df.get('prior_admissions', 0)  # Default if not provided
    df['comorbidity_index'] = df[['diabetes', 'hypertension', 'heart_disease']].sum(axis=1)
    return df

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```