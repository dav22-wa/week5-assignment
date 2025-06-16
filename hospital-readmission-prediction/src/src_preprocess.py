
"""
Preprocessing pipeline for hospital readmission prediction.
Handles missing data, feature engineering, normalization, and encoding.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def create_features(df):
    """Engineer features for readmission prediction."""
    # Calculate number of prior admissions (simulated as grouped count)
    df['prior_admissions'] = df.groupby('patient_id')['admission_date'].transform('count') - 1
    # Comorbidity index: sum of chronic conditions
    chronic_cols = ['diabetes', 'hypertension', 'heart_disease']
    df['comorbidity_index'] = df[chronic_cols].sum(axis=1)
    return df

def preprocess_pipeline():
    """Build preprocessing pipeline for numerical and categorical features."""
    num_features = ['age', 'length_of_stay', 'prior_admissions', 'comorbidity_index']
    cat_features = ['diagnosis_code', 'gender']
    
    # Numerical pipeline: Impute missing values and normalize
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing values and encode
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )
    return preprocessor

def main():
    """Execute preprocessing pipeline and save processed data."""
    df = load_data('../data/sample_data.csv')
    df = create_features(df)
    preprocessor = preprocess_pipeline()
    X = preprocessor.fit_transform(df.drop('readmission', axis=1))
    y = df['readmission']
    
    # Save processed data
    pd.DataFrame(X).to_csv('../data/processed_data.csv', index=False)
    y.to_csv('../data/labels.csv', index=False)
    
    # Save preprocessor
    joblib.dump(preprocessor, '../models/preprocessor.pkl')

if __name__ == '__main__':
    main()
