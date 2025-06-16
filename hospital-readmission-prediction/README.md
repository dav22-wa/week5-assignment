
# Hospital Readmission Prediction
This repository contains code for the AI Development Workflow Assignment (AI for Software Engineering course). It implements a machine learning pipeline to predict patient readmission risk within 30 days of discharge, addressing the case study in Part 2 of the assignment.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/hospital-readmission-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `data/sample_data.csv` file exists (simulated dataset provided).

## Usage
- **Exploratory Analysis**: Run `notebooks/exploratory_analysis.ipynb` in Jupyter to analyze the dataset.
- **Preprocessing**: Run `python src/preprocess.py` to clean and transform data.
- **Training**: Run `python src/train_model.py` to train and evaluate the XGBoost model.
- **Deployment**: Run `python src/deploy_api.py` to start the Flask API.
  - Test the API with a POST request to `http://localhost:5000/predict`:
    ```json
    {
      "patient_id": "P001",
      "age": 65,
      "gender": "M",
      "diagnosis_code": "I10",
      "length_of_stay": 5,
      "diabetes": 1,
      "hypertension": 0,
      "heart_disease": 1,
      "admission_date": "2025-01-01"
    }
    

## Files
- `data/sample_data.csv`: Simulated dataset.
- `notebooks/exploratory_analysis.ipynb`: EDA for dataset insights.
- `src/preprocess.py`: Data preprocessing pipeline.
- `src/train_model.py`: Model training and evaluation.
- `src/deploy_api.py`: Flask API for predictions.
- `docs/ai_workflow_diagram.mmd`: Mermaid.js flowchart of AI Development Workflow.
- `docs/ai_workflow_diagram.png`: Rendered flowchart image.

## Notes
- The model uses XGBoost for high accuracy and interpretability.
- HIPAA compliance is addressed via encryption and access controls (see PDF report).
- The flowchart visualizes the CRISP-DM-inspired AI Development Workflow.

## License
MIT License (see LICENSE file).
```
