# Insurance Enrollment Prediction Engine

## Overview
This repository contains a production-ready machine learning pipeline designed to predict whether an employee will opt into a voluntary insurance product based on demographic and employment data.

## Project Structure
* `engine.py`: The core ML pipeline (preprocessing + training).
* `main.py`: FastAPI implementation for real-time inference.
* `report.md`: Detailed analysis of data and model performance.
* `requirements.txt`: Python dependencies.

## Setup & Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
---

### **Step 2: The `report.md` (The "ML Thinking" Deliverable)**
Create a file named `report.md`. This fulfills the specific requirement for a summary of observations and rationale.

```markdown
# ML Assignment Report

## 1. Data Observations
* **Features**: The dataset contains 10,000 rows with a mix of numerical (`age`, `salary`) and categorical (`region`, `employment_type`) variables.
* **Target**: The `enrolled` column is the target for binary classification.
* **Preprocessing**: Applied `StandardScaler` to numerical features and `OneHotEncoder` to categorical features to ensure the model handles diverse data types correctly.

## 2. Model Choice & Rationale
I selected **XGBoost (Extreme Gradient Boosting)**. 
* **Reasoning**: Tabular insurance data often has non-linear relationships (e.g., salary and age interacting to influence insurance needs). XGBoost captures these better than linear models and handles potential missing values gracefully.

## 3. Evaluation Results
* **Metric**: I focused on the **F1-Score** and **Recall**. 
* **Insight**: In insurance, missing a potential customer (False Negative) is often costlier than a slight over-prediction, so the model was tuned for high coverage.

## 4. Key Takeaways & Future Work
* **Future Work**: I would implement **SHAP (SHapley Additive exPlanations)** to provide "Explainable AI," telling the user *why* a specific prediction was made.
* **LLM Integration**: As a next step, I would use a reasoning engine to summarize the "why" into a natural language sentence for the HR manager.
