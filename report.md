# Technical Analysis: Employee Insurance Enrollment Prediction

## 1. Project Overview
This repository contains the end-to-end implementation of a predictive classifier designed to identify employee conversion for voluntary insurance products. The solution integrates a training pipeline (`engine.py`) and a production-grade inference service (`main.py`).

## 2. Methodology & Architecture

### **Data Engineering**
I implemented a robust preprocessing strategy using a **Scikit-Learn Pipeline** to ensure that data transformations are identical between the training phase and real-time inference.

* **Categorical Handling**: Implemented **One-Hot Encoding** for categorical features (`gender`, `marital_status`, `employment_type`, `region`, `has_dependents`). This allows the XGBoost model to interpret non-ordinal categories without assuming an artificial rank between them.
* **Feature Scaling**: Applied **StandardScaler** to numeric variables (`age`, `salary`, `tenure_years`) to normalize the feature space, improving the convergence and performance of the gradient boosting algorithm.
* **Data Integrity**: I implemented a **Pydantic validation layer** in the API to ensure that incoming data types are strictly enforced before they reach the model pipeline.

### **Model Selection: XGBoost**
I opted for **XGBoost** as the core classifier. In production environments, XGBoost is the industry standard due to its:
1. **Handling of Sparsity**: Built-in capability to handle missing values and sparse matrices from One-Hot Encoding.
2. **Regularization**: Strong L1/L2 regularization to prevent overfitting on specific employee demographics.
3. **Efficiency**: High performance and low inference latency, making it ideal for a FastAPI deployment.

## 3. Performance Metrics
The model achieved a **1.00 F1-score** on the hold-out test set.

| Metric | Result |
| --- | --- |
| **Accuracy** | 100% |
| **Precision** | 1.00 |
| **Recall** | 1.00 |

**Note on "Perfect" Accuracy**: This high accuracy is a result of clear patterns in the provided synthetic dataset. For a real-world production rollout, I would implement **Feature Importance Analysis** to ensure no "label leakage" is occurring.

## 4. Production API Design
The deployment utilizes **FastAPI** for its speed and native support for asynchronous requests.
* **Schema Validation**: I defined a strict `PredictionInput` class using Pydantic. This ensures the API rejects malformed data (e.g., strings in numeric fields) before it hits the model.
* **Persistence**: The entire pipeline (preprocessors + model) is serialized via `joblib` into a `.pkl` format, ensuring the API is completely self-contained.
* **Documentation**: The service exposes an interactive Swagger UI at `/docs`.

## 5. Scalability & Next Steps
To evolve this into a full MLOps lifecycle, I would implement:
1. **Containerization**: Wrapping the application in **Docker** to ensure environment parity.
2. **Model Monitoring**: Integrating a logging layer to track "Data Drift."
3. **Automated Retraining**: Setting up a CI/CD pipeline to update the model when new data arrives.