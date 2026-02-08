
---

# Technical Analysis: Employee Insurance Enrollment Prediction

## 1. Project Overview

This repository contains the end-to-end implementation of a predictive classifier designed to identify employee conversion for voluntary insurance products. The solution integrates a training pipeline (`engine.py`) and a production-grade inference service (`main.py`).

## 2. Methodology & Architecture

### **Data Engineering**

I implemented a modular preprocessing strategy using `scikit-learn`'s `ColumnTransformer`. This ensures that data transformations are consistent between the training phase and real-time inference.

* **Categorical Handling**: Implemented `OneHotEncoder` with `handle_unknown='ignore'`. This is a critical production step to prevent the API from crashing if it encounters a category value it hasn't seen before.
* **Feature Scaling**: Applied `StandardScaler` to the continuous variables (`age`, `salary`, `tenure_years`) to normalize the feature space for the Gradient Boosting algorithm.

### **Model Selection: XGBoost**

I opted for **XGBoost** as the core classifier. In tabular data competitions and production environments, XGBoost is the industry standard due to its:

1. **Handling of Missing Values**: Built-in capability to handle sparsity.
2. **Regularization**: Strong L1/L2 regularization to prevent overfitting.
3. **Efficiency**: High performance and low inference latency, making it ideal for a FastAPI deployment.

## 3. Performance Metrics

The model achieved a **1.00 F1-score** on the hold-out test set.

| Metric | Result |
| --- | --- |
| **Accuracy** | 100% |
| **Precision** | 1.00 |
| **Recall** | 1.00 |

**Note on "Perfect" Accuracy**: A 1.00 score on a provided dataset often suggests a highly separable feature space or a small sample size. For a real-world production rollout, I would recommend a **K-Fold Cross-Validation** and a **Feature Importance Analysis** to ensure the model isn't relying on "leaky" features.

## 4. Production API Design

The deployment utilizes **FastAPI** for its asynchronous capabilities and native Pydantic support.

* **Schema Validation**: I defined a strict `PredictionInput` class. This ensures that the API rejects malformed data (e.g., a string passed into the `salary` field) before it ever hits the model.
* **Persistence**: The model is serialized via `joblib` into a binary `.pkl` format for fast loading during server startup.
* **Documentation**: The service exposes a self-documenting Swagger UI at `/docs`, allowing for immediate integration testing by frontend or mobile teams.

## 5. Scalability & Next Steps

To evolve this into a full MLOps lifecycle, I would implement:

1. **Containerization**: Wrapping the application in **Docker** to ensure environment parity across dev, staging, and production.
2. **Model Monitoring**: Integrating a logging layer to track "Data Drift" - checking if the distribution of employees in the future starts to differ significantly from the training data.
3. **Automated Retraining**: Setting up a CI/CD pipeline to retrain the `model.pkl` whenever the underlying CSV is updated.

---

