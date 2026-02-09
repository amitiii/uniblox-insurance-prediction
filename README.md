# Insurance Enrollment Prediction API

A professional Machine Learning microservice built with **XGBoost** and **FastAPI**. This project predicts whether an employee will enroll in an insurance plan based on demographic and employment metrics.

## 🚀 Project Overview
This repository contains a complete end-to-end ML pipeline:
1. **Model Training**: An XGBoost classifier trained on employee metadata (Age, Salary, Tenure, etc.).
2. **REST API**: A high-performance FastAPI service for real-time inference.
3. **Data Validation**: Strict type-checking using Pydantic schemas.
4. **Monitoring**: Integrated health-check endpoints and professional logging.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/amitiii/uniblox-insurance-prediction.git](https://github.com/amitiii/uniblox-insurance-prediction.git)
   cd uniblox-insurance-prediction
## 🛠️ Tech Stack

* **Language**: Python 3.11+
* **ML Framework**: Scikit-Learn, XGBoost
* **API Framework**: FastAPI, Uvicorn
* **Data Handling**: Pandas, Numpy

## 📦 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/amitiii/uniblox-insurance-prediction.git
cd uniblox-insurance-prediction

```


2. **Set up a Virtual Environment**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



## 📈 Usage

### 1. Training the Model

To process the data and generate the `model.pkl` file, run:

```bash
python engine.py

```

### 2. Running the API

To start the local inference server:

```bash
python -m uvicorn main:app --reload

```

## 🧪 Testing the API

Once the server is running, navigate to:
👉 **`http://127.0.0.1:8000/docs`**

You can use the built-in **Swagger UI** to send test requests to the `/predict` endpoint.
![API Screenshot](screenshotml.png)
![API Screenshot](screenshotml2.png)

## 📊 Model Performance

The current iteration of the model utilizes an XGBoost classifier, achieving:

* **Accuracy**: 100% (on the provided evaluation set)
* **Inference Latency**: < 20ms

*For a deeper dive into the data rationale and technical choices, please refer to the [report.md](https://github.com/amitiii/uniblox-insurance-prediction/blob/main/report.md).*

---

