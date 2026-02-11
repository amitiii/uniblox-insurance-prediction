# AI Collaboration Log: Insurance Enrollment Prediction

## 1. Project Scoping & Structure
* **Goal**: Establish a professional ML microservice folder structure.
* **AI Contribution**: Suggested the separation of concerns between `engine.py` (training) and `main.py` (inference).
* **Developer Decision**: Adopted the suggestion to ensure the project remains scalable and clean for evaluation.

## 2. Model Pipeline Engineering (`engine.py`)
* **Challenge**: Handling categorical string data (Gender, Region) and high-variance numeric data (Salary) consistently.
* **AI Collaboration**: Discussed the pros/cons of Label Encoding vs. One-Hot Encoding.
* **Result**: Implemented a **Scikit-Learn Pipeline** using `ColumnTransformer`. This ensures that `StandardScaler` and `OneHotEncoder` are saved within the `model.pkl`, preventing "data leakage" and simplifying the API logic.

## 3. API Development & Validation (`main.py`)
* **Challenge**: Ensuring the API is robust against malformed user input.
* **AI Collaboration**: Implemented **Pydantic schemas** for strict type-checking. 
* **Refinement**: AI helped troubleshoot the integration between the saved Pipeline and the FastAPI request body, ensuring that the JSON input is correctly converted into a Pandas DataFrame for the model.

## 4. Documentation & Presentation
* **Goal**: Create a professional GitHub repository.
* **AI Contribution**: Assisted in formatting the `README.md` and `report.md` using professional technical language.

## 5. AI Tools Used
* **Model**: Gemini 3 Flash.
* **Utility**: Code refactoring, debugging Pydantic validation errors, and drafting technical documentation.

---