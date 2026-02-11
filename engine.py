import pandas as pd
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    try:
        df = pd.read_csv('employee_data.csv')
        logging.info(f"Dataset loaded successfully with {df.shape[0]} rows.")
        
        X = df.drop(['employee_id', 'enrolled'], axis=1)
        y = df['enrolled']
        
        num_cols = ['age', 'salary', 'tenure_years']
        cat_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=42,
                eval_metric='logloss'
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        logging.info("Training the XGBoost Pipeline...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        print("\n--- Model Performance Report ---")
        print(classification_report(y_test, preds))

        joblib.dump(model, 'model.pkl')
        logging.info("Pipeline saved successfully as model.pkl")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_pipeline()
    