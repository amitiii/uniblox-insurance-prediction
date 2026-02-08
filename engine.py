import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def run_pipeline():
    df = pd.read_csv('employee_data.csv')
    
    X = df.drop(['employee_id', 'enrolled'], axis=1)
    y = df['enrolled']

    num_cols = ['age', 'salary', 'tenure_years']
    cat_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Training the Pro ML Model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, 'model.pkl')
    print("Model saved as model.pkl")

if __name__ == "__main__":
    run_pipeline()