import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = os.path.join("model", "heart_attack_model.pkl")
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "heart_attack"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    return df


def generate_synthetic_dataset(n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    age = rng.integers(29, 78, size=n_samples)
    sex = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    cp = rng.integers(0, 4, size=n_samples)
    trestbps = rng.integers(94, 201, size=n_samples)
    chol = rng.integers(126, 565, size=n_samples)
    fbs = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    restecg = rng.integers(0, 3, size=n_samples)
    thalach = rng.integers(71, 203, size=n_samples)
    exang = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    oldpeak = np.round(rng.uniform(0.0, 6.2, size=n_samples), 1)
    slope = rng.integers(0, 3, size=n_samples)
    ca = rng.integers(0, 5, size=n_samples)
    thal = rng.choice([1, 2, 3], size=n_samples, p=[0.3, 0.35, 0.35])

    risk_score = (
        0.03 * (age - 45)
        + 0.9 * sex
        + 0.7 * (cp > 1)
        + 0.004 * (trestbps - 120)
        + 0.003 * (chol - 200)
        + 0.8 * fbs
        + 0.6 * (restecg > 0)
        + 0.01 * (thalach - 140)
        + 0.9 * exang
        + 0.5 * oldpeak
        + 0.4 * (slope == 0)
        + 0.2 * ca
        + 0.5 * (thal == 3)
    )
    probability = 1 / (1 + np.exp(-(risk_score - 2.2)))
    heart_attack = (probability > rng.random(n_samples)).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            TARGET_COLUMN: heart_attack,
        }
    )


def save_sample_data(path, n_samples=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_synthetic_dataset(n_samples=n_samples)
    df.to_csv(path, index=False)
    return df


def train_model(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Training complete")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Classification report:")
    print(classification_report(y_test, predictions, digits=4))
    return model


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first to generate the model."
        )
    return joblib.load(path)


def predict(patient_info, model=None):
    if model is None:
        model = load_model()
    patient_df = pd.DataFrame([patient_info], columns=FEATURE_COLUMNS)
    probability = model.predict_proba(patient_df)[0][1]
    prediction = int(probability >= 0.5)
    return {"prediction": prediction, "probability": float(probability)}
