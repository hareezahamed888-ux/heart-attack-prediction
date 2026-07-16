import argparse
from heart_attack_model import FEATURE_COLUMNS, load_model, predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict heart attack risk using a trained model."
    )
    parser.add_argument("--age", type=int, required=True, help="Age in years")
    parser.add_argument("--sex", type=int, choices=[0, 1], required=True, help="Sex: 1=male, 0=female")
    parser.add_argument("--cp", type=int, choices=range(0, 4), required=True, help="Chest pain type (0-3)")
    parser.add_argument("--trestbps", type=int, required=True, help="Resting blood pressure")
    parser.add_argument("--chol", type=int, required=True, help="Serum cholesterol in mg/dl")
    parser.add_argument("--fbs", type=int, choices=[0, 1], required=True, help="Fasting blood sugar > 120 mg/dl (1=yes, 0=no)")
    parser.add_argument("--restecg", type=int, choices=range(0, 3), required=True, help="Resting ECG results (0-2)")
    parser.add_argument("--thalach", type=int, required=True, help="Max heart rate achieved")
    parser.add_argument("--exang", type=int, choices=[0, 1], required=True, help="Exercise-induced angina (1=yes, 0=no)")
    parser.add_argument("--oldpeak", type=float, required=True, help="ST depression induced by exercise relative to rest")
    parser.add_argument("--slope", type=int, choices=range(0, 3), required=True, help="Slope of the peak exercise ST segment (0-2)")
    parser.add_argument("--ca", type=int, choices=range(0, 5), required=True, help="Number of major vessels colored by fluoroscopy (0-4)")
    parser.add_argument("--thal", type=int, choices=[1, 2, 3], required=True, help="Thalassemia: 1=normal, 2=fixed defect, 3=reversible defect")
    return parser.parse_args()


def main():
    args = parse_args()
    patient_info = [
        args.age,
        args.sex,
        args.cp,
        args.trestbps,
        args.chol,
        args.fbs,
        args.restecg,
        args.thalach,
        args.exang,
        args.oldpeak,
        args.slope,
        args.ca,
        args.thal,
    ]

    model = load_model()
    result = predict(patient_info, model=model)
    label = "Heart attack likely" if result["prediction"] == 1 else "Heart attack unlikely"

    print("\nHeart Attack Prediction")
    print("------------------------")
    print(f"Prediction: {label}")
    print(f"Probability of heart attack: {result['probability'] * 100:.1f}%")


if __name__ == "__main__":
    main()
