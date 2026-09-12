import os
from heart_attack_model import (
    load_data,
    save_model,
    save_sample_data,
    train_model,
    MODEL_PATH,
)

DATA_PATH = os.path.join("data", "heart_attack_data.csv")


def main():
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Generating synthetic data at data/heart_attack_data.csv...")
        save_sample_data(DATA_PATH, n_samples=400)
    else:
        print(f"Loading dataset from {DATA_PATH}")

    df = load_data(DATA_PATH)
    model = train_model(df)
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
