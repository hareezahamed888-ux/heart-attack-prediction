# Heart Attack Prediction System

This project provides a simple heart attack prediction system using medical and health information. It includes data preparation, model training, and a command-line prediction interface.

## What it does
- trains a machine learning model on heart attack risk features
- saves the trained model to `model/heart_attack_model.pkl`
- predicts patient heart attack risk from medical info

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Predict using the saved model:
   ```bash
   python predict.py --age 55 --sex 1 --cp 2 --trestbps 130 --chol 250 --fbs 0 --restecg 1 --thalach 150 --exang 0 --oldpeak 1.5 --slope 2 --ca 0 --thal 2
   ```

## Data
- `data/heart_attack_data.csv` contains sample patient records.
- If no dataset exists, `train.py` will generate a synthetic dataset automatically.

## Features
- `age`: patient age in years
- `sex`: 1 = male, 0 = female
- `cp`: chest pain type (0-3)
- `trestbps`: resting blood pressure
- `chol`: serum cholesterol in mg/dl
- `fbs`: fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: resting electrocardiographic results (0-2)
- `thalach`: maximum heart rate achieved
- `exang`: exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: slope of peak exercise ST segment (0-2)
- `ca`: number of major vessels colored by fluoroscopy (0-4)
- `thal`: thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
- `heart_attack`: target label (0 = no, 1 = yes)
