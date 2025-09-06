# Hate Speech Detection

### Overview

This project implements a text-classification pipeline to categorize short social media posts (tweets) into one of three classes:

- Hate Speech (0)
- Offensive Language (1)
- Neither / Neutral (2)

This is a multiclass classification problem. The solution includes data exploration, cleaning and preprocessing, model training using a neural network, and a saved model artifact for inference.

### Contest

Developed for CodeChef Weekend Dev Challenge 14: "DL Projects" (attempted on 6 Sep 2025).

## Repository layout

- `main.ipynb` — exploratory analysis and experiment log.
- Part 1/
  - `main.py` — data loading and initial EDA.
  - `hate_speech.csv` — original raw dataset (columns: `tweet`, `class`).
- Part 2/
  - `main.py` — data cleaning and preprocessing pipeline; outputs `cleaned_hate_dataset.csv`.
  - `hate_dataset.csv` — intermediate dataset.
- Part 3/
  - `main.py` — model definition, training, evaluation, and inference utilities.
  - `cleaned_hate_dataset.csv` — final cleaned dataset used for training.
  - `hate_speech_model.pkl` — serialized trained model for deployment.

## Key steps

1. Data exploration (Part 1): inspect class balance, token distributions, and common tokens.
2. Cleaning & preprocessing (Part 2): normalize text, remove noise, tokenize, and vectorize (TF-IDF or embeddings).
3. Model training & evaluation (Part 3): train a neural classifier and evaluate using accuracy, precision/recall, F1, and confusion matrix.

### Part 1 — Data Exploration & Analysis

In Part 1 we perform an exploratory data analysis to understand the dataset before cleaning and modeling. The dataset (`hate_dataset.csv` / `hate_speech.csv`) contains two columns:

- `tweet`: raw tweet text
- `class`: label (0 = Hate Speech, 1 = Offensive Language, 2 = Neither/Neutral)

The EDA includes:

- Class distribution and imbalance checks
- Token length and distribution plots
- Frequent token and n-gram analysis per class

## Quick start

1. Create a virtual environment and install dependencies (from repo root):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run Part 1 (EDA):

```powershell
Set-Location -LiteralPath "Hate Speech Detection project\Part 1"
python main.py
```

3. Run final training & evaluation (Part 3):

```powershell
Set-Location -LiteralPath "Hate Speech Detection project\Part 3"
python main.py
```

### Inference example

```python
import joblib
model = joblib.load('Part 3/hate_speech_model.pkl')
text = "This is a sample tweet to classify"
pred = model.predict([text])
print(pred)
```
