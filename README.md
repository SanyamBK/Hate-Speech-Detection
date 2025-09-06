# Hate Speech Detection Using Neural Networks

Project summary

This repository contains an end-to-end solution for detecting hate speech in text using neural network models. The project was developed for the CodeChef Weekend Dev Challenge 14: "DL Projects". It includes data cleaning, preprocessing, model training, and a saved model artifact for inference.

Contents

- `main.ipynb` — exploratory analysis and experiment notes.
- Part 1/
  - `main.py` — initial data loading and preprocessing for the raw dataset.
  - `hate_speech.csv` — original dataset sample.
- Part 2/
  - `main.py` — data cleaning and text preprocessing pipeline; produces `cleaned_hate_dataset.csv`.
  - `hate_dataset.csv`, `cleaned_hate_dataset.csv` — intermediate and cleaned datasets.
- Part 3/
  - `main.py` — final model training, evaluation, and inference utilities.
  - `hate_speech_model.pkl` — serialized trained model for inference.
  - `cleaned_hate_dataset.csv` — dataset used for final training.

How it works (high level)

1. Data cleaning and normalization: remove noise, handle missing values, and normalize text (lowercasing, punctuation removal).
2. Tokenization and vectorization: convert text to numeric representation using TF-IDF or embeddings depending on the script.
3. Model training: train a neural network classifier (architecture and hyperparameters are in `Part 3/main.py`).
4. Evaluation: standard classification metrics (accuracy, precision, recall, F1). The notebook and Part 3 script produce detailed reports.

Quick start

1. Create and activate a Python virtual environment and install dependencies from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Reproduce final training and evaluation:

```powershell
Set-Location -LiteralPath "Hate Speech Detection project"
python "Part 3\main.py"
```

3. Run inference using the saved model:

```python
from sklearn.externals import joblib
model = joblib.load('Part 3/hate_speech_model.pkl')
pred = model.predict(["sample input text"])
print(pred)
```

Notes

- The project includes both script and notebook versions of experiments; the notebook (`main.ipynb`) contains analysis and plots.
- For deployment, wrap the `predict()` call from `Part 3/main.py` in a Flask/FastAPI endpoint.

