# 🚀 ARCH.AI — Student Admission Predictor

A machine learning system that predicts graduate admission outcomes for Masters and PhD applicants. Given an applicant's academic profile, it estimates both the **probability of admission** and a **binary admit/reject prediction**, served through an interactive Streamlit dashboard.

Built on XGBoost, trained on **185,336 applicant records**, with degree-specific models for Masters and PhD applicants.

---

## Overview

Applying to grad school means juggling GRE scores, GPA, research output, SOPs, and LORs with no clear sense of where you actually stand. ARCH.AI takes an applicant's profile — test scores, GPA, research history, application materials, and target university — and returns an admission probability, a gauge visualization, and a status readout (high / moderate / low chance).

Under the hood, two separate pairs of models are trained — one pair for **Masters** applicants and one for **PhD** applicants — since the two have very different admission dynamics (PhD leans heavily on research output; Masters leans more on academic scores).

---

## Features

- 🎯 **Dual prediction**: a regression model outputs a continuous admission probability (0–1), and a classification model outputs a binary admit/reject call
- 🎓 **Degree-specific models**: separate Masters and PhD models, since the two are predicted by very different signals
- 🧠 **Engineered features**: four custom features (academic strength, application strength, research power, overall profile score) built specifically to capture combined signal that raw columns miss
- 🏫 **University target encoding**: 186 universities encoded by their historical admission rate rather than exploded into one-hot columns
- 📊 **Interactive dashboard**: a dark-themed Streamlit app with sliders for GPA/GRE/research inputs, a live probability gauge, and instant feedback

---

## Tech Stack

| Layer | Technology |
|---|---|
| Modeling | XGBoost (`XGBRegressor`, `XGBClassifier`) |
| Data processing | pandas, NumPy, scikit-learn (`MinMaxScaler`, `train_test_split`) |
| Model persistence | joblib |
| Frontend / App | Streamlit, Plotly (gauge chart) |
| Dataset | 185,336 rows × 41 columns, `ADMISSION_CALCULATOR_AI_DATA_SET.csv` |

---

## Model Performance

Final production models (XGBoost + engineered features):

| | Masters | PhD |
|---|---|---|
| **Trained on** | 152,515 records | 32,821 records |
| **Regression R²** | 0.9220 | 0.9238 |
| **Regression MAE** | 0.0326 (±3.26%) | 0.0325 (±3.25%) |
| **Classification Accuracy** | 95.38% | 95.45% |
| **Classification F1** | 88.75% | 89.19% |

These numbers are the result of three iterations — see [Model Development](#model-development) below.

---

## Project Structure

```
Student-Admission-Ai/
├── Data/
│   ├── ADMISSION_CALCULATOR_AI_DATA_SET.csv   # 185,336-row training dataset
│   └── Admission Calculator AI Proposal.pdf   # Original project proposal
├── Models/
│   ├── masters_regression.pkl                 # XGBoost regressor — Masters
│   ├── masters_classification.pkl             # XGBoost classifier — Masters
│   ├── masters_scaler.pkl                     # Fitted MinMaxScaler — Masters
│   ├── masters_university_means.pkl           # University target-encoding map — Masters
│   ├── phd_regression.pkl                     # XGBoost regressor — PhD
│   ├── phd_classification.pkl                 # XGBoost classifier — PhD
│   ├── phd_scaler.pkl                         # Fitted MinMaxScaler — PhD
│   ├── phd_university_means.pkl               # University target-encoding map — PhD
│   └── feature_list.pkl                       # Ordered feature names the model expects
├── app.py                                     # Streamlit dashboard (prediction UI)
├── model.py                                   # Full training pipeline (preprocessing, feature engineering, training, evaluation, saving)
├── get_metadata.py                            # Extracts feature ordering from a trained model
├── model_iterations.txt                       # Full experiment log across 3 model iterations
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Student-Admission-AI/Student-Admission-Ai.git
cd Student-Admission-Ai
pip install streamlit pandas joblib plotly xgboost scikit-learn numpy
```

> `requirements.txt` is currently empty — the command above installs what the code actually imports. Consider running `pip freeze > requirements.txt` after installing to lock versions.

### Running the app

The trained models are already committed to `Models/`, so you can launch the dashboard directly:

```bash
streamlit run app.py
```

This opens the ARCH.AI dashboard in your browser, where you can enter GPA, GRE, research history, and a target university to get a live admission probability.

### Retraining the models (optional)

If you want to retrain from scratch (e.g. after updating the dataset):

```bash
python model.py
```

This reads `Data/ADMISSION_CALCULATOR_AI_DATA_SET.csv`, cleans it, engineers features, trains both Masters and PhD model pairs, prints evaluation metrics for each, and overwrites the `.pkl` files in `Models/`.

If you retrain and the feature set changes, regenerate the feature list the app depends on:

```bash
python get_metadata.py
```

---

## How It Works

**1. Data cleaning** — Invalid records are filtered out (GRE outside 260–340, GPA outside 2.0–4.0, implausible TOEFL scores, duplicate applicant IDs), and MBA applicants are folded into the Masters category.

**2. Feature engineering** — Four custom features are derived from the raw columns:
- `academic_strength` = GRE total × GPA
- `application_strength` = SOP strength + (LOR avg strength × 2) + LOR count
- `research_power` = research years + (publications × 3) + (conference papers × 2)
- `overall_profile_score` = a normalized combination of all three above

**3. Preprocessing** — Missing optional test scores (GMAT/TOEFL/IELTS) are filled with 0 alongside a "submitted or not" flag column, universities are target-encoded by their historical mean admission probability, categorical fields are one-hot encoded, and everything numeric is scaled to 0–1 with `MinMaxScaler`.

**4. Training** — Two XGBoost models (a regressor and a classifier) are trained separately for Masters and PhD applicants, using an 80/20 train-test split.

**5. Serving** — `app.py` loads the saved models and scaler, builds a single-row input from the sidebar controls, applies the same preprocessing/scaling used in training, and displays the predicted probability on a gauge.

---

## Model Development

Three iterations were run before settling on the final models — full reasoning and metrics for each are logged in [`model_iterations.txt`](./model_iterations.txt):

1. **Random Forest (baseline)** — 100 trees, ~91.5% R², ~95% classification accuracy
2. **XGBoost** — swapped in gradient boosting, improved every metric with a two-line change
3. **XGBoost + feature engineering (final)** — added the four engineered features above; helped PhD models more than Masters, since research signal is more predictive for PhD admissions

Hyperparameter tuning via grid search was deliberately skipped — estimated at ~10 hours of compute for an expected 0.5–1% gain, which wasn't worth it against the project deadline. This trade-off is documented in detail in the iterations log.

---

## Team

Built as a collaborative project for FAST-NUCES coursework.

> ✏️ *Add contributor names/GitHub links here.*

---

## License

> ✏️ *No license file is currently present. Add one (MIT is a common default for student/academic projects) if you'd like this to be explicitly reusable.*