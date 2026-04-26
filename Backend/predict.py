# =============================================================================
# predict.py — PREDICTION LOGIC
# =============================================================================
# This file is the brain of the backend
# It loads the trained .pkl model files and handles all prediction logic
# main.py calls functions from this file when it receives a request from React
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import os

# =============================================================================
# SECTION 1 — LOAD THE MODELS
# =============================================================================
# We load all 4 models and their scalers once when the server starts
# This way they are ready in memory and predictions are instant
# Loading from disk every time a request comes in would be very slow

# Base path to the Models folder
# os.path.dirname(__file__) gets the directory of this file (Backend/)
# then we go one level up (..) to reach the project root, then into Models/
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'Models')

# Load Masters models
# masters_regression.pkl: XGBoost model that predicts admission_probability (0-1)
masters_reg = joblib.load(os.path.join(MODELS_PATH, 'masters_regression.pkl'))
# masters_classification.pkl: XGBoost model that predicts admitted_binary (0 or 1)
masters_clf = joblib.load(os.path.join(MODELS_PATH, 'masters_classification.pkl'))
# masters_scaler.pkl: the fitted MinMaxScaler — scales user inputs the same way training data was scaled
masters_scaler = joblib.load(os.path.join(MODELS_PATH, 'masters_scaler.pkl'))
# masters_university_means.pkl: dict of mean admission probability per university (for target encoding)
masters_university_means = joblib.load(os.path.join(MODELS_PATH, 'masters_university_means.pkl'))

# Load PhD models
phd_reg = joblib.load(os.path.join(MODELS_PATH, 'phd_regression.pkl'))
phd_clf = joblib.load(os.path.join(MODELS_PATH, 'phd_classification.pkl'))
phd_scaler = joblib.load(os.path.join(MODELS_PATH, 'phd_scaler.pkl'))
phd_university_means = joblib.load(os.path.join(MODELS_PATH, 'phd_university_means.pkl'))

print("All models loaded successfully")


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING FUNCTION
# =============================================================================
# Same feature engineering we did in model.py
# We MUST apply the exact same transformations to user input
# otherwise the model gets different shaped data than it was trained on

def engineer_features(data):
    """
    Creates the same engineered features we created during training.
    Must be applied to user input before prediction.
    """

    # academic_strength: combined GRE and GPA signal
    data['academic_strength'] = data['gre_total'] * data['undergrad_gpa']

    # application_strength: combined SOP and LOR quality signal
    data['application_strength'] = (
        data['sop_strength'] +
        (data['lor_avg_strength'] * 2) +
        data['lor_count']
    )

    # research_power: combined research credibility signal
    # publications weighted 3x, conference papers 2x
    data['research_power'] = (
        data['research_experience_years'] +
        (data['publications_count'] * 3) +
        (data['conference_papers'] * 2)
    )

    # overall_profile_score: single holistic applicant score
    data['overall_profile_score'] = (
        (data['academic_strength'] / 1000) +
        data['application_strength'] +
        data['research_power']
    )

    return data


# =============================================================================
# SECTION 3 — PREPROCESSING FUNCTION
# =============================================================================
# Same preprocessing we did in model.py but adapted for a single user input
# instead of a full dataframe of 150k rows
# The key difference: we use the SAVED scaler (fit=False) instead of fitting a new one
# This ensures inputs are scaled exactly the same way as training data

def preprocess_input(data, scaler, university_means, feature_columns):
    """
    Preprocesses a single user input dictionary into the format the model expects.

    Parameters:
        data: dict of user inputs from the React form
        scaler: the fitted MinMaxScaler loaded from .pkl
        university_means: the university encoding dict loaded from .pkl
        feature_columns: the exact list of columns the model was trained on

    Returns:
        X: a single row dataframe ready for model.predict()
    """

    # Convert the user input dict into a single row dataframe
    df = pd.DataFrame([data])

    # -------------------------------------------------------------------------
    # HANDLE MISSING OPTIONAL SCORES
    # -------------------------------------------------------------------------
    # Add flag columns for optional test scores
    # If the user didnt submit a score, it comes in as 0 and flag is set to 0
    df['has_gmat'] = (df['gmat_total'] > 0).astype(int)
    df['has_toefl'] = (df['toefl_score'] > 0).astype(int)
    df['has_ielts'] = (df['ielts_score'] > 0).astype(int)

    # Fill any missing values with 0 just in case
    df['gmat_total'] = df.get('gmat_total', 0)
    df['gmat_verbal'] = df.get('gmat_verbal', 0)
    df['gmat_quant'] = df.get('gmat_quant', 0)
    df['toefl_score'] = df.get('toefl_score', 0)
    df['ielts_score'] = df.get('ielts_score', 0)

    # -------------------------------------------------------------------------
    # TARGET ENCODE THE UNIVERSITY
    # -------------------------------------------------------------------------
    # Replace university name with its mean admission probability
    # using the same dictionary we saved during training
    overall_mean = np.mean(list(university_means.values()))
    df['applied_university'] = df['applied_university'].map(university_means).fillna(overall_mean)

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    df = engineer_features(df)

    # -------------------------------------------------------------------------
    # ONE-HOT ENCODE CATEGORICAL COLUMNS
    # -------------------------------------------------------------------------
    categorical_columns = [
        'university_region', 'university_country', 'university_qs_tier',
        'program_name', 'program_field', 'funding_type', 'undergrad_university_tier'
    ]
    categorical_columns = [c for c in categorical_columns if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # -------------------------------------------------------------------------
    # ALIGN COLUMNS WITH TRAINING DATA
    # -------------------------------------------------------------------------
    # This is critical — the model expects EXACTLY the same columns it was trained on
    # in EXACTLY the same order
    # If a user selects a program that creates a column the model hasnt seen, we drop it
    # If a column the model expects is missing (e.g. a country not selected), we add it as 0

    # Add any missing columns with value 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model was trained on, in the right order
    df = df[feature_columns]

    # -------------------------------------------------------------------------
    # NORMALIZE NUMERICAL COLUMNS
    # -------------------------------------------------------------------------
    numerical_columns = [
        'undergrad_gpa', 'gre_total', 'gre_verbal', 'gre_quantitative',
        'gre_analytical_writing', 'gmat_total', 'gmat_verbal', 'gmat_quant',
        'toefl_score', 'ielts_score', 'sop_strength', 'sop_word_count',
        'lor_count', 'lor_avg_strength', 'lor_from_professor', 'lor_from_industry',
        'research_experience_years', 'publications_count', 'conference_papers',
        'work_experience_years', 'internships_count', 'work_industry_relevance',
        'applied_university', 'academic_strength', 'application_strength',
        'research_power', 'overall_profile_score'
    ]
    numerical_columns = [c for c in numerical_columns if c in df.columns]

    # Use transform (NOT fit_transform) — we never refit the scaler on new data
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df


# =============================================================================
# SECTION 4 — GET FEATURE COLUMNS
# =============================================================================
# We need the exact list of feature columns the model was trained on
# We extract this from the model itself so it always stays in sync

def get_feature_columns(model):
    """
    Extracts the feature column names from a trained XGBoost model.
    XGBoost stores these internally after training.
    """
    return model.get_booster().feature_names


# =============================================================================
# SECTION 5 — MAIN PREDICTION FUNCTION
# =============================================================================
# This is the function that main.py calls when React sends a prediction request
# It takes the user input, preprocesses it, runs both models, and returns results

def predict(user_input: dict, degree_type: str):
    """
    Runs admission prediction for a given user profile.

    Parameters:
        user_input: dict of all user inputs from the React form
        degree_type: 'masters' or 'phd'

    Returns:
        dict containing:
            - admission_probability: float (0-1) e.g. 0.734
            - admission_percentage: float (0-100) e.g. 73.4
            - admitted: bool (True = likely admitted, False = likely rejected)
            - verdict: string e.g. 'Likely Admitted' or 'Likely Rejected'
            - confidence: string e.g. 'High' / 'Medium' / 'Low'
    """

    # Select the right models based on degree type
    if degree_type == 'masters':
        reg_model = masters_reg
        clf_model = masters_clf
        scaler = masters_scaler
        university_means = masters_university_means
    elif degree_type == 'phd':
        reg_model = phd_reg
        clf_model = phd_clf
        scaler = phd_scaler
        university_means = phd_university_means
    else:
        raise ValueError(f"Invalid degree_type: {degree_type}. Must be 'masters' or 'phd'.")

    # Get the exact feature columns this model was trained on
    feature_columns = get_feature_columns(reg_model)

    # Preprocess the user input into the format the model expects
    X = preprocess_input(user_input, scaler, university_means, feature_columns)

    # -------------------------------------------------------------------------
    # RUN REGRESSION MODEL
    # -------------------------------------------------------------------------
    # Predicts a continuous probability between 0 and 1
    # e.g. 0.734 means 73.4% chance of admission
    admission_probability = float(reg_model.predict(X)[0])

    # Clip to valid range just in case of floating point edge cases
    admission_probability = max(0.0, min(1.0, admission_probability))

    # -------------------------------------------------------------------------
    # RUN CLASSIFICATION MODEL
    # -------------------------------------------------------------------------
    # Predicts 0 (rejected) or 1 (admitted)
    admitted_binary = int(clf_model.predict(X)[0])

    # Also get the probability confidence of the classification decision
    # predict_proba returns [prob_rejected, prob_admitted]
    clf_proba = clf_model.predict_proba(X)[0]
    clf_confidence = float(max(clf_proba))  # how confident the model is in its decision

    # -------------------------------------------------------------------------
    # DETERMINE VERDICT AND CONFIDENCE LABEL
    # -------------------------------------------------------------------------
    admitted = admitted_binary == 1
    verdict = "Likely Admitted" if admitted else "Likely Rejected"

    # Confidence label based on how sure the classification model is
    if clf_confidence >= 0.85:
        confidence = "High"
    elif clf_confidence >= 0.65:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -------------------------------------------------------------------------
    # RETURN RESULTS
    # -------------------------------------------------------------------------
    return {
        "admission_probability": round(admission_probability, 4),
        "admission_percentage": round(admission_probability * 100, 2),
        "admitted": admitted,
        "verdict": verdict,
        "confidence": confidence,
        "degree_type": degree_type
    }