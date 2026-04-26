# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
# pandas: reads the CSV and lets us manipulate data like a spreadsheet
import pandas as pd

# numpy: handles mathematical operations, works under the hood with pandas and sklearn
import numpy as np

# RandomForestRegressor: the regression model that predicts admission_probability (0-1)
# RandomForestClassifier: the classification model that predicts admitted_binary (0 or 1)
# We keep these imported as a backup/reference but we are now using XGBoost instead
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# XGBRegressor: XGBoost regression model — predicts admission_probability (0-1)
# XGBClassifier: XGBoost classification model — predicts admitted_binary (0 or 1)
# XGBoost (Extreme Gradient Boosting) is an upgrade over Random Forest
# Instead of building 200 independent trees and averaging them (Random Forest),
# XGBoost builds trees SEQUENTIALLY — each new tree tries to fix the mistakes of the previous one
# This makes it more accurate, faster, and better at handling complex patterns in data
# It is the most widely used ML algorithm in competitions and real world applications
from xgboost import XGBRegressor, XGBClassifier

# train_test_split: splits data into 80% training and 20% testing
# the model learns from training data and gets evaluated on test data it has never seen
from sklearn.model_selection import train_test_split

# MinMaxScaler: normalizes numerical columns to 0-1 range
# this makes sure the model treats all features equally regardless of their original scale
# e.g. GRE (260-340) and GPA (2.0-4.0) get brought to the same 0-1 range
from sklearn.preprocessing import MinMaxScaler

# Regression metrics:
# mean_absolute_error (MAE): average difference between predicted and actual probability
# mean_squared_error (MSE): used to calculate RMSE, penalizes big errors more heavily
# r2_score (R²): how much of the variation in admission probability the model explains (closer to 1 is better)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    # Classification metrics:
    # accuracy_score: what % of applicants were correctly classified
    # precision_score: of those predicted admitted, how many actually were
    # recall_score: of those actually admitted, how many did the model catch
    # f1_score: balance between precision and recall
    # confusion_matrix: table showing true positives, false positives, true negatives, false negatives
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# joblib: saves and loads trained models as .pkl files
# .pkl files are frozen snapshots of the trained model — all its findings and learned patterns
# we save once after training, then the app loads them instantly without retraining
import joblib

# os: lets us create folders and handle file paths
# we use it to make sure the Models folder exists before saving .pkl files
import os


# =============================================================================
# SECTION 2 — LOAD THE DATA
# =============================================================================

# Read the CSV file into a pandas dataframe
# df now holds all 185,336 rows and 41 columns of raw applicant data
df = pd.read_csv('Data/ADMISSION_CALCULATOR_AI_DATA_SET.csv')

# Combine MBA into Masters (MS) since MBA is also a masters level degree
# replace() finds every row where degree_type is 'MBA' and changes it to 'MS'
df['degree_type'] = df['degree_type'].replace('MBA', 'MS')

# Split into two separate dataframes — one for Masters, one for PhD
# df[df['degree_type'] == 'MS'] filters only rows where degree_type is MS
# .copy() creates a completely independent copy so changes dont accidentally affect the original df
masters_df = df[df['degree_type'] == 'MS'].copy()
phd_df = df[df['degree_type'] == 'PhD'].copy()

# Print how many records are in each split to verify it worked
print(f"Masters records: {len(masters_df)}")
print(f"PhD records: {len(phd_df)}")


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING FUNCTION
# =============================================================================
# Feature engineering means creating NEW columns from existing ones
# These new columns give the model extra signal — combined patterns it might miss
# on its own when looking at features individually
# We do this BEFORE preprocessing so the new features also get normalized

def engineer_features(data, degree_label):
    """
    Creates new combined features from existing columns to improve model accuracy.

    Parameters:
        data: the raw dataframe to add features to
        degree_label: 'masters' or 'phd' — some features are degree specific

    Returns:
        data: the dataframe with new feature columns added
    """

    # -------------------------------------------------------------------------
    # FEATURE 1 — ACADEMIC STRENGTH
    # -------------------------------------------------------------------------
    # Multiplies GRE total score by undergraduate GPA
    # Why: a student with both a high GRE AND high GPA is much stronger than
    # someone who is good at only one of them
    # e.g. GRE=320, GPA=3.8 → academic_strength = 320 * 3.8 = 1216 (very strong)
    # e.g. GRE=300, GPA=3.0 → academic_strength = 300 * 3.0 = 900 (average)
    # This combined signal is more informative than either column alone
    # Applies to both Masters and PhD since both require GRE and GPA
    data['academic_strength'] = data['gre_total'] * data['undergrad_gpa']

    # -------------------------------------------------------------------------
    # FEATURE 2 — APPLICATION STRENGTH
    # -------------------------------------------------------------------------
    # Combines Statement of Purpose strength, Letter of Recommendation average
    # strength, and number of LORs into one application quality score
    # Why: these three together paint a complete picture of how strong the
    # application package is beyond just test scores
    # sop_strength: ranges 1-5 (how compelling the personal statement is)
    # lor_avg_strength: ranges 1-5 (average quality of recommendation letters)
    # lor_count: number of recommendation letters submitted
    # We weight lor_avg_strength more (x2) because quality matters more than quantity
    # Applies to both Masters and PhD
    data['application_strength'] = (
        data['sop_strength'] +
        (data['lor_avg_strength'] * 2) +
        data['lor_count']
    )

    # -------------------------------------------------------------------------
    # FEATURE 3 — RESEARCH POWER (PhD focused but included for Masters too)
    # -------------------------------------------------------------------------
    # Combines research experience years, publications count and conference papers
    # into one research credibility score
    # Why: for PhD admissions research experience is the single most important factor
    # Publications are weighted 3x because a published paper is much harder to achieve
    # than just having research experience or attending conferences
    # Conference papers weighted 2x because they show active participation in the field
    # research_experience_years: 0-10+ years
    # publications_count: 0-4 in our dataset
    # conference_papers: 0-3 in our dataset
    # e.g. 2 years experience + 1 publication + 1 conference = 2 + 3 + 2 = 7 (solid)
    # e.g. 0 years + 0 publications + 0 conferences = 0 (weak research profile)
    data['research_power'] = (
        data['research_experience_years'] +
        (data['publications_count'] * 3) +
        (data['conference_papers'] * 2)
    )

    # -------------------------------------------------------------------------
    # FEATURE 4 — OVERALL PROFILE SCORE
    # -------------------------------------------------------------------------
    # A single number that summarizes the entire applicant profile
    # Combines academic strength, application strength and research power
    # normalized by dividing to bring them to a comparable scale
    # Why: gives the model a holistic view of the applicant in one feature
    # This feature alone can be very predictive of admission outcomes
    data['overall_profile_score'] = (
        (data['academic_strength'] / 1000) +  # divide by 1000 to normalize (GRE*GPA scale)
        data['application_strength'] +
        data['research_power']
    )

    print(f"  Feature engineering complete for {degree_label}")
    print(f"  New features added: academic_strength, application_strength, research_power, overall_profile_score")

    return data


# =============================================================================
# SECTION 4 — PREPROCESSING FUNCTION
# =============================================================================
# This function takes a raw dataframe and cleans it up so the model can use it
# We write it as a function so we can call it for both Masters and PhD data
# without writing the same code twice

def preprocess(data, scaler=None, university_means=None, fit=True):
    """
    Cleans and prepares raw applicant data for model training or prediction.

    Parameters:
        data: the raw dataframe to preprocess
        scaler: a fitted MinMaxScaler (pass this when preprocessing test/app data)
        university_means: dict of mean admission probability per university (for target encoding)
        fit: if True, fit the scaler on this data (training). if False, just transform (test/app data)

    Returns:
        X: the cleaned feature matrix ready for the model
        scaler: the fitted scaler (so we can save it and reuse it in the app)
        university_means: the university encoding dict (so we can reuse it in the app)
    """

    # Make a copy so we dont modify the original dataframe
    data = data.copy()

    # -------------------------------------------------------------------------
    # DROP COLUMNS THAT ARE USELESS FOR PREDICTION
    # -------------------------------------------------------------------------
    # applicant_id: just an ID number, has no relationship with admission chances
    # nationality: too many unique values, not directly useful (is_international covers this)
    # admission_year, admission_semester, application_round: administrative info, not predictive
    # waitlisted: this is an outcome variable like admitted_binary, not a feature
    # degree_type: we already split the data by degree type so this column is redundant
    columns_to_drop = [
        'applicant_id', 'nationality', 'admission_year',
        'admission_semester', 'application_round', 'waitlisted', 'degree_type'
    ]
    # only drop columns that actually exist in the dataframe
    data = data.drop(columns=[c for c in columns_to_drop if c in data.columns])

    # -------------------------------------------------------------------------
    # HANDLE MISSING VALUES FOR OPTIONAL TEST SCORES
    # -------------------------------------------------------------------------
    # Not every applicant submits every test score
    # GMAT is only for business programs, TOEFL/IELTS only for international students
    # We cant just leave them blank — the model needs numbers
    # Solution: fill blanks with 0 AND add a flag column so the model knows
    # the difference between "scored 0" and "didnt submit"

    # GMAT scores
    # has_gmat = 1 if the applicant submitted a GMAT score, 0 if they didnt
    data['has_gmat'] = data['gmat_total'].notna().astype(int)
    data['gmat_total'] = data['gmat_total'].fillna(0)
    data['gmat_verbal'] = data['gmat_verbal'].fillna(0)
    data['gmat_quant'] = data['gmat_quant'].fillna(0)

    # TOEFL scores
    data['has_toefl'] = data['toefl_score'].notna().astype(int)
    data['toefl_score'] = data['toefl_score'].fillna(0)

    # IELTS scores
    data['has_ielts'] = data['ielts_score'].notna().astype(int)
    data['ielts_score'] = data['ielts_score'].fillna(0)

    # -------------------------------------------------------------------------
    # TARGET ENCODE THE UNIVERSITY COLUMN
    # -------------------------------------------------------------------------
    # applied_university has 186 unique values
    # One-hot encoding would create 186 new columns which is too many
    # Target encoding is smarter: replace each university name with the
    # AVERAGE admission_probability of all applicants to that university
    # e.g. MIT might become 0.45 (very competitive) and a lower ranked uni might become 0.78
    # This captures university prestige/competitiveness in a single number

    if fit:
        # During training: calculate the mean admission probability per university
        university_means = data.groupby('applied_university')['admission_probability'].mean().to_dict()

    # Replace university names with their mean admission probability
    # If a university wasnt seen during training (unlikely but possible), use the overall mean
    overall_mean = data['admission_probability'].mean() if 'admission_probability' in data.columns else 0.5
    data['applied_university'] = data['applied_university'].map(university_means).fillna(overall_mean)

    # -------------------------------------------------------------------------
    # ONE-HOT ENCODE CATEGORICAL COLUMNS
    # -------------------------------------------------------------------------
    # Models cant work with text — everything needs to be numbers
    # One-hot encoding converts each unique text value into its own column with 1 or 0
    # e.g. program_field = 'STEM' becomes: program_field_STEM=1, program_field_Business=0, program_field_Arts=0
    # drop_first=True removes one column per category to avoid redundancy

    categorical_columns = [
        'university_region',         # e.g. North America, Europe, Asia
        'university_country',        # e.g. USA, UK, Canada
        'university_qs_tier',        # e.g. Top 50, Top 100, etc.
        'program_name',              # e.g. Computer Science, MBA, Physics
        'program_field',             # e.g. STEM, Business, Arts
        'funding_type',              # e.g. Fellowship, TA, RA, Self-funded
        'undergrad_university_tier'  # tier of the undergrad university
    ]

    # only encode columns that actually exist in the dataframe
    categorical_columns = [c for c in categorical_columns if c in data.columns]
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # -------------------------------------------------------------------------
    # SEPARATE FEATURES FROM TARGET VARIABLES
    # -------------------------------------------------------------------------
    # X = the input features (everything we use to make predictions)
    # We drop admission_probability and admitted_binary because those are what we're predicting
    # not inputs to the model

    target_columns = ['admission_probability', 'admitted_binary']
    X = data.drop(columns=[c for c in target_columns if c in data.columns])

    # -------------------------------------------------------------------------
    # NORMALIZE NUMERICAL COLUMNS
    # -------------------------------------------------------------------------
    # MinMaxScaler squishes all numerical values into 0-1 range
    # This is important because the model treats bigger numbers as more important
    # Without scaling: GRE score of 320 looks 80x more important than GPA of 3.9
    # With scaling: both become values like 0.75 and 0.88 — treated equally
    # We also include our new engineered features here so they get normalized too

    numerical_columns = [
        'undergrad_gpa', 'gre_total', 'gre_verbal', 'gre_quantitative',
        'gre_analytical_writing', 'gmat_total', 'gmat_verbal', 'gmat_quant',
        'toefl_score', 'ielts_score', 'sop_strength', 'sop_word_count',
        'lor_count', 'lor_avg_strength', 'lor_from_professor', 'lor_from_industry',
        'research_experience_years', 'publications_count', 'conference_papers',
        'work_experience_years', 'internships_count', 'work_industry_relevance',
        'applied_university',       # already encoded as a number, still needs scaling
        'academic_strength',        # new engineered feature — GRE * GPA
        'application_strength',     # new engineered feature — SOP + LOR quality + count
        'research_power',           # new engineered feature — research + publications + conferences
        'overall_profile_score'     # new engineered feature — holistic applicant score
    ]

    # only scale columns that actually exist in X
    numerical_columns = [c for c in numerical_columns if c in X.columns]

    if fit:
        # During training: fit the scaler on training data and transform it
        # fitting means the scaler learns the min and max of each column
        scaler = MinMaxScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    else:
        # During testing/app use: only transform using the already fitted scaler
        # NEVER refit on test data — that would cause data leakage
        X[numerical_columns] = scaler.transform(X[numerical_columns])

    return X, scaler, university_means


# =============================================================================
# SECTION 5 — TRAIN AND EVALUATE FUNCTION
# =============================================================================
# This function trains both the regression and classification models
# on a given dataframe and evaluates them
# We write it as a function so we can call it for both Masters and PhD

def train_and_evaluate(data, degree_label):
    """
    Trains XGBoost Regressor and Classifier on the given data.
    Evaluates both models and saves them as .pkl files.

    Parameters:
        data: raw dataframe (masters_df or phd_df)
        degree_label: string label for saving files e.g. 'masters' or 'phd'
    """

    print(f"\n{'='*60}")
    print(f"TRAINING {degree_label.upper()} MODELS")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # REMOVE INVALID RECORDS
    # -------------------------------------------------------------------------
    # Remove rows with GRE scores outside valid range (260-340)
    data = data[data['gre_total'].between(260, 340)]
    # Remove rows with GPA outside valid range (2.0-4.0)
    data = data[data['undergrad_gpa'].between(2.0, 4.0)]
    # Remove rows with invalid TOEFL scores (if submitted, must be at least 70)
    data = data[~((data['toefl_score'] > 0) & (data['toefl_score'] < 70))]
    # Drop duplicate applicant IDs — keep only the first occurrence
    data = data.drop_duplicates(subset='applicant_id') if 'applicant_id' in data.columns else data

    print(f"Records after cleaning: {len(data)}")

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    # Create new combined features before preprocessing
    # This must happen before preprocessing so the new features also get normalized
    data = engineer_features(data, degree_label)

    # -------------------------------------------------------------------------
    # SEPARATE TARGETS BEFORE PREPROCESSING
    # -------------------------------------------------------------------------
    # Save the target columns before preprocessing removes them
    y_regression = data['admission_probability']   # continuous value 0-1
    y_classification = data['admitted_binary']      # binary value 0 or 1

    # -------------------------------------------------------------------------
    # PREPROCESS THE DATA
    # -------------------------------------------------------------------------
    # fit=True because this is training data — scaler learns from this data
    X, scaler, university_means = preprocess(data, fit=True)

    # -------------------------------------------------------------------------
    # SPLIT INTO TRAINING AND TESTING SETS
    # -------------------------------------------------------------------------
    # 80% of data goes to training (model learns from this)
    # 20% of data goes to testing (model gets evaluated on this — data it has never seen)
    # random_state=42 ensures the split is the same every time we run the script
    # test_size=0.2 means 20% test data

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    # Same split indices for classification targets
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------------------
    # TRAIN THE XGBOOST REGRESSION MODEL
    # -------------------------------------------------------------------------
    # XGBoost builds trees sequentially — each tree corrects the errors of the previous one
    # This is called "boosting" and is why XGBoost is more accurate than Random Forest
    #
    # n_estimators=200: build 200 sequential trees
    #   more trees = more corrections = more accurate (but slower to train)
    #
    # learning_rate=0.1: how much each new tree corrects the previous ones
    #   lower = more careful corrections = better accuracy but needs more trees
    #   0.1 is the standard sweet spot
    #
    # max_depth=6: how deep each tree can grow
    #   deeper = learns more complex patterns but risks overfitting
    #   6 is the standard sweet spot for most datasets
    #
    # random_state=42: ensures the same results every time we run
    # n_jobs=-1: use all CPU cores to train faster

    print("\nTraining XGBoost Regression Model...")
    reg_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    reg_model.fit(X_train, y_reg_train)  # model studies the training data

    # -------------------------------------------------------------------------
    # EVALUATE THE REGRESSION MODEL
    # -------------------------------------------------------------------------
    y_reg_pred = reg_model.predict(X_test)  # model predicts on unseen test data

    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"\nRegression Results ({degree_label}):")
    print(f"  MAE  (avg error in probability): {mae:.4f}")
    print(f"  RMSE (penalizes big errors more): {rmse:.4f}")
    print(f"  R²   (how much variation explained): {r2:.4f}")

    # -------------------------------------------------------------------------
    # TRAIN THE XGBOOST CLASSIFICATION MODEL
    # -------------------------------------------------------------------------
    # Same XGBoost approach but for binary classification (0 or 1)
    # eval_metric='logloss': the loss function used to measure classification errors
    #   logloss penalizes confident wrong predictions more heavily
    #   e.g. saying someone has 99% chance of admission when they actually get rejected
    #   is penalized much more than saying 51% chance

    print("\nTraining XGBoost Classification Model...")
    clf_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    clf_model.fit(X_train, y_clf_train)

    # -------------------------------------------------------------------------
    # EVALUATE THE CLASSIFICATION MODEL
    # -------------------------------------------------------------------------
    y_clf_pred = clf_model.predict(X_test)

    accuracy = accuracy_score(y_clf_test, y_clf_pred)
    precision = precision_score(y_clf_test, y_clf_pred)
    recall = recall_score(y_clf_test, y_clf_pred)
    f1 = f1_score(y_clf_test, y_clf_pred)
    cm = confusion_matrix(y_clf_test, y_clf_pred)

    print(f"\nClassification Results ({degree_label}):")
    print(f"  Accuracy  (overall correct %): {accuracy:.4f}")
    print(f"  Precision (of predicted admitted, how many were): {precision:.4f}")
    print(f"  Recall    (of actual admitted, how many caught): {recall:.4f}")
    print(f"  F1 Score  (balance of precision and recall): {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correctly predicted rejected): {cm[0][0]}")
    print(f"  False Positives (predicted admitted, actually rejected): {cm[0][1]}")
    print(f"  False Negatives (predicted rejected, actually admitted): {cm[1][0]}")
    print(f"  True Positives  (correctly predicted admitted): {cm[1][1]}")

    # -------------------------------------------------------------------------
    # SAVE THE MODELS AND SCALER AS .PKL FILES
    # -------------------------------------------------------------------------
    # Make sure the Models folder exists
    os.makedirs('Models', exist_ok=True)

    # Save regression model — contains all 200 trained XGBoost trees and their findings
    joblib.dump(reg_model, f'Models/{degree_label}_regression.pkl')
    print(f"\nSaved: Models/{degree_label}_regression.pkl")

    # Save classification model
    joblib.dump(clf_model, f'Models/{degree_label}_classification.pkl')
    print(f"Saved: Models/{degree_label}_classification.pkl")

    # Save the scaler — critical! the app needs to scale user inputs the same way
    joblib.dump(scaler, f'Models/{degree_label}_scaler.pkl')
    print(f"Saved: Models/{degree_label}_scaler.pkl")

    # Save university means — needed for target encoding in the app
    joblib.dump(university_means, f'Models/{degree_label}_university_means.pkl')
    print(f"Saved: Models/{degree_label}_university_means.pkl")


# =============================================================================
# SECTION 6 — RUN EVERYTHING
# =============================================================================
# This is the entry point — when you run model.py, this is what executes

if __name__ == '__main__':
    # Train and evaluate Masters models (MS + MBA combined)
    train_and_evaluate(masters_df, 'masters')

    # Train and evaluate PhD models
    train_and_evaluate(phd_df, 'phd')

    print("\n\nAll models trained and saved successfully!")
    print("You can now run the backend server.")