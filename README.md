# 🎓 Student Performance Prediction System

Machine Learning based Student Performance Prediction using **XGBoost**, wrapped in an interactive Streamlit dashboard.

**Author:** Mohammed Aasik &nbsp;|&nbsp; [GitHub](https://github.com/mdaasik-tech) &nbsp;|&nbsp; [LinkedIn](https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/)

## Project Overview
Predicts a student's **final exam score** from study habits, attendance, previous GPA, school and family details, then derives a **PASS / FAIL** result from a configurable passing score.

## Problem Statement
Schools want to spot students at risk *before* the exam. Given a student's profile, can we estimate the expected score and flag likely failures early?

## Objective
Build, tune and evaluate an XGBoost regression model, and expose it through a dashboard with dataset exploration, model evaluation, feature importance and a live prediction form.

## Machine Learning Approach
The app reproduces the original notebook (`STUDENT_PERFORMANCE_PREDICTION_SYSTEM.ipynb`):

1. Remove duplicate rows
2. Drop identifier / leakage columns (`student_id`, `pass_fail`, `letter_grade`)
3. Train/test split — **80 / 20, `random_state=42`**
4. Fit `LabelEncoder` (categoricals) and `StandardScaler` (numericals) on the **training set only**
5. Tune `XGBRegressor(objective="reg:squarederror")` with `RandomizedSearchCV`
6. Clip predictions to 0–100, evaluate, and convert to PASS / FAIL

## Dataset
`student_performance_prediction_dataset.csv` — 5,000 students, 16 columns. Loaded automatically; any compatible CSV can be uploaded from the sidebar.

## Features
| Type | Columns |
|---|---|
| Numerical | `study_hours_per_week`, `attendance_rate_pct`, `previous_gpa`, `sleep_hours_per_night`, `physical_activity_hours` |
| Categorical | `school_type`, `school_location`, `gender`, `parent_education`, `internet_access`, `extracurricular_activities`, `tutoring_sessions` |
| Excluded | `student_id` (identifier), `pass_fail`, `letter_grade` (derived from the target → leakage) |

## Target
`final_exam_score` (continuous, 35–98.9 in the default data).

## Data Preprocessing
- Column names stripped; `N/A`, `null`, empty strings → missing; `±inf` → missing
- Numeric strings converted to numbers; mixed columns flagged with a warning
- Date columns expanded into year / month / day / day-of-week
- Missing values imputed **after** the split (median / most frequent), learned on train only
- Identifier and possible-leakage columns detected and excluded by default (user can review)

## Label Encoding
One `LabelEncoder` per categorical column, fitted on the training set. Unseen categories never crash the app (test rows → `-1`; the prediction form only offers known values).

## Standard Scaling
One `StandardScaler` for all numerical features, fitted on the training set and reused for every prediction.

## XGBoost
`XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)` plus tuned parameters.

## Hyperparameter Tuning
`RandomizedSearchCV` — `n_iter=40`, `cv=5`, `scoring="neg_root_mean_squared_error"`, same search space as the notebook. In the sidebar you can either use the notebook's result (instant) or re-run the search (≈ 1 minute; it recovers the same parameters on the default data).

## Best Parameters
| Parameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 2 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.7 |
| min_child_weight | 7 |
| gamma | 0.3 |
| reg_alpha | 1 |
| reg_lambda | 5 |

The app always displays the parameters of the model actually in use.

## Evaluation Metrics
MAE, MSE, RMSE and R² on the held-out test set, plus a baseline-vs-tuned comparison. On the default data (test size 0.20, seed 42) the notebook reports MAE 3.318, RMSE 4.176, R² 0.837; the app computes all values live.

## PASS / FAIL Logic
```
predicted_score >= passing_score  →  PASS
predicted_score <  passing_score  →  FAIL
```
`passing_score` defaults to **40** (the notebook's threshold) and can be changed in the sidebar. Note: the dataset's own `pass_fail` column corresponds to a cutoff of about 60; the app shows this hint next to the setting.

## Streamlit Features
Sidebar controls · CSV upload · automatic column-type / target / identifier detection · feature selection · metric cards · 7 tabs (Overview, Dataset, Model Performance, Predictions, Student Predictor, Feature Analysis, Model Information) · interactive Plotly charts · dynamic prediction form · CSV/JSON downloads · theme-safe styling (native Streamlit components only, no hard-coded colours).

## Project Structure
```
student-performance-prediction/
├── app.py
├── student_performance_prediction_dataset.csv
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```

## Future Improvements
- Add SHAP explanations for individual predictions
- Save/load trained models with `joblib`
- Compare against other algorithms (as a clearly separate experiment)
- Classification view for the letter grade

## Author
**Mohammed Aasik** — Aspiring Machine Learning Engineer

- GitHub: [github.com/mdaasik-tech](https://github.com/mdaasik-tech)
- LinkedIn: [Mohammed Aasik](https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/)
