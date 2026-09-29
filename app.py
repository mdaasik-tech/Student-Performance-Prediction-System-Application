"""
🎓 Student Performance Prediction System
Machine Learning based Student Performance Prediction using XGBoost

Author  : Mohammed Aasik
GitHub  : https://github.com/mdaasik-tech
LinkedIn: https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/

The ML pipeline mirrors the original notebook (STUDENT_PERFORMANCE_PREDICTION_SYSTEM.ipynb):
    remove duplicates -> drop identifier/leakage columns -> train/test split (80/20, seed 42)
    -> LabelEncoder (categoricals, fit on train) -> StandardScaler (numericals, fit on train)
    -> XGBRegressor tuned with RandomizedSearchCV -> clip predictions to 0-100 -> PASS/FAIL.
"""

import hashlib
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =====================================================================
# 1. PROJECT CONFIGURATION (taken from the notebook)
# =====================================================================
DEFAULT_CSV = Path(__file__).parent / "student_performance_prediction_dataset.csv"

NB_TARGET = "final_exam_score"
NB_TEST_SIZE = 0.20
NB_RANDOM_STATE = 42
NB_PASSING_SCORE = 40  # notebook: score < 40 -> FAIL
NB_N_ITER = 40
NB_CV = 5
NB_SCORING = "neg_root_mean_squared_error"

# Search space used in the notebook's RandomizedSearchCV
NB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2, 5],
}
# best_params_ printed by the notebook's search (best CV RMSE = 4.1013)
NB_BEST_PARAMS = {
    "subsample": 0.8, "reg_lambda": 5, "reg_alpha": 1, "n_estimators": 300,
    "min_child_weight": 7, "max_depth": 2, "learning_rate": 0.1,
    "gamma": 0.3, "colsample_bytree": 0.7,
}
NB_BEST_CV_RMSE = 4.1013

TARGET_HINTS = {"performance", "score", "finalscore", "finalexamscore", "marks", "totalmarks",
                "finalmarks", "grade"}
ID_TOKENS = {"id", "uuid", "guid", "roll", "rollno", "rollnumber", "serial", "regno", "regid"}
LEAKAGE_PATTERN = re.compile(r"pass|fail|grade|result|status|outcome|remark", re.I)
BOOL_VALUES = {"yes", "no", "true", "false", "y", "n", "t", "f", "0", "1"}
NULL_STRINGS = {"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan,
                "null": np.nan, "NULL": np.nan, "N/A": np.nan, "NA": np.nan, "n/a": np.nan}
MIN_ROWS = 30

AUTHOR_NAME = "Mohammed Aasik"
AUTHOR_GITHUB = "https://github.com/mdaasik-tech"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/"


# =====================================================================
# 2. DATA LOADING
# =====================================================================
@st.cache_data(show_spinner=False)
def load_default_dataset():
    """Load the bundled student performance dataset."""
    if not DEFAULT_CSV.exists():
        return None
    return pd.read_csv(DEFAULT_CSV)


def load_uploaded_dataset(file_bytes: bytes):
    """Read an uploaded CSV. Returns (dataframe, error_message)."""
    if not file_bytes or not file_bytes.strip():
        return None, "⚠️ Uploaded dataset is empty."
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "⚠️ Uploaded dataset is empty."
        except Exception:
            return None, "⚠️ The file could not be read as a valid CSV."
    else:
        return None, "⚠️ The file could not be decoded as text."
    if df.shape[0] == 0 or df.shape[1] == 0:
        return None, "⚠️ Uploaded dataset is empty."
    return df, None


# =====================================================================
# 3. CLEANING + FEATURE TYPE DETECTION
# =====================================================================
@st.cache_data(show_spinner=False)
def clean_dataset(df_raw: pd.DataFrame):
    """Strip names/whitespace, unify null tokens, handle inf, empty rows, duplicates."""
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():  # make repeated column names unique instead of crashing
        seen, new = {}, []
        for c in df.columns:
            seen[c] = seen.get(c, -1) + 1
            new.append(c if seen[c] == 0 else f"{c}_{seen[c]}")
        df.columns = new
    report = {
        "rows_raw": len(df),
        "duplicates_raw": int(df.duplicated().sum()),
        "missing_raw": pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
        }),
        "dtypes_raw": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df_raw.memory_usage(deep=True).sum() / 1024**2, 3),
    }
    # text columns: trim spaces and convert "N/A", "null", "" ... to NaN
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            s = s.astype("string").str.strip().replace(NULL_STRINGS)
            df[col] = s.astype(object).where(s.notna(), np.nan)
    num_cols = df.select_dtypes(include=[np.number]).columns
    report["inf_replaced"] = int(np.isinf(df[num_cols].to_numpy(dtype=float)).sum()) if len(num_cols) else 0
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    report["empty_rows_removed"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # the notebook also drops duplicates
    report["duplicates_removed"] = before - len(df)
    return df, report


def _tokens(name: str):
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def detect_id_columns(df: pd.DataFrame):
    """Columns that look like identifiers (by name or by being unique text)."""
    ids = []
    for col in df.columns:
        low = col.lower().replace("_", "").replace(" ", "")
        by_name = any(t in ID_TOKENS for t in _tokens(col)) or (
            low.endswith("id") and low[:-2] in {"student", "registration", "reg", "user", "roll"})
        s = df[col].dropna()
        looks_date = len(s) > 0 and s.astype(str).str.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.9
        unique_text = (not pd.api.types.is_numeric_dtype(s) and len(s) > 0 and not looks_date
                       and s.nunique() == len(s) and len(s) >= 20)
        if by_name or unique_text:
            ids.append(col)
    return ids


def detect_feature_types(df: pd.DataFrame):
    """
    Inspect dtypes/values and decide how each column is processed.
    Returns (converted_df, types, notes) where types[col] is one of:
    numeric | categorical | boolean | datetime | identifier | high_cardinality
    """
    df = df.copy()
    types, notes = {}, []
    ids = set(detect_id_columns(df))
    for col in list(df.columns):
        s = df[col]
        nonnull = s.dropna()
        if col in ids:
            types[col] = "identifier"
        elif pd.api.types.is_bool_dtype(s):
            df[col] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            types[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        else:  # text-like column
            as_num = pd.to_numeric(nonnull, errors="coerce")
            ratio = float(as_num.notna().mean()) if len(nonnull) else 0.0
            if ratio == 1.0:
                df[col] = pd.to_numeric(s, errors="coerce")
                types[col] = "numeric"
                notes.append(f"🔢 `{col}` was stored as text but contains only numbers → converted to numeric.")
            elif ratio >= 0.9:
                df[col] = pd.to_numeric(s, errors="coerce")
                types[col] = "numeric"
                bad = int(nonnull.shape[0] - as_num.notna().sum())
                notes.append(f"⚠️ `{col}` is a mixed column ({bad} non-numeric values were set to missing "
                             f"and will be imputed). Please verify it.")
            else:
                parsed = None
                hint = any(t in {"date", "dob", "time", "datetime", "day"} for t in _tokens(col))
                if len(nonnull) and ratio < 0.5 and (hint or nonnull.astype(str).str.contains(r"[-/:]").mean() > 0.9):
                    try:
                        parsed = pd.to_datetime(nonnull.head(200), errors="coerce", format="mixed")
                    except Exception:
                        parsed = None
                if parsed is not None and parsed.notna().mean() >= 0.9:
                    types[col] = "datetime"
                    full = pd.to_datetime(s, errors="coerce", format="mixed")
                    for part, val in (("year", full.dt.year), ("month", full.dt.month),
                                      ("day", full.dt.day), ("dayofweek", full.dt.dayofweek)):
                        df[f"{col}_{part}"] = val.astype(float)
                        types[f"{col}_{part}"] = "numeric"
                    notes.append(f"📅 `{col}` looks like a date → year/month/day/dayofweek columns were extracted.")
                else:
                    df[col] = s.map(lambda v: v if pd.isna(v) else str(v))
                    vals = {str(v).lower() for v in df[col].dropna().unique()}
                    if 0 < len(vals) <= 2 and vals <= BOOL_VALUES:
                        types[col] = "boolean"
                    elif df[col].nunique() > max(50, 0.2 * len(df)):
                        types[col] = "high_cardinality"
                        notes.append(f"ℹ️ `{col}` has too many distinct text values → excluded from features.")
                    else:
                        types[col] = "categorical"
                    if 0.5 <= ratio < 0.9:
                        notes.append(f"⚠️ `{col}` mixes numbers and text; it is treated as categorical.")
    # keep the dataframe column order (the model depends on it)
    types = {c: types[c] for c in df.columns if c in types}
    return df, types, notes


def _norm(name: str):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_target_column(df: pd.DataFrame, types: dict):
    """The notebook target wins. Otherwise use name hints. Returns (target_or_None, candidates)."""
    if types.get(NB_TARGET) == "numeric":
        return NB_TARGET, [NB_TARGET]
    cands = [c for c, t in types.items() if t == "numeric"
             and (_norm(c) in TARGET_HINTS or "score" in _norm(c) or "marks" in _norm(c))]
    return (cands[0] if len(cands) == 1 else None), cands


def detect_leakage_columns(df: pd.DataFrame, types: dict, target: str):
    """Columns likely derived from the target (excluded by default)."""
    leak = [c for c in df.columns if c != target and LEAKAGE_PATTERN.search(c)]
    if types.get(target) == "numeric":
        for c, t in types.items():
            if t == "numeric" and c != target and c not in leak:
                corr = df[[c, target]].corr().iloc[0, 1]
                if pd.notna(corr) and abs(corr) > 0.98:
                    leak.append(c)
    return leak


def infer_dataset_cutoff(df: pd.DataFrame, target: str):
    """If the dataset has its own Pass/Fail column, estimate the score cutoff it implies."""
    for c in df.columns:
        if c == target or df[c].dtype.kind in "fi":
            continue
        vals = {str(v).lower(): v for v in df[c].dropna().unique()}
        if set(vals) == {"pass", "fail"}:
            lab = df[c].astype(str).str.lower()
            hi_fail, lo_pass = df.loc[lab == "fail", target].max(), df.loc[lab == "pass", target].min()
            if pd.notna(hi_fail) and pd.notna(lo_pass) and hi_fail < lo_pass:
                return c, float((hi_fail + lo_pass) / 2)
    return None, None


def validate_dataset(df: pd.DataFrame, target: str, features: list):
    """Return (errors, warnings) — user-friendly strings."""
    errors, warns = [], []
    if not features:
        errors.append("⚠️ Dataset does not contain enough usable features. Select at least one feature.")
    valid_rows = int(df[target].notna().sum()) if target in df else 0
    if valid_rows < MIN_ROWS:
        errors.append(f"⚠️ The dataset contains too few valid rows for model training (need ≥ {MIN_ROWS}, found {valid_rows}).")
    elif df[target].nunique() < 2:
        errors.append("⚠️ The target column has only one unique value — nothing to predict.")
    if len(features) == 1:
        warns.append("ℹ️ Only one feature is selected; predictions will be limited.")
    missing_target = int(df[target].isna().sum())
    if missing_target:
        warns.append(f"ℹ️ {missing_target} rows have a missing target and are excluded from training.")
    return errors, warns


# =====================================================================
# 4. PREPROCESSING (fit on TRAIN only -> no data leakage)
# =====================================================================
def fit_encoders(X_train: pd.DataFrame, cat_cols: list):
    return {c: LabelEncoder().fit(X_train[c].astype(str)) for c in cat_cols}


def encode_features(X: pd.DataFrame, cat_cols: list, encoders: dict):
    """LabelEncoder transform; unseen categories become -1 instead of crashing."""
    X, unseen = X.copy(), {}
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(encoders[c].classes_)}
        codes = X[c].astype(str).map(mapping)
        unseen[c] = int(codes.isna().sum())
        X[c] = codes.fillna(-1).astype(int)
    return X, unseen


def scale_features(X: pd.DataFrame, num_cols: list, scaler: StandardScaler):
    X = X.copy()
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])
    return X


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}


def clip_predictions(pred, clip_range):
    return np.clip(pred, *clip_range) if clip_range else pred


def build_model(random_state: int, params: dict | None = None):
    """XGBRegressor configured exactly like the notebook."""
    return XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=-1, **(params or {}))


def tune_model(X_train, y_train, random_state, n_iter, cv):
    """RandomizedSearchCV with the notebook's search space/scoring."""
    search = RandomizedSearchCV(
        estimator=build_model(random_state), param_distributions=NB_PARAM_GRID, n_iter=n_iter,
        scoring=NB_SCORING, cv=cv, random_state=random_state, n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)
    return search


@st.cache_resource(show_spinner=False)
def train_model(data: pd.DataFrame, target: str, num_cols: tuple, cat_cols: tuple,
                test_size: float, random_state: int, tuning_mode: str, n_iter: int, cv: int):
    """Full pipeline: split -> impute -> encode -> scale -> baseline -> tuned XGBoost -> evaluate."""
    num_cols, cat_cols = list(num_cols), list(cat_cols)
    features = [c for c in data.columns if c in num_cols + cat_cols]  # keep dataset column order
    data = data.dropna(subset=[target])
    X, y = data[features].copy(), data[target].astype(float)

    # 1) split FIRST — everything below learns from the training rows only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 2) missing values (statistics learned on train)
    if num_cols:
        imp = SimpleImputer(strategy="median").fit(X_train[num_cols])
        X_train[num_cols] = imp.transform(X_train[num_cols])
        X_test[num_cols] = imp.transform(X_test[num_cols])
    else:
        imp = None
    if cat_cols:
        imp_c = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
        X_train[cat_cols] = imp_c.transform(X_train[cat_cols])
        X_test[cat_cols] = imp_c.transform(X_test[cat_cols])
    else:
        imp_c = None
    raw_train = X_train.copy()  # un-encoded copy for the prediction form

    # 3) LabelEncoder (categoricals) + StandardScaler (numericals), fit on train
    encoders = fit_encoders(X_train, cat_cols)
    X_train_p, _ = encode_features(X_train, cat_cols, encoders)
    X_test_p, unseen = encode_features(X_test, cat_cols, encoders)
    scaler = StandardScaler().fit(X_train_p[num_cols]) if num_cols else None
    X_train_p = scale_features(X_train_p, num_cols, scaler)
    X_test_p = scale_features(X_test_p, num_cols, scaler)

    clip_range = (0, 100) if (y_train.min() >= 0 and y_train.max() <= 100) else None  # notebook: 0-100

    # 4) baseline XGBoost (default settings) for comparison
    baseline = build_model(random_state).fit(X_train_p, y_train)
    baseline_metrics = evaluate_model(y_test, clip_predictions(baseline.predict(X_test_p), clip_range))

    # 5) tuned XGBoost
    cv_results, cv_rmse = None, None
    if tuning_mode == "notebook":
        best_params = dict(NB_BEST_PARAMS)
        model = build_model(random_state, best_params).fit(X_train_p, y_train)
        cv_rmse, source = NB_BEST_CV_RMSE, "Best parameters from the original notebook's RandomizedSearchCV"
    else:
        cv = int(max(2, min(cv, len(X_train_p) // 5)))
        search = tune_model(X_train_p, y_train, random_state, n_iter, cv)
        best_params, model = search.best_params_, search.best_estimator_
        cv_rmse = float(-search.best_score_)
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        source = f"RandomizedSearchCV re-run on this data (n_iter={n_iter}, cv={cv})"

    # 6) evaluate on the untouched test set
    y_pred = clip_predictions(model.predict(X_test_p), clip_range)
    metrics = evaluate_model(y_test, y_pred)

    ranges = {c: (float(raw_train[c].min()), float(raw_train[c].max()), float(raw_train[c].median()),
                  bool((raw_train[c] % 1 == 0).all())) for c in num_cols}
    categories = {c: list(encoders[c].classes_) for c in cat_cols}
    modes = {c: raw_train[c].mode().iloc[0] for c in cat_cols}
    importance = pd.DataFrame({"Feature": X_train_p.columns, "Importance": model.feature_importances_}
                              ).sort_values("Importance", ascending=False).reset_index(drop=True)
    return dict(model=model, encoders=encoders, scaler=scaler, features=features, num=num_cols, cat=cat_cols,
                X_train_p=X_train_p, X_test_p=X_test_p, y_train=y_train, y_test=y_test, y_pred=y_pred,
                metrics=metrics, baseline_metrics=baseline_metrics, best_params=best_params, cv_rmse=cv_rmse,
                source=source, cv_results=cv_results, clip_range=clip_range, ranges=ranges,
                categories=categories, modes=modes, importance=importance, unseen=unseen,
                target=target, n_rows=len(data), n_train=len(X_train_p), n_test=len(X_test_p),
                test_size=test_size, random_state=random_state, tuning_mode=tuning_mode,
                n_iter=n_iter, cv=cv)


# =====================================================================
# 5. PREDICTION LOGIC
# =====================================================================
def get_performance(score: float, passing_score: float):
    """PASS/FAIL uses the configurable passing score; level bands follow the notebook."""
    result = "PASS" if score >= passing_score else "FAIL"
    if score < 40:
        level = "Poor"
    elif score < 50:
        level = "Average"
    elif score < 70:
        level = "Good"
    elif score < 85:
        level = "Very Good"
    else:
        level = "Excellent"
    return result, level


def generate_predictions(art: dict, passing_score: float):
    """Test-set predictions table."""
    actual, pred = art["y_test"].values, art["y_pred"]
    resid = actual - pred
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(actual != 0, np.abs(resid) / np.abs(actual) * 100, np.nan)
    return pd.DataFrame({
        "Actual Performance": actual.round(2), "Predicted Performance": pred.round(2),
        "Residual": resid.round(2), "Absolute Error": np.abs(resid).round(2),
        "Percentage Error (%)": np.round(pct, 2),
        "Actual Result": np.where(actual >= passing_score, "PASS", "FAIL"),
        "Prediction Result": np.where(pred >= passing_score, "PASS", "FAIL"),
    }, index=art["y_test"].index)


def predict_student(values: dict, art: dict):
    """Validate one student's inputs, apply the SAME preprocessing as training, predict."""
    row = {}
    for c in art["features"]:
        if c not in values or values[c] is None or (isinstance(values[c], float) and not np.isfinite(values[c])):
            raise ValueError(f"Please provide a valid value for '{c}'.")
        row[c] = values[c]
    df = pd.DataFrame([row])[art["features"]]  # exact feature order
    for c in art["num"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    for c in art["cat"]:
        if str(df.loc[0, c]) not in art["categories"][c]:
            raise ValueError(f"'{df.loc[0, c]}' is not a known value for '{c}'. "
                             f"Choose one of: {', '.join(art['categories'][c])}.")
    df, _ = encode_features(df, art["cat"], art["encoders"])
    df = scale_features(df, art["num"], art["scaler"])
    pred = float(clip_predictions(art["model"].predict(df[art["features"]]), art["clip_range"])[0])
    return pred


# =====================================================================
# 6. VISUAL HELPERS
# =====================================================================
def show_chart(build_fig):
    """Render a Plotly chart; skip gracefully if it cannot be built."""
    try:
        st.plotly_chart(build_fig(), width="stretch")
    except Exception:
        st.info("ℹ️ This visualization is not available for the current dataset.")


def to_csv(df: pd.DataFrame, index=False):
    return df.to_csv(index=index).encode("utf-8")


def create_download_file(art, cfg, passing_score):
    payload = {"target": art["target"], "features": art["features"], "model": "XGBRegressor",
               "best_parameters": art["best_params"], "parameter_source": art["source"],
               "cv_rmse": art["cv_rmse"], "test_metrics": art["metrics"],
               "baseline_metrics": art["baseline_metrics"], "test_size": art["test_size"],
               "random_state": art["random_state"], "passing_score": passing_score}
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# =====================================================================
# 7. TABS
# =====================================================================
def tab_overview(df, types, art, passing):
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Project summary")
        st.write(f"**Dataset size:** {len(df):,} rows × {df.shape[1]} columns")
        st.write(f"**Target column:** `{art['target']}`")
        st.write(f"**Numerical features ({len(art['num'])}):** {', '.join(art['num']) or '—'}")
        st.write(f"**Categorical features ({len(art['cat'])}):** {', '.join(art['cat']) or '—'}")
        st.write("**Model:** XGBoost Regressor")
        st.write(f"**RMSE:** {art['metrics']['RMSE']:.3f}  |  **R² Score:** {art['metrics']['R2']:.3f}")
    with c2, st.container(border=True):
        st.subheader("🏆 Best XGBoost Parameters")
        st.dataframe(pd.DataFrame({"Parameter": art["best_params"].keys(),
                                   "Best Value": [str(v) for v in art["best_params"].values()]}),
                     hide_index=True, width="stretch")
        st.caption(art["source"])

    t = df[art["target"]].dropna()

    def dist():
        fig = px.histogram(t, nbins=30, title="Performance Distribution",
                           labels={"value": art["target"], "count": "Students"})
        fig.add_vline(x=passing, line_dash="dash", annotation_text=f"Passing score {passing:g}")
        fig.update_layout(showlegend=False, xaxis_title=art["target"], yaxis_title="Number of students")
        return fig

    def passfail():
        counts = pd.Series(np.where(t >= passing, "PASS", "FAIL")).value_counts().reset_index()
        counts.columns = ["Result", "Students"]
        return px.bar(counts, x="Result", y="Students", color="Result", text="Students",
                      title=f"Pass vs Fail (passing score = {passing:g})")

    a, b = st.columns(2)
    with a:
        show_chart(dist)
    with b:
        show_chart(passfail)

    if art["num"]:
        def fvp():
            imp = art["importance"]
            feat = imp[imp["Feature"].isin(art["num"])]["Feature"].iloc[0]
            fig = px.scatter(df, x=feat, y=art["target"], opacity=0.5, title=f"Top numeric feature vs performance: {feat}")
            return fig
        show_chart(fvp)


def tab_dataset(raw_report, df, types, art, feats, ids, leak):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows (raw → clean)", f"{raw_report['rows_raw']:,} → {len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Memory usage", f"{raw_report['memory_mb']} MB")
    m4.metric("Duplicate rows (raw)", raw_report["duplicates_raw"])

    with st.expander("🧹 Cleaning report", expanded=False):
        st.write(f"- Duplicate rows removed: **{raw_report['duplicates_removed']}**")
        st.write(f"- Completely empty rows removed: **{raw_report['empty_rows_removed']}**")
        st.write(f"- Infinite values replaced with missing: **{raw_report['inf_replaced']}**")
        st.write("- Remaining missing values are imputed **after** the train/test split "
                 "(median for numbers, most frequent for categories).")

    left, right = st.columns(2)
    with left:
        st.subheader("Data types")
        st.dataframe(pd.DataFrame({"Column": list(raw_report["dtypes_raw"]),
                                   "Data Type": list(raw_report["dtypes_raw"].values())}),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("Missing values")
        st.dataframe(raw_report["missing_raw"], hide_index=True, width="stretch")

    st.subheader("Feature classification")
    role = []
    for c, t in types.items():
        if c == art["target"]:
            role.append("Target")
        elif c in art["features"]:
            role.append("Yes")
        elif c in ids:
            role.append("No (identifier)")
        elif c in leak:
            role.append("No (possible leakage)")
        else:
            role.append("No")
    st.dataframe(pd.DataFrame({"Column": list(types), "Type": list(types.values()), "Used for Model": role}),
                 hide_index=True, width="stretch")

    st.subheader("Numerical feature summary")
    nums = [c for c, t in types.items() if t == "numeric"]
    if nums:
        st.dataframe(df[nums].describe().T.round(3), width="stretch")
    else:
        st.info("ℹ️ No numerical columns in this dataset.")
    st.subheader("Categorical feature summary")
    cats = [c for c, t in types.items() if t in ("categorical", "boolean")]
    if cats:
        st.dataframe(df[cats].describe().T, width="stretch")
    else:
        st.info("ℹ️ No categorical columns in this dataset.")

    st.subheader("First 10 rows")
    st.dataframe(df.head(10), width="stretch")
    with st.expander("Cleaned dataset (all rows)"):
        st.dataframe(df, width="stretch")

    model_ready = pd.concat([
        art["X_train_p"].assign(**{art["target"]: art["y_train"], "split": "train"}),
        art["X_test_p"].assign(**{art["target"]: art["y_test"], "split": "test"})])
    d1, d2 = st.columns(2)
    d1.download_button("📥 Download Processed Dataset", to_csv(df), "processed_dataset.csv", "text/csv", width="stretch")
    d2.download_button("📥 Download Model-Ready Data (encoded + scaled)", to_csv(model_ready, index=True),
                       "model_ready_dataset.csv", "text/csv", width="stretch")


def tab_performance(art):
    m = art["metrics"]
    c = st.columns(4)
    c[0].metric("MAE", f"{m['MAE']:.3f}")
    c[1].metric("MSE", f"{m['MSE']:.3f}")
    c[2].metric("RMSE", f"{m['RMSE']:.3f}")
    c[3].metric("R² Score", f"{m['R2']:.4f}")
    with st.expander("What do these metrics mean?"):
        st.markdown("- **MAE** – average absolute prediction error.\n"
                    "- **MSE** – squared error; punishes large mistakes more strongly.\n"
                    "- **RMSE** – square root of MSE, in the same unit as the target.\n"
                    "- **R²** – share of the target's variation explained by the model (1.0 = perfect).")

    b = art["baseline_metrics"]
    comp = pd.DataFrame({"Metric": ["MAE", "RMSE", "R²"],
                         "Baseline XGBoost": [b["MAE"], b["RMSE"], b["R2"]],
                         "Optimized XGBoost": [m["MAE"], m["RMSE"], m["R2"]]})
    with st.container(border=True):
        st.subheader("Baseline vs Optimized XGBoost")
        st.dataframe(comp.round(4), hide_index=True, width="stretch")
        show_chart(lambda: px.bar(comp.melt("Metric", var_name="Model", value_name="Score"), x="Metric",
                                  y="Score", color="Model", barmode="group", title="Baseline vs Optimized"))

    y_true, y_pred = art["y_test"].values, art["y_pred"]

    def avp():
        fig = px.scatter(x=y_true, y=y_pred, opacity=0.55, title="Actual vs Predicted Performance",
                         labels={"x": "Actual Performance", "y": "Predicted Performance"})
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(dash="dash")))
        return fig
    show_chart(avp)

    st.subheader("Residual analysis")
    resid = y_true - y_pred
    r = st.columns(3)
    r[0].metric("Mean residual", f"{resid.mean():.3f}")
    r[1].metric("Maximum residual", f"{resid.max():.3f}")
    r[2].metric("Minimum residual", f"{resid.min():.3f}")
    a, b2 = st.columns(2)
    with a:
        show_chart(lambda: px.histogram(x=resid, nbins=30, title="Residual Distribution",
                                        labels={"x": "Residual (Actual − Predicted)"}).update_layout(yaxis_title="Frequency"))
    with b2:
        show_chart(lambda: px.scatter(x=y_pred, y=resid, opacity=0.55, title="Residuals vs Predicted",
                                      labels={"x": "Predicted Performance", "y": "Residual"}).add_hline(y=0, line_dash="dash"))
    st.caption("Residual = Actual − Predicted.  **Positive** → the model *underestimated* the student. "
               "**Negative** → the model *overestimated* the student.")


def tab_predictions(art, passing):
    table = generate_predictions(art, passing)
    c1, c2 = st.columns([2, 1])
    sort_by = c1.selectbox("Sort by", ["Absolute Error", "Percentage Error (%)", "Residual",
                                       "Actual Performance", "Predicted Performance"])
    order = c2.radio("Order", ["Largest first", "Smallest first"], horizontal=True)
    table = table.sort_values(sort_by, ascending=(order == "Smallest first"))
    agree = float((table["Actual Result"] == table["Prediction Result"]).mean() * 100)
    st.caption(f"Predicted PASS/FAIL matches the actual PASS/FAIL (at passing score {passing:g}) "
               f"for **{agree:.1f}%** of test students.")
    st.dataframe(table, width="stretch")
    show_chart(lambda: px.histogram(table, x="Absolute Error", nbins=30, title="Prediction Error Distribution"))
    st.download_button("📥 Download Predictions CSV", to_csv(table, index=True), "predictions.csv", "text/csv")


def tab_predictor(art, passing):
    st.write("Enter the student's details. The form is generated from the features the model was trained on.")
    sig = f"{art['target']}_{'-'.join(art['features'])}_{art['n_rows']}"
    values = {}
    with st.form("student_form"):
        cols = st.columns(3)
        for i, feat in enumerate(art["features"]):
            with cols[i % 3]:
                if feat in art["num"]:
                    lo, hi, med, is_int = art["ranges"][feat]
                    values[feat] = st.number_input(
                        feat, value=float(round(med, 2)), min_value=0.0 if lo >= 0 else None,
                        step=1.0 if is_int else 0.1, format="%.0f" if is_int else "%.2f",
                        help=f"Training range: {lo:g} – {hi:g}", key=f"in_{sig}_{feat}")
                else:
                    opts = art["categories"][feat]
                    values[feat] = st.selectbox(feat, opts, index=opts.index(art["modes"][feat]),
                                                key=f"in_{sig}_{feat}")
        submitted = st.form_submit_button("🎯 Predict Student Performance", type="primary")
    if not submitted:
        return
    try:
        score = predict_student(values, art)
    except ValueError as e:
        st.error(f"⚠️ {e}")
        return
    except Exception:
        st.error("⚠️ The prediction could not be completed. Please check the input values.")
        return
    result, level = get_performance(score, passing)
    with st.container(border=True):
        st.markdown("### 📊 Predicted Performance")
        st.markdown(f"# {score:.2f}")
        st.markdown("### 🎓 FINAL RESULT")
        (st.success if result == "PASS" else st.error)(f"## {result}")
        k = st.columns(3)
        k[0].metric("Passing Score", f"{passing:g}")
        k[1].metric("Performance Status", result)
        k[2].metric("Performance Level", level)
    outside = [f for f in art["num"] if not art["ranges"][f][0] <= values[f] <= art["ranges"][f][1]]
    if outside:
        st.warning(f"⚠️ Outside the training range: {', '.join(outside)}. The prediction may be less reliable.")
    st.caption("This is a statistical estimate from a machine-learning model, not a guaranteed outcome.")


def tab_features(df, types, art):
    imp = art["importance"]
    show_chart(lambda: px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                              title="XGBoost Feature Importance"))
    st.caption("Feature importance shows how much the model *used* each feature to make predictions. "
               "It does **not** prove that a feature causes performance. Label-encoding keeps one column per "
               "original feature, so each bar maps directly to an original column.")
    st.dataframe(imp.round(4), hide_index=True, width="stretch")

    st.subheader("Explore features")
    feat = st.selectbox("Choose a feature", art["features"])
    a, b = st.columns(2)
    if feat in art["num"]:
        with a:
            show_chart(lambda: px.histogram(df, x=feat, nbins=30, title=f"Distribution of {feat}"))
        with b:
            show_chart(lambda: px.scatter(df, x=feat, y=art["target"], opacity=0.5, title=f"{feat} vs {art['target']}"))
    else:
        with a:
            show_chart(lambda: px.histogram(df, x=feat, title=f"Distribution of {feat}").update_xaxes(categoryorder="total descending"))
        with b:
            show_chart(lambda: px.box(df, x=feat, y=art["target"], title=f"{art['target']} by {feat}"))

    nums = art["num"] + [art["target"]]
    if len(nums) >= 2:
        show_chart(lambda: px.imshow(df[nums].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                     zmin=-1, zmax=1, title="Correlation Heatmap (numerical features)"))
    if art["cv_results"] is not None:
        top = art["cv_results"].head(10).copy()
        top["CV RMSE"] = -top["mean_test_score"]
        top["Model"] = [f"Model {i + 1}" for i in range(len(top))]
        show_chart(lambda: px.bar(top, x="CV RMSE", y="Model", orientation="h", title="Top 10 RandomizedSearchCV models"))


def tab_model_info(art, passing):
    st.subheader("Model")
    st.write("**XGBoost Regressor** (`objective='reg:squarederror'`)")
    st.write(f"**Target:** `{art['target']}`")
    st.write(f"**Features ({len(art['features'])}):** {', '.join(art['features'])}")
    steps = ["Dataset", "Data Cleaning (duplicates, missing, inf)", "Feature Selection",
             f"Train/Test Split ({int((1 - art['test_size']) * 100)}/{int(art['test_size'] * 100)}, seed {art['random_state']})"]
    if art["cat"]:
        steps.append("Label Encoding (fit on train)")
    if art["num"]:
        steps.append("Standard Scaling (fit on train)")
    steps += [f"Hyperparameter Tuning ({'RandomizedSearchCV' if art['tuning_mode'] != 'notebook' else 'RandomizedSearchCV — notebook result'})",
              "Best XGBoost Model", "Prediction (clipped to 0–100)" if art["clip_range"] else "Prediction",
              "Evaluation (MAE, MSE, RMSE, R²)", f"PASS / FAIL (score ≥ {passing:g})"]
    st.code("\n   ↓\n".join(steps), language=None)

    st.subheader("🏆 Best XGBoost Parameters")
    st.dataframe(pd.DataFrame({"Parameter": art["best_params"].keys(),
                               "Best Value": [str(v) for v in art["best_params"].values()]}),
                 hide_index=True, width="stretch")
    st.caption(f"{art['source']} · Cross-validation RMSE: {art['cv_rmse']:.4f}")
    with st.expander("Search space (RandomizedSearchCV)"):
        st.json(NB_PARAM_GRID)
        st.caption(f"Notebook settings: n_iter={NB_N_ITER}, cv={NB_CV}, scoring='{NB_SCORING}'.")
    m = art["metrics"]
    st.subheader("Evaluation (test set)")
    st.write(f"MAE **{m['MAE']:.3f}** · MSE **{m['MSE']:.3f}** · RMSE **{m['RMSE']:.3f}** · R² **{m['R2']:.4f}**")

    st.subheader("Beginner-friendly guide")
    with st.expander("What is XGBoost?", expanded=True):
        st.write("XGBoost is a *gradient boosting* algorithm. It builds many small decision trees one after another; "
                 "each new tree focuses on fixing the mistakes of the trees before it.")
    with st.expander("Why XGBoost for student performance?"):
        st.write("Student outcomes depend on combinations of habits (study time, attendance, earlier grades…). "
                 "XGBoost can learn non-linear relationships and interactions between features automatically.")
    glossary = {
        "Features / Target": "Features are the inputs (study hours, attendance…). The target is what we predict.",
        "Label Encoding": "Turns text categories (e.g. Public/Private) into numbers. Learned on training data only.",
        "StandardScaler": "Rescales numeric features to mean 0 and spread 1 so they are on the same scale.",
        "Boosting / Trees": "Many shallow decision trees added one by one, each correcting the previous errors.",
        "Learning rate": "How big a step each new tree takes. Smaller = slower but often more careful.",
        "Max depth": "How many questions a tree may ask in a row. Deeper trees can overfit.",
        "Number of estimators": "How many trees are built.",
        "Hyperparameter tuning": "Trying many settings (RandomizedSearchCV + cross-validation) and keeping the best.",
        "Prediction & residual": "Prediction is the model's estimate. Residual = actual − predicted.",
        "MAE / MSE / RMSE / R²": "See the *Model Performance* tab for plain-language definitions.",
    }
    for k, v in glossary.items():
        if k.startswith("Label") and not art["cat"]:
            continue
        if k.startswith("Standard") and not art["num"]:
            continue
        with st.expander(k):
            st.write(v)


# =====================================================================
# 8. MAIN APP
# =====================================================================
def main():
    st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
    st.title("🎓 Student Performance Prediction System")
    st.caption("Machine Learning based Student Performance Prediction using XGBoost")

    sb = st.sidebar
    sb.header("Project")
    sb.write("Student Performance Prediction")
    sb.write("**Model:** XGBoost")
    with sb.expander("👨‍💻 Author", expanded=True):
        st.markdown(f"**{AUTHOR_NAME}**  \n"
                    f"[GitHub]({AUTHOR_GITHUB})  ·  [LinkedIn]({AUTHOR_LINKEDIN})")

    # ---------- dataset ----------
    sb.subheader("📂 Dataset")
    uploaded = sb.file_uploader("Upload Student Performance Dataset", type=["csv"])
    if uploaded is not None:
        data_bytes = uploaded.getvalue()
        raw, err = load_uploaded_dataset(data_bytes)
        if err:
            st.error(err)
            st.stop()
        source, sig = uploaded.name, hashlib.md5(data_bytes).hexdigest()[:10]
    else:
        raw = load_default_dataset()
        if raw is None:
            st.error("⚠️ Default dataset `student_performance_prediction_dataset.csv` was not found next to app.py. "
                     "Please upload a CSV.")
            st.stop()
        source, sig = "Default dataset", "default"
    sb.caption(f"Using: **{source}**")

    df, raw_report = clean_dataset(raw)
    if df.shape[0] == 0 or df.shape[1] < 2:
        st.error("⚠️ The dataset has no usable rows/columns after cleaning.")
        st.stop()
    df, types, notes = detect_feature_types(df)
    for n in notes:
        st.warning(n)

    # ---------- target ----------
    sb.subheader("🧩 Feature Selection")
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if not numeric_cols:
        st.error("⚠️ No suitable target column was detected (a numeric performance/score column is required).")
        st.stop()
    target_guess, cands = detect_target_column(df, types)
    if target_guess is None:
        sb.warning("⚠️ Please select the correct target column." if cands else
                   "⚠️ No suitable target column was detected. Please select one.")
    target = sb.selectbox("🎯 Select Target Column", numeric_cols,
                          index=numeric_cols.index(target_guess) if target_guess else None,
                          placeholder="Choose the performance/score column", key=f"target_{sig}")
    if target is None:
        st.info("👈 Select the target column in the sidebar to continue.")
        st.stop()

    # ---------- features ----------
    ids = detect_id_columns(df)
    leak = detect_leakage_columns(df, types, target)
    id_sel = sb.multiselect("Identifier columns (excluded)", [c for c in df.columns if c != target],
                            default=[c for c in ids if c != target], key=f"ids_{sig}_{target}")
    num_opts = [c for c in numeric_cols if c != target and c not in id_sel]
    cat_opts = [c for c, t in types.items() if t in ("categorical", "boolean") and c not in id_sel]
    num_sel = sb.multiselect("Numerical features", num_opts, default=[c for c in num_opts if c not in leak],
                             key=f"num_{sig}_{target}")
    cat_sel = sb.multiselect("Categorical features", cat_opts, default=[c for c in cat_opts if c not in leak],
                             key=f"cat_{sig}_{target}")
    if leak:
        sb.caption(f"⚠️ Possible target leakage (unselected by default): {', '.join(leak)}")

    # ---------- settings ----------
    sb.subheader("⚙️ Model Settings")
    test_size = sb.slider("Test Size", 0.10, 0.40, NB_TEST_SIZE, 0.05)
    random_state = int(sb.number_input("Random State", 0, 10_000, NB_RANDOM_STATE, 1))
    is_nb = (sig == "default" and target == NB_TARGET)
    tuning_label = sb.radio("Hyperparameters",
                            ["Use notebook's best parameters", "Re-run RandomizedSearchCV"],
                            index=0 if is_nb else 1, key=f"tune_{sig}_{target}",
                            help="Re-running follows the notebook: n_iter=40, cv=5, RMSE scoring. It can take a minute or two.")
    tuning_mode = "notebook" if tuning_label.startswith("Use") else "retune"
    n_iter, cv = NB_N_ITER, NB_CV
    if tuning_mode == "retune":
        n_iter = int(sb.number_input("Search iterations (n_iter)", 5, 100, NB_N_ITER, 5))
        cv = int(sb.number_input("CV folds", 2, 10, NB_CV, 1))

    sb.subheader("🎯 Prediction Settings")
    passing = float(sb.number_input("Passing Score", value=float(NB_PASSING_SCORE), step=1.0,
                                    help="Notebook default: score ≥ 40 → PASS"))
    col_name, cutoff = infer_dataset_cutoff(df, target)
    if cutoff is not None:
        sb.caption(f"ℹ️ The dataset's own `{col_name}` column implies a cutoff of about {cutoff:g}. "
                   f"The notebook uses {NB_PASSING_SCORE}.")

    sb.subheader("🚀 Train Model")
    run_clicked = sb.button("🚀 Train / Run Model", type="primary", width="stretch")

    # ---------- validate + train ----------
    features = [c for c in df.columns if c in num_sel + cat_sel and c not in id_sel]
    cfg = (sig, target, tuple(num_sel), tuple(cat_sel), tuple(id_sel), test_size, random_state, tuning_mode, n_iter, cv)
    needs_run = run_clicked or st.session_state.get("trained_sig") != sig or "art" not in st.session_state
    if needs_run:
        errors, warns = validate_dataset(df, target, features)
        for w in warns:
            st.info(w)
        if errors:
            st.session_state.pop("art", None)
            for e in errors:
                st.error(e)
            st.stop()
        label = "Training XGBoost..." if tuning_mode == "notebook" else "Running RandomizedSearchCV + training XGBoost (this can take a minute)..."
        try:
            with st.spinner(label):
                nc = tuple(c for c in features if c in num_sel)
                cc = tuple(c for c in features if c in cat_sel)
                art = train_model(df[features + [target]], target, nc, cc, float(test_size), random_state,
                                  tuning_mode, n_iter, cv)
        except Exception as e:  # never show a traceback to the user
            st.session_state.pop("art", None)
            st.error(f"⚠️ Dataset is incompatible with the model or training failed: {str(e)[:200]}")
            st.stop()
        st.session_state.update(art=art, cfg=cfg, trained_sig=sig, df=df)
    art = st.session_state["art"]
    if st.session_state.get("cfg") != cfg:
        st.info("⚙️ Settings changed — click **🚀 Train / Run Model** to apply them. "
                "(Showing results for the previous settings.)")
    if sum(art["unseen"].values()):
        st.warning("⚠️ Some test rows contained categories not seen in training; they were encoded as −1.")

    # ---------- metric cards ----------
    m = art["metrics"]
    k = st.columns(4)
    k[0].metric("TOTAL STUDENTS", f"{art['n_rows']:,}")
    k[1].metric("FEATURES USED", len(art["features"]))
    k[2].metric("RMSE", f"{m['RMSE']:.3f}")
    k[3].metric("R² SCORE", f"{m['R2']:.4f}")

    tabs = st.tabs(["📊 Overview", "📁 Dataset", "📈 Model Performance", "🎯 Predictions",
                    "🎓 Student Predictor", "🔍 Feature Analysis", "⚙️ Model Information"])
    with tabs[0]:
        tab_overview(df, types, art, passing)
    with tabs[1]:
        tab_dataset(raw_report, df, types, art, features, ids, leak)
    with tabs[2]:
        tab_performance(art)
        st.download_button("📥 Download Model Evaluation", create_download_file(art, cfg, passing),
                           "model_evaluation.json", "application/json")
    with tabs[3]:
        tab_predictions(art, passing)
    with tabs[4]:
        tab_predictor(art, passing)
    with tabs[5]:
        tab_features(df, types, art)
    with tabs[6]:
        tab_model_info(art, passing)

    st.divider()
    st.caption(f"Built by **{AUTHOR_NAME}** · [GitHub]({AUTHOR_GITHUB}) · [LinkedIn]({AUTHOR_LINKEDIN})")


if __name__ == "__main__":
    main()
