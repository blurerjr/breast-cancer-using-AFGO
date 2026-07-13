import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection · A-FGO",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}

[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7eb8f7;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a2540 0%, #0f1520 60%, #0d1a2e 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 2rem 2rem 1.6rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(126,184,247,0.08) 0%, transparent 70%);
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7eb8f7;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f0f4ff;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: 0.88rem;
    color: #8090a8;
    line-height: 1.6;
    max-width: 520px;
}

/* Section headers */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7eb8f7;
    border-left: 3px solid #7eb8f7;
    padding-left: 0.6rem;
    margin: 1.5rem 0 1rem 0;
}

/* Result card */
.result-card {
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-malignant {
    background: linear-gradient(135deg, #2a1018, #1e0d14);
    border: 1px solid #7f1d1d;
}
.result-benign {
    background: linear-gradient(135deg, #0d2018, #091a12);
    border: 1px solid #14532d;
}
.result-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.result-diagnosis {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.4rem 0;
}
.result-malignant .result-diagnosis { color: #f87171; }
.result-benign .result-diagnosis { color: #4ade80; }
.result-malignant .result-title { color: #fca5a5; }
.result-benign .result-title { color: #86efac; }
.result-note {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.8rem;
    line-height: 1.5;
}

/* Probability bar */
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    margin: 0.3rem 0;
    color: #9ca3af;
}
.prob-label { font-weight: 500; color: #d1d5db; }

/* Feature badge */
.feature-badge {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7eb8f7;
    margin: 0.15rem;
}

/* Model selector pills */
.stRadio > div {
    gap: 0.5rem;
}

/* Input labels */
.stNumberInput label, .stSlider label {
    font-size: 0.82rem !important;
    color: #9ca3af !important;
    font-weight: 500 !important;
}

/* Disclaimer */
.disclaimer {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.76rem;
    color: #6b7280;
    line-height: 1.6;
    margin-top: 1.5rem;
}
.disclaimer strong { color: #9ca3af; }

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1e2535;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS & METADATA
# ─────────────────────────────────────────────
MODEL_DIR = "AFGO_Complete_Model_Bundle"

@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, "pipeline_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "decision_tree_model.pkl"), "rb") as f:
        dt = pickle.load(f)
    return meta, lr, rf, dt

try:
    meta, lr_model, rf_model, dt_model = load_assets()
    scaler = meta["scaler_transform_state"]
    selected_indices = meta["afgo_selected_indices"]
    selected_names = meta["afgo_feature_names"]
    total_features = meta["total_baseline_features"]
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)

# ─────────────────────────────────────────────
# FEATURE METADATA (ranges from WBCD dataset)
# ─────────────────────────────────────────────
FEATURE_META = {
    "area_mean": {
        "label": "Area (Mean)",
        "min": 143.5, "max": 2501.0, "default": 654.9,
        "step": 0.1,
        "help": "Mean nuclear area of digitized fine needle aspirate cells"
    },
    "concave points_mean": {
        "label": "Concave Points (Mean)",
        "min": 0.0, "max": 0.2, "default": 0.048,
        "step": 0.001,
        "help": "Mean number of concave portions of the cell nucleus contour"
    },
    "radius_se": {
        "label": "Radius (SE)",
        "min": 0.1, "max": 2.9, "default": 0.405,
        "step": 0.001,
        "help": "Standard error of cell nucleus radius"
    },
    "smoothness_se": {
        "label": "Smoothness (SE)",
        "min": 0.001, "max": 0.031, "default": 0.007,
        "step": 0.0001,
        "help": "Standard error of local variation in radius lengths"
    },
    "concave points_se": {
        "label": "Concave Points (SE)",
        "min": 0.0, "max": 0.053, "default": 0.015,
        "step": 0.0001,
        "help": "Standard error of number of concave contour portions"
    },
    "texture_worst": {
        "label": "Texture (Worst)",
        "min": 12.0, "max": 49.5, "default": 25.7,
        "step": 0.01,
        "help": "Worst (largest) value for standard deviation of gray-scale values"
    },
    "perimeter_worst": {
        "label": "Perimeter (Worst)",
        "min": 50.4, "max": 251.2, "default": 107.3,
        "step": 0.01,
        "help": "Worst (largest) value for nucleus perimeter"
    },
    "smoothness_worst": {
        "label": "Smoothness (Worst)",
        "min": 0.071, "max": 0.223, "default": 0.132,
        "step": 0.0001,
        "help": "Worst value for local variation in radius lengths"
    },
    "concave points_worst": {
        "label": "Concave Points (Worst)",
        "min": 0.0, "max": 0.291, "default": 0.114,
        "step": 0.0001,
        "help": "Worst value for number of concave portions of the contour"
    },
    "symmetry_worst": {
        "label": "Symmetry (Worst)",
        "min": 0.156, "max": 0.664, "default": 0.290,
        "step": 0.001,
        "help": "Worst value for cell nucleus symmetry"
    },
}

# Ranked by A-FGO significance (master rank order)
FEATURE_RANK_ORDER = [
    "perimeter_worst", "concave points_worst", "area_mean",
    "concave points_mean", "radius_se", "texture_worst",
    "smoothness_worst", "symmetry_worst", "concave points_se", "smoothness_se"
]

MODEL_INFO = {
    "Logistic Regression": {
        "key": "lr",
        "accuracy": "98.25%",
        "roc_auc": "~99%",
        "description": "Highest accuracy. Fast, interpretable linear model. Best for confident single-result output.",
        "icon": "⭐",
    },
    "Random Forest": {
        "key": "rf",
        "accuracy": "97.66%",
        "roc_auc": "~99%",
        "description": "Ensemble of 100 decision trees. Robust to noise. Provides reliable probability estimates.",
        "icon": "🌲",
    },
    "Decision Tree": {
        "key": "dt",
        "accuracy": "95.32%",
        "roc_auc": "~97%",
        "description": "Single interpretable tree (max depth 5). Fastest inference. Lower accuracy but most explainable.",
        "icon": "🌿",
    },
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 A-FGO System")
    st.markdown("<hr style='border-color:#1e2535;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    st.markdown("### Model Selection")
    chosen_model_name = st.radio(
        "Choose classifier",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['icon']}  {x}",
        label_visibility="collapsed",
    )

    info = MODEL_INFO[chosen_model_name]
    st.markdown(f"""
    <div style="background:#1a2030;border:1px solid #1e3050;border-radius:8px;padding:0.8rem 1rem;margin-top:0.6rem;font-size:0.78rem;color:#8090a8;line-height:1.6">
        <div style="color:#7eb8f7;font-weight:600;margin-bottom:0.3rem">{chosen_model_name}</div>
        <div>Accuracy: <span style="color:#e8eaf0;font-weight:500">{info['accuracy']}</span></div>
        <div>ROC-AUC: <span style="color:#e8eaf0;font-weight:500">{info['roc_auc']}</span></div>
        <div style="margin-top:0.5rem">{info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2535;margin:1.2rem 0'>", unsafe_allow_html=True)

    st.markdown("### A-FGO Feature Set")
    st.markdown(
        "<div style='font-size:0.75rem;color:#6b7280;margin-bottom:0.5rem'>"
        f"10 of 30 features selected by the optimizer</div>",
        unsafe_allow_html=True
    )
    badges = "".join([f'<span class="feature-badge">#{i+1} {name}</span>' for i, name in enumerate(FEATURE_RANK_ORDER)])
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2535;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#4b5563;line-height:1.6'>
        Dataset: Wisconsin Breast Cancer Dataset (WBCD)<br>
        Samples: 569 · Features: 30 → 10<br>
        Algorithm: Adaptive Fire-Grey Optimizer
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">A-FGO · Breast Cancer Detection System</div>
    <div class="hero-title">Morphological Cell Analysis</div>
    <div class="hero-sub">Classify fine needle aspirate (FNA) biopsy measurements as Benign or Malignant using A-FGO-optimized features. Run a single manual analysis or upload a full 30-feature WBCD CSV for batch screening.</div>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error(f"⚠️ Failed to load model files from `{MODEL_DIR}/`. Ensure all `.pkl` files are present in the repository.\n\n`{LOAD_ERROR}`")
    st.stop()

# ─────────────────────────────────────────────
# ALL WBCD FEATURE NAMES (30 columns, canonical order)
# ─────────────────────────────────────────────
ALL_WBCD_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

def run_batch_prediction(df_raw, model, scaler, selected_indices, total_features):
    """
    Accepts a raw 30-column WBCD dataframe.
    Drops non-feature columns (id, diagnosis, Unnamed: 32) if present,
    builds the full 30-feature matrix, scales it, then slices the
    A-FGO selected columns for inference.
    """
    drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df_raw.columns]
    label_col = df_raw["diagnosis"].copy() if "diagnosis" in df_raw.columns else None
    df = df_raw.drop(columns=drop_cols)

    # Validate columns
    missing = [c for c in ALL_WBCD_FEATURES if c not in df.columns]
    if missing:
        return None, None, None, f"Missing required columns: {missing}"

    # Reorder to canonical WBCD order and fill NaNs with column mean
    X_raw = df[ALL_WBCD_FEATURES].copy()
    if X_raw.isnull().any().any():
        X_raw = X_raw.fillna(X_raw.mean())

    X_scaled = scaler.transform(X_raw.values)
    X_selected = X_scaled[:, selected_indices]

    preds = model.predict(X_selected)
    probas = model.predict_proba(X_selected)

    return preds, probas, label_col, None

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["🔬  Single Analysis", "📂  Batch CSV Analysis"])

# ══════════════════════════════════════════════
# TAB 1 — MANUAL INPUT
# ══════════════════════════════════════════════
with tab_manual:
    st.markdown('<div class="section-label">Cell Morphology Measurements</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.8rem;color:#6b7280;margin-bottom:1.2rem'>"
        "Features are ordered by A-FGO significance rank (most influential first).</div>",
        unsafe_allow_html=True
    )

    input_values = {}

    col1, col2 = st.columns(2)
    for i, fname in enumerate(FEATURE_RANK_ORDER):
        m = FEATURE_META[fname]
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            val = st.number_input(
                label=f"#{i+1} · {m['label']}",
                min_value=float(m["min"]),
                max_value=float(m["max"]),
                value=float(m["default"]),
                step=float(m["step"]),
                format="%.4f" if m["step"] < 0.01 else "%.2f",
                help=m["help"],
                key=fname,
            )
            input_values[fname] = val

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SINGLE PREDICTION
    # ─────────────────────────────────────────────
    predict_btn = st.button("Run Diagnosis", use_container_width=True, type="primary")

    if predict_btn:
        # Build full 30-feature vector (zeros for non-selected features)
        full_vector = np.zeros((1, total_features))
        for fname, val in input_values.items():
            idx = selected_names.index(fname)
            full_col = selected_indices[idx]
            full_vector[0, full_col] = val

        # Scale using fitted scaler
        full_scaled = scaler.transform(full_vector)

        # Extract only the selected feature columns
        X_input = full_scaled[:, selected_indices]

        # Select model
        model_map = {"Logistic Regression": lr_model, "Random Forest": rf_model, "Decision Tree": dt_model}
        model = model_map[chosen_model_name]

        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        prob_benign = probabilities[0]
        prob_malignant = probabilities[1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-malignant">
                <div class="result-title">Tumor Classification Result</div>
                <div class="result-diagnosis">⚠ MALIGNANT</div>
                <div style="color:#fca5a5;font-size:0.88rem;font-weight:500">The morphological profile is consistent with a malignant tumor.</div>
                <div style="margin-top:1rem">
                    <div class="prob-row"><span class="prob-label">Malignant probability</span><span style="color:#f87171;font-weight:600">{prob_malignant*100:.1f}%</span></div>
                    <div class="prob-row"><span class="prob-label">Benign probability</span><span>{prob_benign*100:.1f}%</span></div>
                </div>
                <div class="result-note">⚕ This result is for research and educational purposes only. Clinical diagnosis must be confirmed by a qualified medical professional.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-benign">
                <div class="result-title">Tumor Classification Result</div>
                <div class="result-diagnosis">✓ BENIGN</div>
                <div style="color:#86efac;font-size:0.88rem;font-weight:500">The morphological profile is consistent with a benign tumor.</div>
                <div style="margin-top:1rem">
                    <div class="prob-row"><span class="prob-label">Benign probability</span><span style="color:#4ade80;font-weight:600">{prob_benign*100:.1f}%</span></div>
                    <div class="prob-row"><span class="prob-label">Malignant probability</span><span>{prob_malignant*100:.1f}%</span></div>
                </div>
                <div class="result-note">⚕ This result is for research and educational purposes only. Clinical diagnosis must be confirmed by a qualified medical professional.</div>
            </div>
            """, unsafe_allow_html=True)

        # Feature contribution summary
        st.markdown("<div class='section-label' style='margin-top:1.8rem'>Input Summary</div>", unsafe_allow_html=True)
        summary_rows = ""
        for i, fname in enumerate(FEATURE_RANK_ORDER):
            m = FEATURE_META[fname]
            val = input_values[fname]
            summary_rows += f"""
            <tr>
                <td style="color:#6b7280;padding:0.3rem 0.8rem 0.3rem 0;font-size:0.78rem">#{i+1}</td>
                <td style="color:#d1d5db;padding:0.3rem 0.8rem;font-size:0.8rem">{m['label']}</td>
                <td style="color:#7eb8f7;font-family:'JetBrains Mono',monospace;font-size:0.78rem;text-align:right;padding:0.3rem 0 0.3rem 0.8rem">{val:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;margin-top:0.5rem">
            <thead>
                <tr style="border-bottom:1px solid #1e2535">
                    <th style="color:#4b5563;font-size:0.72rem;text-align:left;padding-bottom:0.4rem">Rank</th>
                    <th style="color:#4b5563;font-size:0.72rem;text-align:left;padding-bottom:0.4rem">Feature</th>
                    <th style="color:#4b5563;font-size:0.72rem;text-align:right;padding-bottom:0.4rem">Value</th>
                </tr>
            </thead>
            <tbody>{summary_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BATCH CSV ANALYSIS
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-label">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#6b7280;line-height:1.7;margin-bottom:1.2rem">
        Upload a CSV containing the full 30 WBCD feature columns. The system will automatically
        extract the 10 A-FGO features, apply the trained scaler, and classify each row.
        Columns like <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">id</code>,
        <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">diagnosis</code>, and
        <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">Unnamed: 32</code>
        are ignored automatically if present.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        help="Must contain the 30 standard WBCD feature columns. Extra columns (id, diagnosis) are handled automatically.",
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # Preview uploaded file
        st.markdown(f"""
        <div style="background:#1a2030;border:1px solid #1e3050;border-radius:8px;
                    padding:0.7rem 1rem;margin:0.8rem 0;font-size:0.8rem;color:#8090a8">
            📄 <span style="color:#e8eaf0;font-weight:500">{uploaded_file.name}</span>
            &nbsp;·&nbsp; {len(df_raw):,} rows &nbsp;·&nbsp; {df_raw.shape[1]} columns detected
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview uploaded data (first 5 rows)", expanded=False):
            st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        run_batch_btn = st.button("Run Batch Analysis", use_container_width=True, type="primary", key="batch_btn")

        if run_batch_btn:
            model_map = {"Logistic Regression": lr_model, "Random Forest": rf_model, "Decision Tree": dt_model}
            model = model_map[chosen_model_name]

            with st.spinner("Running A-FGO pipeline on all rows..."):
                preds, probas, label_col, error = run_batch_prediction(
                    df_raw, model, scaler, selected_indices, total_features
                )

            if error:
                st.error(f"⚠️ {error}")
            else:
                # ── Summary metrics ──────────────────────────────
                n_total = len(preds)
                n_malignant = int(np.sum(preds == 1))
                n_benign = int(np.sum(preds == 0))
                pct_m = n_malignant / n_total * 100
                pct_b = n_benign / n_total * 100

                # Ground truth accuracy if diagnosis column was present
                accuracy_block = ""
                if label_col is not None:
                    gt = label_col.map({"M": 1, "B": 0}).values if label_col.dtype == object else label_col.values
                    correct = int(np.sum(preds == gt))
                    acc = correct / n_total * 100
                    accuracy_block = f"""
                    <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #1e2535">
                        <div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">vs Ground Truth Labels</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#7eb8f7">{acc:.2f}%
                            <span style="font-size:0.8rem;font-weight:400;color:#6b7280"> accuracy &nbsp;({correct}/{n_total} correct)</span>
                        </div>
                    </div>"""

                st.markdown(f"""
                <div style="background:#1a2030;border:1px solid #1e3050;border-radius:10px;
                            padding:1.2rem 1.5rem;margin:1rem 0">
                    <div style="font-size:0.72rem;color:#7eb8f7;font-weight:600;letter-spacing:0.1em;
                                text-transform:uppercase;margin-bottom:0.8rem">Batch Analysis Summary · {chosen_model_name}</div>
                    <div style="display:flex;gap:2rem;flex-wrap:wrap">
                        <div>
                            <div style="font-size:0.72rem;color:#6b7280;margin-bottom:0.2rem">Total Samples</div>
                            <div style="font-size:1.6rem;font-weight:700;color:#e8eaf0">{n_total:,}</div>
                        </div>
                        <div>
                            <div style="font-size:0.72rem;color:#6b7280;margin-bottom:0.2rem">Benign</div>
                            <div style="font-size:1.6rem;font-weight:700;color:#4ade80">{n_benign:,}
                                <span style="font-size:0.8rem;font-weight:400;color:#6b7280"> ({pct_b:.1f}%)</span>
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.72rem;color:#6b7280;margin-bottom:0.2rem">Malignant</div>
                            <div style="font-size:1.6rem;font-weight:700;color:#f87171">{n_malignant:,}
                                <span style="font-size:0.8rem;font-weight:400;color:#6b7280"> ({pct_m:.1f}%)</span>
                            </div>
                        </div>
                    </div>
                    {accuracy_block}
                </div>
                """, unsafe_allow_html=True)

                # ── Build results dataframe ──────────────────────
                results_df = df_raw.copy()

                # Add id column if not present
                if "id" not in results_df.columns:
                    results_df.insert(0, "id", range(1, len(results_df) + 1))

                results_df["A_FGO_Prediction"] = ["MALIGNANT" if p == 1 else "BENIGN" for p in preds]
                results_df["Prob_Benign (%)"] = np.round(probas[:, 0] * 100, 2)
                results_df["Prob_Malignant (%)"] = np.round(probas[:, 1] * 100, 2)

                # ── Results table (styled) ───────────────────────
                st.markdown('<div class="section-label" style="margin-top:1.5rem">Per-Row Results</div>', unsafe_allow_html=True)

                display_cols = ["id", "A_FGO_Prediction", "Prob_Benign (%)", "Prob_Malignant (%)"]
                if "diagnosis" in results_df.columns:
                    display_cols.insert(2, "diagnosis")

                display_df = results_df[display_cols].copy()

                def color_prediction(val):
                    if val == "MALIGNANT":
                        return "color: #f87171; font-weight: 600"
                    elif val == "BENIGN":
                        return "color: #4ade80; font-weight: 600"
                    return ""

                styled = display_df.style.applymap(color_prediction, subset=["A_FGO_Prediction"])
                st.dataframe(styled, use_container_width=True, height=380)

                # ── Download button ──────────────────────────────
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Full Results CSV",
                    data=csv_out,
                    file_name=f"afgo_batch_results_{chosen_model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.markdown("""
                <div style="font-size:0.75rem;color:#4b5563;margin-top:0.6rem;text-align:center">
                    Downloaded CSV contains all original columns plus A_FGO_Prediction, Prob_Benign (%), and Prob_Malignant (%).
                </div>
                """, unsafe_allow_html=True)

    else:
        # Empty state guidance
        st.markdown("""
        <div style="border:1px dashed #2d3748;border-radius:10px;padding:2rem;text-align:center;margin-top:0.5rem">
            <div style="font-size:2rem;margin-bottom:0.6rem">📂</div>
            <div style="color:#6b7280;font-size:0.85rem;line-height:1.7">
                Upload a CSV with the full 30 WBCD feature columns.<br>
                The A-FGO pipeline will extract the correct 10 features, apply scaling, and return predictions for every row.<br><br>
                <span style="color:#4b5563;font-size:0.78rem">
                    Expected columns include: <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">radius_mean</code>,
                    <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">texture_mean</code>,
                    <code style="color:#7eb8f7;background:#1e2535;padding:0.1rem 0.3rem;border-radius:3px">perimeter_mean</code> … and all remaining WBCD features.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <strong>⚕ Medical Disclaimer</strong><br>
    This system is a machine learning research prototype trained on the Wisconsin Breast Cancer Dataset. 
    It is intended for educational and research purposes only. Results must not be used as a substitute 
    for professional clinical judgment, pathological examination, or any form of licensed medical diagnosis.
    Always consult a qualified healthcare professional for any medical decisions.
</div>
""", unsafe_allow_html=True)
