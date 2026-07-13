import streamlit as st
import numpy as np
import pickle
import os

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
    <div class="hero-sub">Enter the 10 A-FGO-optimized morphological features extracted from a fine needle aspirate (FNA) biopsy image to classify the tumor as Benign or Malignant.</div>
</div>
""", unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error(f"⚠️ Failed to load model files from `{MODEL_DIR}/`. Ensure all `.pkl` files are present in the repository.\n\n`{LOAD_ERROR}`")
    st.stop()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
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
# PREDICTION
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
