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
# CUSTOM CSS  –  bright, high-contrast theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #f0f4f8; color: #1a202c; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1e3a5f !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e2eaf4 !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #93c5fd !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr { border-color: #2d5080 !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e4976 60%, #155e8e 100%);
    border-radius: 14px;
    padding: 2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px rgba(30,58,95,0.18);
}
.hero-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #93c5fd; margin-bottom: 0.5rem;
}
.hero-title { font-size: 1.8rem; font-weight: 700; color: #ffffff; line-height: 1.2; margin-bottom: 0.6rem; }
.hero-sub   { font-size: 0.88rem; color: #bfdbfe; line-height: 1.7; max-width: 540px; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #1e4976;
    border-left: 3px solid #3b82f6;
    padding-left: 0.6rem; margin: 1.6rem 0 1rem 0;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 0.8rem 0;
    box-shadow: 0 1px 6px rgba(30,58,95,0.07);
}

/* ── Result cards ── */
.result-card { border-radius: 12px; padding: 1.6rem 1.8rem; margin-top: 1.4rem; text-align: center; }
.result-malignant { background: #fff1f2; border: 2px solid #fca5a5; }
.result-benign    { background: #f0fdf4; border: 2px solid #86efac; }
.result-title     { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }
.result-diagnosis { font-size: 2.1rem; font-weight: 800; margin: 0.4rem 0; }
.result-malignant .result-title     { color: #b91c1c; }
.result-malignant .result-diagnosis { color: #dc2626; }
.result-benign .result-title        { color: #15803d; }
.result-benign .result-diagnosis    { color: #16a34a; }
.result-note { font-size: 0.76rem; color: #6b7280; margin-top: 0.9rem; line-height: 1.5; }

/* ── Prob rows ── */
.prob-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.83rem; margin: 0.3rem 0; color: #374151;
}
.prob-label { font-weight: 500; }

/* ── Summary card (batch) ── */
.summary-card {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08);
}
.summary-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #1e4976; margin-bottom: 0.9rem;
}
.stat-label { font-size: 0.72rem; color: #6b7280; margin-bottom: 0.2rem; font-weight: 500; }
.stat-value { font-size: 1.65rem; font-weight: 800; }
.stat-sub   { font-size: 0.78rem; font-weight: 400; color: #6b7280; }

/* ── Accuracy block ── */
.accuracy-block {
    margin-top: 0.9rem; padding-top: 0.9rem;
    border-top: 1px solid #dbeafe;
}
.accuracy-label {
    font-size: 0.7rem; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.25rem; font-weight: 600;
}
.accuracy-value { font-size: 1.55rem; font-weight: 800; color: #1e4976; }
.accuracy-sub   { font-size: 0.8rem; font-weight: 400; color: #6b7280; }

/* ── Feature badge ── */
.feature-badge {
    display: inline-block;
    background: #dbeafe; border: 1px solid #93c5fd;
    border-radius: 6px; padding: 0.2rem 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #1e4976; margin: 0.15rem;
}

/* ── Input labels ── */
.stNumberInput label { font-size: 0.82rem !important; color: #374151 !important; font-weight: 500 !important; }

/* ── Disclaimer ── */
.disclaimer {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.76rem; color: #78350f; line-height: 1.6; margin-top: 1.5rem;
}
.disclaimer strong { color: #92400e; }

/* ── Divider ── */
.custom-divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.4rem 0; }

/* ── Empty state ── */
.empty-state {
    border: 2px dashed #bfdbfe; border-radius: 12px;
    padding: 2rem; text-align: center; background: #f8faff;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 2px solid #dbeafe; }
.stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; padding: 0.5rem 1.1rem; color: #4b5563; }
.stTabs [aria-selected="true"] { color: #1e4976 !important; border-bottom: 2px solid #1e4976 !important; }
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
    scaler          = meta["scaler_transform_state"]
    selected_indices = meta["afgo_selected_indices"]
    selected_names   = meta["afgo_feature_names"]
    total_features   = meta["total_baseline_features"]
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ALL_WBCD_FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
]

FEATURE_META = {
    "area_mean":             {"label":"Area (Mean)",             "min":143.5, "max":2501.0,"default":654.9, "step":0.1,  "help":"Mean nuclear area of digitized FNA cells"},
    "concave points_mean":   {"label":"Concave Points (Mean)",   "min":0.0,   "max":0.2,   "default":0.048, "step":0.001,"help":"Mean number of concave portions of the nucleus contour"},
    "radius_se":             {"label":"Radius (SE)",             "min":0.1,   "max":2.9,   "default":0.405, "step":0.001,"help":"Standard error of cell nucleus radius"},
    "smoothness_se":         {"label":"Smoothness (SE)",         "min":0.001, "max":0.031, "default":0.007, "step":0.0001,"help":"Standard error of local variation in radius lengths"},
    "concave points_se":     {"label":"Concave Points (SE)",     "min":0.0,   "max":0.053, "default":0.015, "step":0.0001,"help":"Standard error of number of concave contour portions"},
    "texture_worst":         {"label":"Texture (Worst)",         "min":12.0,  "max":49.5,  "default":25.7,  "step":0.01, "help":"Worst value for standard deviation of gray-scale values"},
    "perimeter_worst":       {"label":"Perimeter (Worst)",       "min":50.4,  "max":251.2, "default":107.3, "step":0.01, "help":"Worst value for nucleus perimeter"},
    "smoothness_worst":      {"label":"Smoothness (Worst)",      "min":0.071, "max":0.223, "default":0.132, "step":0.0001,"help":"Worst value for local variation in radius lengths"},
    "concave points_worst":  {"label":"Concave Points (Worst)",  "min":0.0,   "max":0.291, "default":0.114, "step":0.0001,"help":"Worst value for number of concave portions of the contour"},
    "symmetry_worst":        {"label":"Symmetry (Worst)",        "min":0.156, "max":0.664, "default":0.290, "step":0.001, "help":"Worst value for cell nucleus symmetry"},
}

FEATURE_RANK_ORDER = [
    "perimeter_worst","concave points_worst","area_mean","concave points_mean",
    "radius_se","texture_worst","smoothness_worst","symmetry_worst","concave points_se","smoothness_se",
]

MODEL_INFO = {
    "Logistic Regression": {"accuracy":"98.25%","roc_auc":"~99%","icon":"⭐",
        "description":"Highest accuracy. Fast, interpretable linear model."},
    "Random Forest":       {"accuracy":"97.66%","roc_auc":"~99%","icon":"🌲",
        "description":"Ensemble of 100 trees. Robust, reliable probabilities."},
    "Decision Tree":       {"accuracy":"95.32%","roc_auc":"~97%","icon":"🌿",
        "description":"Single tree (depth 5). Fastest, most explainable."},
}

# ─────────────────────────────────────────────
# BATCH HELPER
# ─────────────────────────────────────────────
def run_batch_prediction(df_raw, model, scaler, selected_indices, total_features):
    drop_cols = [c for c in ["id","diagnosis","Unnamed: 32"] if c in df_raw.columns]
    label_col = df_raw["diagnosis"].copy() if "diagnosis" in df_raw.columns else None
    df = df_raw.drop(columns=drop_cols)
    missing = [c for c in ALL_WBCD_FEATURES if c not in df.columns]
    if missing:
        return None, None, None, f"Missing required columns: {missing}"
    X_raw = df[ALL_WBCD_FEATURES].copy()
    if X_raw.isnull().any().any():
        X_raw = X_raw.fillna(X_raw.mean())
    X_scaled   = scaler.transform(X_raw.values)
    X_selected = X_scaled[:, selected_indices]
    preds  = model.predict(X_selected)
    probas = model.predict_proba(X_selected)
    return preds, probas, label_col, None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 A-FGO System")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Model Selection")
    chosen_model_name = st.radio(
        "Choose classifier",
        options=list(MODEL_INFO.keys()),
        format_func=lambda x: f"{MODEL_INFO[x]['icon']}  {x}",
        label_visibility="collapsed",
    )
    info = MODEL_INFO[chosen_model_name]
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                border-radius:8px;padding:0.85rem 1rem;margin-top:0.5rem;font-size:0.78rem;line-height:1.7">
        <div style="color:#93c5fd;font-weight:700;margin-bottom:0.3rem">{chosen_model_name}</div>
        <div>Accuracy · <strong>{info['accuracy']}</strong></div>
        <div>ROC-AUC · <strong>{info['roc_auc']}</strong></div>
        <div style="margin-top:0.5rem;color:#bfdbfe">{info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### A-FGO Feature Set")
    st.markdown("<div style='font-size:0.74rem;color:#93c5fd;margin-bottom:0.5rem'>10 of 30 features selected</div>", unsafe_allow_html=True)
    badges = "".join([f'<span class="feature-badge">#{i+1} {n}</span>' for i,n in enumerate(FEATURE_RANK_ORDER)])
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#93c5fd;line-height:1.7'>Dataset · WBCD (569 samples)<br>Features · 30 → 10<br>Algorithm · Adaptive Fire-Grey Optimizer</div>", unsafe_allow_html=True)

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
    st.error(f"⚠️ Failed to load model files from `{MODEL_DIR}/`.\n\n`{LOAD_ERROR}`")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["🔬  Single Analysis", "📂  Batch CSV Analysis"])

# ══════════════════════════════════════════════
# TAB 1 — SINGLE MANUAL INPUT
# ══════════════════════════════════════════════
with tab_manual:
    st.markdown('<div class="section-label">Cell Morphology Measurements</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem;color:#4b5563;margin-bottom:1.2rem'>Features ordered by A-FGO significance rank — most influential first.</div>", unsafe_allow_html=True)

    input_values = {}
    col1, col2 = st.columns(2)
    for i, fname in enumerate(FEATURE_RANK_ORDER):
        m = FEATURE_META[fname]
        with (col1 if i % 2 == 0 else col2):
            val = st.number_input(
                label=f"#{i+1} · {m['label']}",
                min_value=float(m["min"]), max_value=float(m["max"]),
                value=float(m["default"]), step=float(m["step"]),
                format="%.4f" if m["step"] < 0.01 else "%.2f",
                help=m["help"], key=fname,
            )
            input_values[fname] = val

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    predict_btn = st.button("Run Diagnosis", use_container_width=True, type="primary")

    if predict_btn:
        full_vector = np.zeros((1, total_features))
        for fname, val in input_values.items():
            idx      = selected_names.index(fname)
            full_col = selected_indices[idx]
            full_vector[0, full_col] = val

        full_scaled = scaler.transform(full_vector)
        X_input     = full_scaled[:, selected_indices]

        model_map  = {"Logistic Regression": lr_model, "Random Forest": rf_model, "Decision Tree": dt_model}
        model      = model_map[chosen_model_name]
        prediction = model.predict(X_input)[0]
        probs      = model.predict_proba(X_input)[0]
        prob_b, prob_m = probs[0], probs[1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-malignant">
                <div class="result-title">Tumor Classification Result</div>
                <div class="result-diagnosis">⚠ MALIGNANT</div>
                <div style="color:#b91c1c;font-size:0.88rem;font-weight:500">The morphological profile is consistent with a malignant tumor.</div>
                <div style="margin-top:1rem">
                    <div class="prob-row"><span class="prob-label">Malignant probability</span><span style="color:#dc2626;font-weight:700">{prob_m*100:.1f}%</span></div>
                    <div class="prob-row"><span class="prob-label">Benign probability</span><span style="color:#374151">{prob_b*100:.1f}%</span></div>
                </div>
                <div class="result-note">⚕ For research and educational purposes only. Clinical diagnosis must be confirmed by a qualified medical professional.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-benign">
                <div class="result-title">Tumor Classification Result</div>
                <div class="result-diagnosis">✓ BENIGN</div>
                <div style="color:#15803d;font-size:0.88rem;font-weight:500">The morphological profile is consistent with a benign tumor.</div>
                <div style="margin-top:1rem">
                    <div class="prob-row"><span class="prob-label">Benign probability</span><span style="color:#16a34a;font-weight:700">{prob_b*100:.1f}%</span></div>
                    <div class="prob-row"><span class="prob-label">Malignant probability</span><span style="color:#374151">{prob_m*100:.1f}%</span></div>
                </div>
                <div class="result-note">⚕ For research and educational purposes only. Clinical diagnosis must be confirmed by a qualified medical professional.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-label' style='margin-top:1.8rem'>Input Summary</div>", unsafe_allow_html=True)
        rows_html = ""
        for i, fname in enumerate(FEATURE_RANK_ORDER):
            m   = FEATURE_META[fname]
            val = input_values[fname]
            bg  = "#f8faff" if i % 2 == 0 else "#ffffff"
            rows_html += f"""
            <tr style="background:{bg}">
                <td style="color:#6b7280;padding:0.4rem 0.8rem 0.4rem 0;font-size:0.78rem">#{i+1}</td>
                <td style="color:#1a202c;padding:0.4rem 0.8rem;font-size:0.8rem;font-weight:500">{m['label']}</td>
                <td style="color:#1e4976;font-family:'JetBrains Mono',monospace;font-size:0.79rem;
                           text-align:right;padding:0.4rem 0 0.4rem 0.8rem;font-weight:600">{val:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <div class="card" style="padding:0.5rem 0">
        <table style="width:100%;border-collapse:collapse">
            <thead>
                <tr style="border-bottom:2px solid #dbeafe;background:#f0f7ff">
                    <th style="color:#1e4976;font-size:0.72rem;text-align:left;padding:0.5rem 0.8rem 0.5rem 0">Rank</th>
                    <th style="color:#1e4976;font-size:0.72rem;text-align:left;padding:0.5rem 0.8rem">Feature</th>
                    <th style="color:#1e4976;font-size:0.72rem;text-align:right;padding:0.5rem 0 0.5rem 0.8rem">Value</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — BATCH CSV ANALYSIS
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-label">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem;color:#4b5563;line-height:1.7;margin-bottom:1.2rem">
        Upload a CSV with all 30 WBCD feature columns. The pipeline automatically extracts the
        10 A-FGO features, applies the trained scaler, and classifies every row.
        Columns <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">id</code>,
        <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">diagnosis</code>, and
        <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">Unnamed: 32</code>
        are handled automatically if present.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        help="Must contain the 30 standard WBCD feature columns.",
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.markdown(f"""
        <div class="card" style="padding:0.65rem 1rem;display:flex;align-items:center;gap:0.5rem">
            <span style="font-size:1.1rem">📄</span>
            <span style="color:#1e4976;font-weight:600">{uploaded_file.name}</span>
            <span style="color:#6b7280;font-size:0.8rem">&nbsp;·&nbsp; {len(df_raw):,} rows &nbsp;·&nbsp; {df_raw.shape[1]} columns</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview uploaded data (first 5 rows)", expanded=False):
            st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        run_batch_btn = st.button("Run Batch Analysis", use_container_width=True, type="primary", key="batch_btn")

        if run_batch_btn:
            model_map = {"Logistic Regression": lr_model, "Random Forest": rf_model, "Decision Tree": dt_model}
            model = model_map[chosen_model_name]

            with st.spinner("Running A-FGO pipeline on all rows…"):
                preds, probas, label_col, error = run_batch_prediction(
                    df_raw, model, scaler, selected_indices, total_features
                )

            if error:
                st.error(f"⚠️ {error}")
            else:
                n_total    = len(preds)
                n_malignant = int(np.sum(preds == 1))
                n_benign    = int(np.sum(preds == 0))
                pct_m = n_malignant / n_total * 100
                pct_b = n_benign   / n_total * 100

                # ── Summary card (counts only) ───────────────────
                st.markdown(f"""
                <div class="summary-card">
                    <div class="summary-title">Batch Analysis Summary · {chosen_model_name}</div>
                    <div style="display:flex;gap:2.5rem;flex-wrap:wrap">
                        <div>
                            <div class="stat-label">Total Samples</div>
                            <div class="stat-value" style="color:#1a202c">{n_total:,}</div>
                        </div>
                        <div>
                            <div class="stat-label">Benign</div>
                            <div class="stat-value" style="color:#16a34a">
                                {n_benign:,}<span class="stat-sub"> ({pct_b:.1f}%)</span>
                            </div>
                        </div>
                        <div>
                            <div class="stat-label">Malignant</div>
                            <div class="stat-value" style="color:#dc2626">
                                {n_malignant:,}<span class="stat-sub"> ({pct_m:.1f}%)</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Accuracy block — SEPARATE st.markdown call ───
                # (fixes the bug: accuracy_block was an f-string embedded inside
                #  another f-string, causing Python to re-interpolate its braces
                #  and producing escaped/broken HTML in Streamlit)
                if label_col is not None:
                    gt      = label_col.map({"M": 1, "B": 0}).values if label_col.dtype == object else label_col.values
                    correct = int(np.sum(preds == gt))
                    acc     = correct / n_total * 100
                    st.markdown(f"""
                    <div class="summary-card" style="margin-top:0">
                        <div class="accuracy-label">vs Ground Truth Labels</div>
                        <div class="accuracy-value">
                            {acc:.2f}%
                            <span class="accuracy-sub">&nbsp; accuracy &nbsp;({correct:,} / {n_total:,} correct)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Per-row results table ────────────────────────
                results_df = df_raw.copy()
                if "id" not in results_df.columns:
                    results_df.insert(0, "id", range(1, len(results_df) + 1))
                results_df["A_FGO_Prediction"]  = ["MALIGNANT" if p == 1 else "BENIGN" for p in preds]
                results_df["Prob_Benign (%)"]   = np.round(probas[:, 0] * 100, 2)
                results_df["Prob_Malignant (%)"] = np.round(probas[:, 1] * 100, 2)

                st.markdown('<div class="section-label" style="margin-top:1.5rem">Per-Row Results</div>', unsafe_allow_html=True)
                display_cols = ["id", "A_FGO_Prediction", "Prob_Benign (%)", "Prob_Malignant (%)"]
                if "diagnosis" in results_df.columns:
                    display_cols.insert(2, "diagnosis")
                display_df = results_df[display_cols].copy()

                def color_prediction(val):
                    if val == "MALIGNANT": return "color: #dc2626; font-weight: 700"
                    if val == "BENIGN":    return "color: #16a34a; font-weight: 700"
                    return ""

                styled = display_df.style.applymap(color_prediction, subset=["A_FGO_Prediction"])
                st.dataframe(styled, use_container_width=True, height=380)

                # ── Download ─────────────────────────────────────
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Full Results CSV",
                    data=csv_out,
                    file_name=f"afgo_results_{chosen_model_name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown("""
                <div style="font-size:0.75rem;color:#6b7280;margin-top:0.5rem;text-align:center">
                    Downloaded CSV includes all original columns plus A_FGO_Prediction, Prob_Benign (%), and Prob_Malignant (%).
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2.2rem;margin-bottom:0.7rem">📂</div>
            <div style="color:#374151;font-size:0.88rem;font-weight:600;margin-bottom:0.4rem">Upload a WBCD CSV to begin batch analysis</div>
            <div style="color:#6b7280;font-size:0.8rem;line-height:1.7">
                The file must contain all 30 standard WBCD feature columns.<br>
                The A-FGO pipeline will extract the correct 10 features, apply scaling, and return predictions for every row.<br><br>
                <span style="font-size:0.76rem">Expected columns include:
                <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">radius_mean</code>
                <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">texture_mean</code>
                <code style="color:#1e4976;background:#dbeafe;padding:0.1rem 0.35rem;border-radius:4px">perimeter_mean</code> … and remaining WBCD features.
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
