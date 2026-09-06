import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

try:
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ClarityHealth • AI Lab Report Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    color: #0e1d26 !important;
}

.stApp {
    background-color: #f5faff;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e0f0fc !important;
}

.header-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #005087 0%, #1769aa 100%);
    padding: 24px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0, 80, 135, 0.15);
}

.header-title {
    font-family: 'Manrope', sans-serif;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    color: #ffffff !important;
}

.header-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #d4e6ff;
    margin-top: 4px;
}

.logo-badge {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 6px 14px;
    border-radius: 9999px;
    font-family: 'Manrope', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.03em;
}

.medical-disclaimer-box {
    background-color: #e9f5ff;
    border: 1px solid #b8daff;
    border-radius: 12px;
    padding: 14px 18px;
    color: #00497c;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 24px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.clarity-card {
    background-color: #ffffff;
    border: 1px solid #e0f0fc;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(23, 33, 43, 0.04);
    margin-bottom: 20px;
}

.clarity-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.clarity-title {
    font-family: 'Manrope', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #0e1d26;
}

.stButton > button {
    background-color: #1769aa !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #005087 !important;
    box-shadow: 0 4px 12px rgba(23, 105, 170, 0.25) !important;
}

.stDownloadButton > button {
    background-color: #1769aa !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-feature-settings: "tnum" 1 !important;
    font-weight: 700 !important;
    color: #0e1d26 !important;
}

/* ---- Force light, legible widgets regardless of system/browser theme ---- */
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #f5faff !important;
}

[data-testid="stSidebar"] * {
    color: #0e1d26 !important;
}

/* Selectbox / dropdown widgets */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #c1c7d2 !important;
    color: #0e1d26 !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] div {
    color: #0e1d26 !important;
}
div[role="listbox"] {
    background-color: #ffffff !important;
}
li[role="option"] {
    background-color: #ffffff !important;
    color: #0e1d26 !important;
}
li[role="option"]:hover {
    background-color: #e9f5ff !important;
}

/* Number / text inputs */
[data-testid="stNumberInput"] input, input[type="text"], input[type="number"] {
    background-color: #ffffff !important;
    color: #0e1d26 !important;
    border: 1px solid #c1c7d2 !important;
}

/* Checkboxes label text */
[data-testid="stCheckbox"] label p {
    color: #0e1d26 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #52616b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1769aa !important;
}

/* File uploader: clean single label, no overlapping ghost text */
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 1.5px dashed #9dcaff !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #414750 !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: #1769aa !important;
}
[data-testid="stBaseButton-secondary"] {
    background-color: #1769aa !important;
    color: #ffffff !important;
    border: none !important;
    white-space: nowrap !important;
    overflow: visible !important;
    min-width: fit-content !important;
    padding: 8px 18px !important;
}
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] span,
[data-testid="stBaseButton-secondary"] div {
    color: #ffffff !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* Captions / help text / expander text visibility */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #52616b !important;
}
[data-testid="stExpander"] * {
    color: #0e1d26 !important;
}

/* Plain markdown/paragraph text on the light background */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    color: #17212b !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="header-banner">
    <div>
        <h1 class="header-title">🧪 ClarityHealth Lab Analyzer</h1>
        <div class="header-subtitle">Clinical Precision & AI-Powered Lab Report Interpretation</div>
    </div>
    <div class="logo-badge">ClarityHealth AI v2.5</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="medical-disclaimer-box">
    <div style="font-size: 20px; line-height: 1;">🛡️</div>
    <div>
        <strong>INFORMATIONAL SCREENING & CLINICAL ADVISORY:</strong> This application performs automated Optical
        Character Recognition (OCR) and machine learning risk estimation for informational purposes only. It is
        <strong>NOT a medical diagnosis</strong> and should never replace consultation with a qualified healthcare
        professional. ML predictions come from models trained on public reference datasets (Pima Diabetes, an
        anemia CBC dataset, and a thyroid clinical dataset) — they are not calibrated to any individual patient
        population and should be treated as an educational reference point, not a clinical result.
    </div>
</div>
""", unsafe_allow_html=True)

REFERENCE_RANGES = {
    "hemoglobin": {"unit": "g/dL", "male": (13.5, 17.5), "female": (12.0, 15.5)},
    "glucose": {"unit": "mg/dL", "range": (70, 99)},
    "cholesterol": {"unit": "mg/dL", "range": (0, 199)},
    "tsh": {"unit": "mIU/L", "range": (0.4, 4.0)}
}

# ============================================================
# MODEL LOADING
# Real models trained on matching public datasets:
#   Diabetes -> XGBoost, 8 features (Pima Indians Diabetes dataset)
#   Anemia   -> LightGBM, 5 features (CBC anemia dataset)
#   Thyroid  -> XGBoost, 21 features (UCI-style thyroid dataset) + LabelEncoder
# Source: github.com/mrzaeem102-web/Medi-Scan-Ai
# ============================================================
MODEL_DIR = "models"
MODEL_FILES = {
    "model_diabetes": os.path.join(MODEL_DIR, "xgb_diabetes.pkl"),
    "scaler_diabetes": os.path.join(MODEL_DIR, "scaler_diabetes.pkl"),
    "model_anemia": os.path.join(MODEL_DIR, "lgb_anemia.pkl"),
    "scaler_anemia": os.path.join(MODEL_DIR, "scaler_anemia.pkl"),
    "model_thyroid": os.path.join(MODEL_DIR, "xgb_thyroid.pkl"),
    "scaler_thyroid": os.path.join(MODEL_DIR, "scaler_thyroid.pkl"),
    "thyroid_target_encoder": os.path.join(MODEL_DIR, "thyroid_target_encoder.pkl"),
}


@st.cache_resource
def load_models():
    loaded, errors = {}, {}
    for key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            errors[key] = f"File not found: {path}"
            continue
        try:
            loaded[key] = joblib.load(path)
        except Exception as exc:
            errors[key] = f"{path}: {exc}"
    return loaded, errors


models, model_errors = load_models()

DIABETES_READY = "model_diabetes" in models and "scaler_diabetes" in models
ANEMIA_READY = "model_anemia" in models and "scaler_anemia" in models
THYROID_READY = "model_thyroid" in models and "scaler_thyroid" in models


def get_binary_prediction(model, scaled_data):
    prediction = model.predict(scaled_data)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(np.max(model.predict_proba(scaled_data)[0]))
        except Exception:
            probability = None
    return prediction, probability


def decode_prediction(prediction, encoder=None):
    if encoder is None:
        return prediction
    try:
        return encoder.inverse_transform(np.asarray([prediction]).astype(int))[0]
    except Exception:
        return prediction


# ============================================================
# OCR EXTRACTION (unchanged pipeline)
# ============================================================
def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"OCR Error: {e}"


def extract_text_from_pdf(uploaded_file):
    if not PDF_AVAILABLE:
        return "PyMuPDF is not installed."
    try:
        pdf_bytes = uploaded_file.read()
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""
        for page in document:
            page_text = page.get_text()
            if page_text.strip():
                all_text += page_text + "\n"
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                all_text += extract_text_from_image(image) + "\n"
        return all_text
    except Exception as e:
        return f"PDF extraction error: {e}"


def clean_number(value):
    try:
        value = value.replace(",", "").replace("O", "0").replace("o", "0")
        return float(value)
    except (ValueError, AttributeError):
        return None


def extract_lab_values(text):
    text_lower = text.lower()
    values = {"hemoglobin": None, "glucose": None, "cholesterol": None, "tsh": None}
    patterns = {
        "hemoglobin": [r"(?:hemoglobin|haemoglobin|hgb|hb)\s*[:\-]?\s*(\d+(?:\.\d+)?)"],
        "glucose": [r"(?:blood\s+glucose|glucose|blood\s+sugar|sugar)\s*[:\-]?\s*(\d+(?:\.\d+)?)"],
        "cholesterol": [r"(?:total\s+cholesterol|cholesterol)\s*[:\-]?\s*(\d+(?:\.\d+)?)"],
        "tsh": [r"(?:tsh|thyroid\s+stimulating\s+hormone)\s*[:\-]?\s*(\d+(?:\.\d+)?)"]
    }
    for name, pats in patterns.items():
        for pattern in pats:
            match = re.search(pattern, text_lower, flags=re.IGNORECASE)
            if match:
                number = clean_number(match.group(1))
                if number is not None:
                    values[name] = number
                    break
    return values


def analyze_value(test_name, value, sex="male"):
    if value is None:
        return {"status": "Not detected", "message": "Value could not be extracted from document."}
    ref = REFERENCE_RANGES[test_name]
    low, high = ref.get(sex, ref.get("range", ref["male"]))
    if value < low:
        status = "Low"
    elif value > high:
        status = "High"
    else:
        status = "Normal"
    return {"status": status, "message": f"{value} {ref['unit']} vs reference range ({low} - {high} {ref['unit']})"}


def statistical_analysis(values):
    numeric = [v for v in values.values() if v is not None]
    if not numeric:
        return None
    array = np.array(numeric, dtype=float)
    return {"mean": np.mean(array), "median": np.median(array), "std": np.std(array),
             "minimum": np.min(array), "maximum": np.max(array)}


def calculate_rule_based_risk(values):
    points, reasons = 0, []
    glucose, hemoglobin = values.get("glucose"), values.get("hemoglobin")
    tsh, cholesterol = values.get("tsh"), values.get("cholesterol")
    if glucose is not None:
        if glucose >= 126:
            points += 2; reasons.append("Fasting glucose is elevated (≥126 mg/dL) warranting clinical evaluation.")
        elif glucose >= 100:
            points += 1; reasons.append("Glucose is above standard fasting reference baseline.")
    if hemoglobin is not None:
        if hemoglobin < 10:
            points += 2; reasons.append("Hemoglobin is markedly below reference baseline.")
        elif hemoglobin < 12:
            points += 1; reasons.append("Hemoglobin is below standard reference range.")
    if tsh is not None and (tsh < 0.4 or tsh > 4.0):
        points += 1; reasons.append("TSH biomarker is outside expected clinical limits.")
    if cholesterol is not None:
        if cholesterol >= 240:
            points += 2; reasons.append("Total cholesterol is elevated (≥240 mg/dL).")
        elif cholesterol >= 200:
            points += 1; reasons.append("Total cholesterol exceeds standard desirable limits.")
    level = "Higher concern" if points >= 4 else "Moderate concern" if points >= 2 else "Lower concern"
    return level, reasons


def generate_report(values, analyses):
    lines = ["### 📝 ClarityHealth Diagnostic Interpretation Report", ""]
    names = {"hemoglobin": "Hemoglobin", "glucose": "Glucose", "cholesterol": "Cholesterol", "tsh": "TSH"}
    for name, value in values.items():
        if value is not None:
            lines.append(f"- **{names[name]}:** {value} {REFERENCE_RANGES[name]['unit']} → **{analyses[name]['status']}**")
    level, reasons = calculate_rule_based_risk(values)
    lines += ["", f"### Overall Triage Assessment: **{level}**", ""]
    if reasons:
        lines.append("### Key Analytical Observations")
        lines.extend([f"- {r}" for r in reasons])
    else:
        lines.append("All extracted biomarkers align with standard reference baseline intervals.")
    lines += ["", "---", "⚠️ *Reference intervals may vary across laboratory methodologies. Always review original laboratory reports with a licensed healthcare clinician.*"]
    return "\n".join(lines)


def generate_report_html(values, analyses):
    """HTML version of the report, safe to embed inside a raw <div> card."""
    names = {"hemoglobin": "Hemoglobin", "glucose": "Glucose", "cholesterol": "Cholesterol", "tsh": "TSH"}
    rows_html = ""
    for name, value in values.items():
        if value is not None:
            rows_html += (
                f'<div style="margin-bottom:4px;">'
                f'<b>{names[name]}:</b> {value} {REFERENCE_RANGES[name]["unit"]} &rarr; <b>{analyses[name]["status"]}</b>'
                f'</div>'
            )
    level, reasons = calculate_rule_based_risk(values)
    reasons_html = ""
    if reasons:
        reasons_html += '<div style="font-weight:700;margin-top:14px;margin-bottom:6px;">Key Analytical Observations</div>'
        for r in reasons:
            reasons_html += f'<div style="margin-bottom:4px;">&bull; {r}</div>'
    else:
        reasons_html = '<div style="margin-top:10px;">All extracted biomarkers align with standard reference baseline intervals.</div>'

    return (
        '<div style="font-family:\'Manrope\',sans-serif;font-size:16px;font-weight:700;color:#0e1d26;margin-bottom:12px;">'
        '📝 ClarityHealth Diagnostic Interpretation Report</div>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:14px;color:#17212b;line-height:1.6;">{rows_html}</div>'
        f'<div style="font-family:\'Manrope\',sans-serif;font-size:15px;font-weight:700;color:#0e1d26;margin-top:16px;">Overall Triage Assessment: {level}</div>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:14px;color:#17212b;line-height:1.6;">{reasons_html}</div>'
        '<hr style="margin:16px 0;border:none;border-top:1px solid #e0f0fc;">'
        '<div style="font-size:13px;color:#52616b;font-style:italic;">⚠️ Reference intervals may vary across laboratory methodologies. '
        'Always review original laboratory reports with a licensed healthcare clinician.</div>'
    )


def render_status_chip(status):
    if status == "Normal":
        color, bg, border, icon = "#005a3c", "#a6f5cb", "#88d6af", "✓"
    elif status == "High":
        color, bg, border, icon = "#ba1a1a", "#ffdad6", "#ffb4ab", "▲"
    elif status == "Low":
        color, bg, border, icon = "#00497c", "#d1e4ff", "#9dcaff", "▼"
    else:
        color, bg, border, icon = "#414750", "#e0f0fc", "#c1c7d2", "•"
    return f"""<span style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; background-color: {bg}; color: {color}; border: 1px solid {border}; border-radius: 9999px; font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600;">{icon} {status}</span>"""


def render_range_meter(test_name, value, sex="male"):
    if value is None:
        return """<div style="font-size: 12px; color: #717781; font-style: italic; margin-top: 8px;">No numerical value detected for range visualizer.</div>"""
    ref = REFERENCE_RANGES[test_name]
    low, high = ref.get(sex, ref.get("range", ref["male"]))
    span = high - low
    scale_min = max(0, low - span * 0.8)
    scale_max = high + span * 0.8
    if scale_max == scale_min:
        pct = 50
    else:
        pct = ((value - scale_min) / (scale_max - scale_min)) * 100
    pct = max(4, min(96, pct))
    norm_start = ((low - scale_min) / (scale_max - scale_min)) * 100
    norm_end = ((high - scale_min) / (scale_max - scale_min)) * 100
    return (
        '<div style="margin-top:12px;margin-bottom:4px;">'
        '<div style="display:flex;justify-content:space-between;font-size:11px;color:#52616b;font-family:\'Inter\',sans-serif;margin-bottom:6px;">'
        f'<span>Ref: {low} - {high} {ref["unit"]}</span>'
        f'<span>Extracted: <b>{value}</b></span>'
        '</div>'
        '<div style="position:relative;width:100%;height:8px;background:#e0f0fc;border-radius:9999px;overflow:hidden;">'
        f'<div style="position:absolute;left:{norm_start}%;width:{norm_end - norm_start}%;height:100%;background:#2e7d5b;opacity:0.85;"></div>'
        f'<div style="position:absolute;left:0%;width:{norm_start}%;height:100%;background:#c1c7d2;opacity:0.5;"></div>'
        f'<div style="position:absolute;left:{norm_end}%;width:{100 - norm_end}%;height:100%;background:#ffdad6;opacity:0.7;"></div>'
        '</div>'
        '<div style="position:relative;width:100%;height:14px;margin-top:-11px;">'
        f'<div style="position:absolute;left:calc({pct}% - 7px);top:0px;width:14px;height:14px;background:#005087;border:2px solid #ffffff;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.3);" title="{value} {ref["unit"]}"></div>'
        '</div>'
        '</div>'
    )


def render_risk_card(risk_level, reasons):
    if risk_level == "Higher concern":
        rail, bg, title_color, badge = "#ba1a1a", "#fff5f5", "#ba1a1a", "HIGH ATTENTION"
    elif risk_level == "Moderate concern":
        rail, bg, title_color, badge = "#c98500", "#fffbf0", "#92400e", "ADVISORY WATCHLIST"
    else:
        rail, bg, title_color, badge = "#2e7d5b", "#f4fdf8", "#005a3c", "OPTIMAL BASELINE"

    reasons_html = "".join([f"<li style='margin-bottom: 6px;'>{r}</li>" for r in reasons]) if reasons else "<li>No obvious analytical flags detected across extracted values.</li>"

    return f"""
    <div style="border-left: 6px solid {rail}; background-color: {bg}; padding: 20px 24px; border-radius: 12px; border: 1px solid #dce3e8; border-left-color: {rail}; margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h4 style="margin: 0; color: {title_color}; font-family: 'Manrope', sans-serif; font-size: 18px; font-weight: 700;">Screening Flag: {risk_level}</h4>
            <span style="font-size: 11px; font-weight: 700; color: {rail}; letter-spacing: 0.05em; background: rgba(255,255,255,0.8); padding: 4px 10px; border-radius: 9999px; border: 1px solid {rail};">{badge}</span>
        </div>
        <ul style="margin: 8px 0 0 18px; padding: 0; color: #17212b; font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6;">
            {reasons_html}
        </ul>
    </div>
    """


def render_ml_result(label, prediction, probability, positive_is_risk=True):
    if isinstance(prediction, (int, np.integer)) and positive_is_risk:
        is_risk = int(prediction) == 1
        flag_text = "Elevated Risk Flag" if is_risk else "Standard Risk Profile"
        flag_color = "#ba1a1a" if is_risk else "#005a3c"
    else:
        flag_text = f"Predicted class: {prediction}"
        flag_color = "#ba1a1a" if str(prediction).lower() != "negative" else "#005a3c"
    st.markdown(f"<div style='color: {flag_color}; font-weight: 700; font-size: 16px; margin: 8px 0;'>{flag_text}</div>", unsafe_allow_html=True)
    if probability is not None:
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Model confidence: {probability * 100:.1f}%")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### 👤 Patient Profile Settings")
sex = st.sidebar.selectbox("Biological sex for reference intervals", ["male", "female"], index=0)
st.sidebar.caption("Reference ranges for Hemoglobin adapt based on biological sex selection.")

st.sidebar.divider()
st.sidebar.markdown("### 🤖 Clinical ML Models")
st.sidebar.markdown(f"- **Diabetes Model (XGBoost):** {'🟢 Loaded' if DIABETES_READY else '⚪ Unavailable'}")
st.sidebar.markdown(f"- **Anemia Model (LightGBM):** {'🟢 Loaded' if ANEMIA_READY else '⚪ Unavailable'}")
st.sidebar.markdown(f"- **Thyroid Model (XGBoost):** {'🟢 Loaded' if THYROID_READY else '⚪ Unavailable'}")
if model_errors:
    with st.sidebar.expander("Model load details"):
        for k, v in model_errors.items():
            st.caption(f"{k}: {v}")

st.sidebar.divider()
st.sidebar.caption("ClarityHealth AI • Educational Prototype Only")

# ============================================================
# TABS: OCR Report Analysis  |  ML Risk Predictors
# ============================================================
tab_ocr, tab_ml = st.tabs(["📄 Lab Report Analysis (OCR)", "🤖 ML Risk Predictors"])

# ------------------------------------------------------------
# TAB 1 — OCR-driven extraction + reference-range screening
# ------------------------------------------------------------
with tab_ocr:
    uploaded_file = st.file_uploader("📥 Upload Laboratory Source Document (PDF or Image)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        with st.spinner("Processing document & running Optical Character Recognition..."):
            if "pdf" in uploaded_file.type:
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_image(Image.open(uploaded_file))

        st.success("✅ Laboratory report processed successfully.")

        with st.expander("🔍 Inspect Extracted OCR Raw Text"):
            st.text_area("OCR Raw Content", text, height=220)

        values = extract_lab_values(text)
        st.session_state["ocr_values"] = values
        analyses = {name: analyze_value(name, value, sex) for name, value in values.items()}

        st.markdown("<h3 style='margin-top: 25px;'>🧪 Extracted Biomarker Cards</h3>", unsafe_allow_html=True)

        cols = st.columns(4)
        names = {"hemoglobin": "Hemoglobin", "glucose": "Blood Glucose", "cholesterol": "Total Cholesterol", "tsh": "TSH"}

        for i, (name, value) in enumerate(values.items()):
            with cols[i]:
                unit = REFERENCE_RANGES[name]["unit"]
                val_str = f"{value} <span style='font-size: 13px; font-weight: 500; color: #52616b;'>{unit}</span>" if value is not None else "Not detected"
                status = analyses[name]["status"]
                chip_html = render_status_chip(status)
                range_meter_html = render_range_meter(name, value, sex)

                card_html = (
                    '<div class="clarity-card">'
                    '<div class="clarity-card-header">'
                    f'<span class="clarity-title">{names[name]}</span>'
                    f'{chip_html}'
                    '</div>'
                    '<div style="font-family:\'Inter\',sans-serif;font-size:24px;font-weight:700;color:#0e1d26;margin-bottom:4px;">'
                    f'{val_str}'
                    '</div>'
                    f'{range_meter_html}'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top: 20px;'>📊 Diagnostic Reference Interval Table</h3>", unsafe_allow_html=True)
        rows = []
        for name, value in values.items():
            result = analyses[name]
            rows.append({
                "Biomarker Test": names[name],
                "Extracted Value": value if value is not None else "Not detected",
                "Unit": REFERENCE_RANGES[name]["unit"],
                "Clinical Status": result["status"],
                "Reference Evaluation": result["message"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<h3 style='margin-top: 25px;'>📈 Statistical Baseline Summary</h3>", unsafe_allow_html=True)
        stats = statistical_analysis(values)
        if stats:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mean Value", round(stats["mean"], 2))
            c2.metric("Median Value", round(stats["median"], 2))
            c3.metric("Std Deviation", round(stats["std"], 2))
            c4.metric("Minimum", round(stats["minimum"], 2))
            c5.metric("Maximum", round(stats["maximum"], 2))
        else:
            st.info("Insufficient numeric lab values detected for statistical calculation.")

        st.markdown("<h3 style='margin-top: 25px;'>⚠️ Algorithmic Screening Triage</h3>", unsafe_allow_html=True)
        risk_level, risk_reasons = calculate_rule_based_risk(values)
        st.markdown(render_risk_card(risk_level, risk_reasons), unsafe_allow_html=True)

        st.markdown("<h3 style='margin-top: 25px;'>📄 Personalized Summary Report</h3>", unsafe_allow_html=True)
        report_text = generate_report(values, analyses)
        report_html = generate_report_html(values, analyses)
        st.markdown(f'<div class="clarity-card">{report_html}</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download ClarityHealth Summary Report",
            data=report_text,
            file_name="ClarityHealth_Lab_Report.txt",
            mime="text/plain"
        )

        st.info("💡 Detected values (hemoglobin, glucose, TSH) are pre-filled in the **ML Risk Predictors** tab where applicable — head there to run a full model prediction with the remaining clinical inputs.")
    else:
        st.markdown("""
        <div class="clarity-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 40px; margin-bottom: 10px;">📋</div>
            <h3 style="margin-bottom: 8px; color: #005087;">Ready for Lab Report Analysis</h3>
            <p style="color: #52616b; max-width: 500px; margin: 0 auto 16px auto;">Upload a PDF or image of your laboratory findings above to extract biomarker values, generate statistical baselines, and review a rule-based screening flag.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2 — Real trained ML models (XGBoost / LightGBM)
# ------------------------------------------------------------
with tab_ml:
    st.caption(
        "These models were trained on public reference datasets (Pima Diabetes, an anemia CBC dataset, "
        "and a thyroid clinical dataset) and require the exact clinical inputs shown below — the same "
        "features used during training. Predictions are a statistical estimate from those datasets, not a "
        "diagnosis for any specific individual."
    )

    ocr_values = st.session_state.get("ocr_values", {})

    m_tab1, m_tab2, m_tab3 = st.tabs(["🩺 Diabetes", "🩸 Anemia", "🦋 Thyroid"])

    # ---------------- DIABETES ----------------
    with m_tab1:
        st.markdown('<div class="clarity-title">Diabetes Risk — XGBoost Classifier</div>', unsafe_allow_html=True)
        if not DIABETES_READY:
            st.error("Diabetes model unavailable. Ensure xgb_diabetes.pkl and scaler_diabetes.pkl are present in models/.")
        else:
            with st.form("diabetes_form"):
                col1, col2 = st.columns(2)
                with col1:
                    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=30, value=1, step=1)
                    glucose = st.number_input("Glucose", min_value=0.0, max_value=400.0,
                                               value=float(ocr_values.get("glucose") or 120.0), step=1.0)
                    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=250.0, value=70.0, step=1.0)
                    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
                with col2:
                    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
                    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
                    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
                    age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)
                submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

            if submitted:
                try:
                    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                       insulin, bmi, pedigree, age]], dtype=float)
                    scaled = models["scaler_diabetes"].transform(data)
                    prediction, probability = get_binary_prediction(models["model_diabetes"], scaled)
                    render_ml_result("Diabetes", int(prediction), probability)
                except Exception as exc:
                    st.error(f"Diabetes prediction failed: {exc}")

    # ---------------- ANEMIA ----------------
    with m_tab2:
        st.markdown('<div class="clarity-title">Anemia Risk — LightGBM Classifier</div>', unsafe_allow_html=True)
        if not ANEMIA_READY:
            st.error("Anemia model unavailable. Ensure lgb_anemia.pkl and scaler_anemia.pkl are present in models/.")
        else:
            with st.form("anemia_form"):
                col1, col2 = st.columns(2)
                with col1:
                    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if sex == "female" else 1)
                    hemoglobin = st.number_input("Hemoglobin", min_value=0.0, max_value=30.0,
                                                  value=float(ocr_values.get("hemoglobin") or 12.0), step=0.1)
                    mch = st.number_input("MCH", min_value=0.0, max_value=60.0, value=27.0, step=0.1)
                with col2:
                    mchc = st.number_input("MCHC", min_value=0.0, max_value=60.0, value=32.0, step=0.1)
                    mcv = st.number_input("MCV", min_value=0.0, max_value=150.0, value=85.0, step=0.1)
                submitted = st.form_submit_button("🔍 Predict Anemia Risk", use_container_width=True)

            if submitted:
                try:
                    gender_value = 1 if gender == "Male" else 0
                    data = np.array([[gender_value, hemoglobin, mch, mchc, mcv]], dtype=float)
                    scaled = models["scaler_anemia"].transform(data)
                    prediction, probability = get_binary_prediction(models["model_anemia"], scaled)
                    render_ml_result("Anemia", int(prediction), probability)
                except Exception as exc:
                    st.error(f"Anemia prediction failed: {exc}")

    # ---------------- THYROID ----------------
    with m_tab3:
        st.markdown('<div class="clarity-title">Thyroid Condition — XGBoost Classifier (9-class)</div>', unsafe_allow_html=True)
        if not THYROID_READY:
            st.error("Thyroid model unavailable. Ensure xgb_thyroid.pkl, scaler_thyroid.pkl and thyroid_target_encoder.pkl are present in models/.")
        else:
            with st.form("thyroid_form"):
                st.markdown("##### Patient information")
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
                with c2:
                    t_sex = st.selectbox("Sex", ["Female", "Male"], index=0 if sex == "female" else 1, key="thy_sex")
                with c3:
                    sick = st.checkbox("Sick")

                st.markdown("##### Medication and history")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    on_thyroxine = st.checkbox("On Thyroxine")
                with c2:
                    query_on_thyroxine = st.checkbox("Query on Thyroxine")
                with c3:
                    on_antithyroid_medication = st.checkbox("On Antithyroid Medication")
                with c4:
                    pregnant = st.checkbox("Pregnant")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    thyroid_surgery = st.checkbox("Thyroid Surgery")
                with c2:
                    i131_treatment = st.checkbox("I-131 Treatment")
                with c3:
                    query_hypothyroid = st.checkbox("Query Hypothyroid")
                with c4:
                    query_hyperthyroid = st.checkbox("Query Hyperthyroid")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    lithium = st.checkbox("Lithium")
                with c2:
                    goitre = st.checkbox("Goitre")
                with c3:
                    tumor = st.checkbox("Tumor")
                with c4:
                    hypopituitary = st.checkbox("Hypopituitary")

                psych = st.checkbox("Psychological condition")

                st.markdown("##### Thyroid laboratory values")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    tsh = st.number_input("TSH", min_value=0.0,
                                           value=float(ocr_values.get("tsh") or 2.0), step=0.01)
                with c2:
                    t3 = st.number_input("T3", min_value=0.0, value=1.8, step=0.01)
                with c3:
                    tt4 = st.number_input("TT4", min_value=0.0, value=100.0, step=0.1)
                with c4:
                    t4u = st.number_input("T4U", min_value=0.0, value=1.0, step=0.01)
                with c5:
                    fti = st.number_input("FTI", min_value=0.0, value=100.0, step=0.1)

                submitted = st.form_submit_button("🔍 Predict Thyroid Class", use_container_width=True)

            if submitted:
                try:
                    sex_value = 1 if t_sex == "Male" else 0
                    # Feature order matches the 21-feature thyroid training pipeline.
                    data = np.array([[
                        t_age, sex_value,
                        int(on_thyroxine), int(query_on_thyroxine), int(on_antithyroid_medication),
                        int(sick), int(pregnant), int(thyroid_surgery), int(i131_treatment),
                        int(query_hypothyroid), int(query_hyperthyroid), int(lithium),
                        int(goitre), int(tumor), int(hypopituitary), int(psych),
                        tsh, t3, tt4, t4u, fti
                    ]], dtype=float)
                    scaled = models["scaler_thyroid"].transform(data)
                    raw_prediction, probability = get_binary_prediction(models["model_thyroid"], scaled)
                    decoded = decode_prediction(raw_prediction, models.get("thyroid_target_encoder"))
                    render_ml_result("Thyroid", decoded, probability, positive_is_risk=False)
                except Exception as exc:
                    st.error(f"Thyroid prediction failed: {exc}")

st.divider()
st.caption("ClarityHealth AI Lab Analyzer • OCR + Rule-Based Screening + Real Trained ML Models • Educational Prototype Only")
