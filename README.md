# 🧪 MEDISCAN AI

Streamlit app that combines:

1. **OCR-based lab report screening** — upload a PDF/image lab report, extract hemoglobin, glucose,
   cholesterol and TSH, compare against example reference ranges, and get a rule-based screening flag.
2. **Real trained ML risk predictors** — separate manual-entry forms for Diabetes, Anemia, and Thyroid
   condition, backed by models actually trained on matching public datasets (not placeholder models).

> ⚠️ **Medical disclaimer:** Educational prototype only. Not a diagnostic device. Does not replace
> professional medical evaluation.

## Model provenance

The `models/` folder contains real trained models, sourced from
[mrzaeem102-web/Medi-Scan-Ai](https://github.com/mrzaeem102-web/Medi-Scan-Ai):

| Disease  | Model     | Features (in order) | Artifacts |
|----------|-----------|----------------------|-----------|
| Diabetes | XGBoost   | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age | `xgb_diabetes.pkl`, `scaler_diabetes.pkl` |
| Anemia   | LightGBM  | Gender (1=M/0=F), Hemoglobin, MCH, MCHC, MCV | `lgb_anemia.pkl`, `scaler_anemia.pkl` |
| Thyroid  | XGBoost (9-class) | Age, Sex, on-thyroxine, query-on-thyroxine, on-antithyroid-medication, sick, pregnant, thyroid-surgery, I131-treatment, query-hypothyroid, query-hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI | `xgb_thyroid.pkl`, `scaler_thyroid.pkl`, `thyroid_target_encoder.pkl` |

These were verified to load correctly and produce sensible predictions against sample rows drawn from
the corresponding public datasets (`diabetes.csv`, `anemia.csv`, `Thyroid-Dataset.csv`) before being
wired into the app. They were **not** trained on any data uploaded through this app's interface, and
they are not calibrated to any specific individual or clinical population — treat outputs as an
educational reference point only.

The **OCR tab** only extracts 4 general biomarkers (hemoglobin, glucose, cholesterol, TSH) and cannot
supply all the fields the trained models need (e.g. Pregnancies, MCH, or the 21 thyroid clinical
fields). Where OCR does detect a matching value (hemoglobin, glucose, TSH), the **ML Risk Predictors**
tab pre-fills that field — the rest must be entered manually.

## Project Structure

```text
ClarityHealth-Lab-Analyzer/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── xgb_diabetes.pkl
│   ├── scaler_diabetes.pkl
│   ├── lgb_anemia.pkl
│   ├── scaler_anemia.pkl
│   ├── xgb_thyroid.pkl
│   ├── scaler_thyroid.pkl
│   └── thyroid_target_encoder.pkl
└── .gitignore
```

## Installation

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Tesseract OCR

Install the Tesseract OCR engine separately and make sure `tesseract --version` works.
If it is not in PATH on Windows, configure `pytesseract.pytesseract.tesseract_cmd` in `app.py`.

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Reference Ranges

The OCR tab's reference ranges are simplified examples. Real reference intervals vary by laboratory,
units, age, sex, pregnancy status, methodology and clinical context.

## Privacy

Lab reports may contain sensitive information. A production deployment should implement secure
processing, minimal retention, encryption, authentication and applicable privacy/healthcare compliance.

## Future Work

- Better image preprocessing and table OCR
- Extraction of units and report-specific reference ranges
- OCR-to-model auto-fill for the remaining diabetes/anemia/thyroid fields where present in the report
- Model calibration and external validation against a real clinical population
- Database and user authentication
- Clinician review workflow
