# Breast Cancer Survival & Recurrence Risk Prediction

A clinical triage decision-support system that predicts breast cancer survival risk from
diagnosis and treatment features, using a stratified Cox Proportional Hazards survival model.

**Live app:** [https://breast-cancer-survival-app.streamlit.app/](https://breast-cancer-survival-app.streamlit.app/)

**Full technical report:** see `Breast_Cancer_Survival_Report.docx` in this repository.

---

## Problem

Given a patient's staging, tumor characteristics, hormone receptor status, and lymph node
assessment, estimate their 60-month (5-year) survival probability and assign a clinically
interpretable risk band (Low / Medium / High) to support decisions on follow-up intensity.

This is framed as a **time-to-event (survival analysis)** problem rather than binary
classification, so censored patients (those last known alive) contribute information rather
than being discarded.

## Dataset

[SEER Breast Cancer dataset](https://www.kaggle.com/datasets/reihanenamdari/breast-cancer)
(Kaggle, CC0-licensed), n = 4,024 patients.

## Approach

1. **Data cleaning** — fixed a malformed `Grade` category, ordinal-encoded staging variables.
2. **Feature selection** — VIF and correlation analysis showed `Differentiate` was perfectly
   collinear with `Grade`, and `6th Stage` was a derived composite of `T Stage`/`N Stage`; both
   were dropped with quantitative justification.
3. **Modeling** — fit a Cox Proportional Hazards model; detected a proportional-hazards
   violation on Estrogen Receptor status via `proportional_hazard_test`, fixed by stratifying
   the model on that variable.
4. **Comparison model** — tuned a Random Survival Forest across four configurations to address
   overfitting; it never beat the stratified Cox model's held-out concordance.
5. **Model selection** — stratified Cox PH selected on test-set concordance (0.718), not
   training performance.
6. **Risk banding** — converted the model's 60-month survival probability into Low/Medium/High
   risk bands, validated against actual outcomes (7x difference in death rate between bands).
7. **Deployment** — packaged as an interactive Streamlit app for clinical use.

Full reasoning, debugging log (including a real data-alignment bug and how it was caught), and
all evaluation visuals are in the technical report and the notebook.

## Repository Contents

| File | Description |
|---|---|
| `breast_cancer_survival_analysis.ipynb` | Full development notebook — data cleaning, VIF analysis, Cox/RSF modeling, debugging, evaluation |
| `Breast_Cancer_Survival_Report.docx` | Consolidated technical report |
| `app.py` | Streamlit application |
| `cox_model.pkl` | Trained, stratified Cox PH model |
| `scored_patients.csv` | Full dataset scored with predicted risk bands |
| `requirements.txt` | Python dependencies for deployment |

## Running Locally

```bash
git clone https://github.com/Aybeeofili/Breast-Cancer-Survival-App.git
cd Breast-Cancer-Survival-App
pip install -r requirements.txt
streamlit run app.py
```

## Model Performance

| Model | Test Concordance Index |
|---|---|
| Cox PH (stratified) | **0.718** |
| Random Survival Forest (tuned) | 0.710 |

## Limitations

- Risk band thresholds (90% / 70% survival probability) were chosen by inspecting the
  combined train+test distribution, not independently cross-validated.
- This is a decision-support tool trained on a single public dataset; it is not validated for
  clinical use and should not be used to make real treatment decisions.

## References

See the References section of the technical report for full citations (Cox, 1972; Ishwaran et
al., 2008; Davidson-Pilon, 2019; Pölsterl, 2020).
