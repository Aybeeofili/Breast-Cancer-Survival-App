import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Oncology Survival Analytics Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens (from brief)
# ----------------------------------------------------------------------------
PRIMARY = "#1E3A5F"
SECONDARY = "#3F5E7A"
ACCENT = "#5E548E"
SUCCESS = "#4F7A65"
WARNING = "#B08D57"
DANGER = "#9E4B4B"
BACKGROUND = "#F8FAFC"
BORDER = "#D9E2EC"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BACKGROUND};
    }}
    html, body, [class*="css"] {{
        font-family: 'Source Sans Pro', 'Segoe UI', Arial, sans-serif;
    }}
    .main-header {{
        background-color: {PRIMARY};
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
        box-shadow: 0 1px 3px rgba(30,58,95,0.15);
    }}
    .main-header h1 {{
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }}
    .main-header p {{
        color: #C9D6E3;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }}
    .metric-card {{
        background-color: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(30,58,95,0.06);
        text-align: left;
    }}
    .metric-label {{
        color: {SECONDARY};
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        color: {PRIMARY};
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .section-card {{
        background-color: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 2px rgba(30,58,95,0.06);
        margin-bottom: 1rem;
    }}
    .section-title {{
        color: {PRIMARY};
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 0.5rem;
    }}
    .badge-low {{
        background-color: {SUCCESS}20;
        color: {SUCCESS};
        border: 1px solid {SUCCESS};
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-medium {{
        background-color: {WARNING}20;
        color: {WARNING};
        border: 1px solid {WARNING};
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-high {{
        background-color: {DANGER}20;
        color: {DANGER};
        border: 1px solid {DANGER};
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .unused-field-note {{
        color: {SECONDARY};
        font-size: 0.7rem;
        font-style: italic;
        margin-top: -0.5rem;
        margin-bottom: 0.4rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: {PRIMARY};
    }}
    .stButton > button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY};
        color: white;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid {BORDER};
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model + data
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("cox_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("scored_patients.csv")

cph = load_model()
data = load_data()

COVARIATE_COLS = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive',
                   'Grade', 'T Stage', 'N Stage', 'Race_Other', 'Race_White',
                   'Marital Status_Married', 'Marital Status_Separated', 'Marital Status_Single ',
                   'Marital Status_Widowed', 'A Stage_Regional', 'Progesterone Status_Positive']
STRATA_COL = 'Estrogen Status_Positive'
NUMERIC_COLS = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive']

TRAIN_MEANS = data[NUMERIC_COLS].mean()
TRAIN_STDS = data[NUMERIC_COLS].std()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Oncology Survival Analytics Platform</h1>
    <p>Breast cancer recurrence &amp; survival risk assessment — clinical decision support</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Top-level dashboard metrics
# ----------------------------------------------------------------------------
total_patients = len(data)
survival_rate = (1 - data['event'].mean()) * 100
mortality_rate = data['event'].mean() * 100
avg_survival = data['duration'].mean()

m1, m2, m3, m4 = st.columns(4)
for col, label, value, color in [
    (m1, "Total Patients", f"{total_patients:,}", PRIMARY),
    (m2, "Survival Rate", f"{survival_rate:.1f}%", SUCCESS),
    (m3, "Mortality Rate", f"{mortality_rate:.1f}%", DANGER),
    (m4, "Avg. Survival (months)", f"{avg_survival:.1f}", SECONDARY),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

tab_predict, tab_analytics = st.tabs(["🩺 Patient Risk Assessment", "📊 Cohort Analytics"])

# ============================================================================
# TAB 1 — PREDICTION
# ============================================================================
with tab_predict:
    left, right = st.columns([1, 1.1])

    with left:
        st.markdown('<div class="section-card"><div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        age = st.slider("Age", 20, 100, 55)
        race = st.selectbox("Race", ["White", "Black", "Other"])
        marital = st.selectbox("Marital Status", ["Married", "Divorced", "Single", "Widowed", "Separated"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Cancer Staging</div>', unsafe_allow_html=True)
        t_stage = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
        n_stage = st.selectbox("N Stage", ["N1", "N2", "N3"])
        stage_6th = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
        st.markdown('<div class="unused-field-note">Recorded for clinical context — not used in prediction (redundant with T/N Stage; see feature-selection analysis)</div>', unsafe_allow_html=True)
        a_stage = st.selectbox("A Stage", ["Regional", "Distant"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Tumor Characteristics</div>', unsafe_allow_html=True)
        tumor_size = st.slider("Tumor Size (mm)", 1, 140, 25)
        grade = st.selectbox("Grade", ["1", "2", "3", "anaplastic; Grade IV"])
        differentiation = st.selectbox("Differentiation", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"])
        st.markdown('<div class="unused-field-note">Recorded for clinical context — not used in prediction (perfectly collinear with Grade)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Hormonal Status</div>', unsafe_allow_html=True)
        estrogen = st.selectbox("Estrogen Status", ["Positive", "Negative"])
        progesterone = st.selectbox("Progesterone Status", ["Positive", "Negative"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="section-title">Lymph Node Assessment</div>', unsafe_allow_html=True)
        nodes_examined = st.slider("Regional Nodes Examined", 1, 60, 14)
        nodes_positive = st.slider("Regional Nodes Positive", 0, 46, 2)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("Run Risk Assessment", use_container_width=True)

    with right:
        if predict_clicked:
            # Build model input row
            grade_map = {'1': 1, '2': 2, '3': 3, 'anaplastic; Grade IV': 4}
            t_map = {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4}
            n_map = {'N1': 1, 'N2': 2, 'N3': 3}

            row = {c: 0 for c in COVARIATE_COLS}
            row['Age'] = age
            row['Tumor Size'] = tumor_size
            row['Regional Node Examined'] = nodes_examined
            row['Reginol Node Positive'] = nodes_positive
            row['Grade'] = grade_map[grade]
            row['T Stage'] = t_map[t_stage]
            row['N Stage'] = n_map[n_stage]
            if race == "Other":
                row['Race_Other'] = 1
            elif race == "White":
                row['Race_White'] = 1
            if marital == "Married":
                row['Marital Status_Married'] = 1
            elif marital == "Separated":
                row['Marital Status_Separated'] = 1
            elif marital == "Single":
                row['Marital Status_Single '] = 1
            elif marital == "Widowed":
                row['Marital Status_Widowed'] = 1
            row['A Stage_Regional'] = 1 if a_stage == "Regional" else 0
            row['Progesterone Status_Positive'] = 1 if progesterone == "Positive" else 0
            row[STRATA_COL] = 1 if estrogen == "Positive" else 0

            input_df = pd.DataFrame([row])

            surv_func = cph.predict_survival_function(input_df, times=[60])
            surv_prob = float(surv_func.iloc[0, 0])

            if surv_prob >= 0.90:
                risk_band, badge_class, band_color = "Low", "badge-low", SUCCESS
            elif surv_prob >= 0.70:
                risk_band, badge_class, band_color = "Medium", "badge-medium", WARNING
            else:
                risk_band, badge_class, band_color = "High", "badge-high", DANGER

            predicted_status = "Likely Alive" if surv_prob >= 0.5 else "Likely Deceased"
            confidence = abs(surv_prob - 0.5) * 2 * 100

            st.markdown('<div class="section-card"><div class="section-title">Prediction Results — 60-Month Horizon</div>', unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            r1.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Status</div>
                <div class="metric-value" style="color:{band_color}">{predicted_status}</div>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Survival Probability</div>
                <div class="metric-value">{surv_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            st.write("")
            r3, r4 = st.columns(2)
            r3.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <span class="{badge_class}">{risk_band} Risk</span>
            </div>""", unsafe_allow_html=True)
            r4.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{confidence:.0f}%</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Feature contribution explanation
            st.markdown('<div class="section-card"><div class="section-title">Feature Importance Explanation</div>', unsafe_allow_html=True)

            coefs = cph.params_
            contributions = {}
            for feat in COVARIATE_COLS:
                if feat not in coefs.index:
                    continue
                val = row[feat]
                if feat in NUMERIC_COLS:
                    z = (val - TRAIN_MEANS[feat]) / TRAIN_STDS[feat]
                    contributions[feat] = coefs[feat] * z
                else:
                    contributions[feat] = coefs[feat] * val

            contrib_series = pd.Series(contributions).sort_values(key=abs, ascending=True).tail(8)
            colors = [DANGER if v > 0 else SUCCESS for v in contrib_series.values]

            fig = go.Figure(go.Bar(
                x=contrib_series.values,
                y=contrib_series.index,
                orientation='h',
                marker_color=colors,
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Relative contribution to hazard (log scale)",
                font=dict(color=PRIMARY, size=12),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bars increase predicted risk relative to the training population average; green bars decrease it.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; padding:3rem 1rem; color:{SECONDARY};">
                Complete the patient form and run the assessment to view survival prediction, risk level, and feature contributions.
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 2 — ANALYTICS
# ============================================================================
with tab_analytics:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card"><div class="section-title">Survival Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(data, x="duration", color=data['event'].map({0: 'Alive', 1: 'Dead'}),
                            nbins=30, color_discrete_map={'Alive': SUCCESS, 'Dead': DANGER},
                            labels={'duration': 'Survival Months', 'color': 'Status'})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12), legend_title_text='Status')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card"><div class="section-title">Stage Distribution (T Stage)</div>', unsafe_allow_html=True)
        stage_counts = data['T Stage'].map({1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4'}).value_counts().sort_index()
        fig = px.bar(x=stage_counts.index, y=stage_counts.values,
                     color_discrete_sequence=[PRIMARY],
                     labels={'x': 'T Stage', 'y': 'Patients'})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-card"><div class="section-title">Tumor Size Analysis</div>', unsafe_allow_html=True)
        fig = px.box(data, x=data['event'].map({0: 'Alive', 1: 'Dead'}), y="Tumor Size",
                     color=data['event'].map({0: 'Alive', 1: 'Dead'}),
                     color_discrete_map={'Alive': SUCCESS, 'Dead': DANGER},
                     labels={'x': 'Status', 'Tumor Size': 'Tumor Size (mm)'})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12), showlegend=False,
                           xaxis_title="Status")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card"><div class="section-title">Survival by Age Group</div>', unsafe_allow_html=True)
        age_bins = pd.cut(data['Age'], bins=[20, 40, 50, 60, 70, 100], labels=['20-40', '41-50', '51-60', '61-70', '71+'])
        age_surv = (1 - data.groupby(age_bins, observed=True)['event'].mean()) * 100
        fig = px.bar(x=age_surv.index.astype(str), y=age_surv.values,
                     color_discrete_sequence=[ACCENT],
                     labels={'x': 'Age Group', 'y': 'Survival Rate (%)'})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Survival by Hormone Status</div>', unsafe_allow_html=True)
    hc1, hc2 = st.columns(2)
    with hc1:
        er_surv = (1 - data.groupby('Estrogen Status_Positive')['event'].mean()) * 100
        er_surv.index = er_surv.index.map({0: 'ER Negative', 1: 'ER Positive'})
        fig = px.bar(x=er_surv.index, y=er_surv.values, color_discrete_sequence=[SECONDARY],
                     labels={'x': 'Estrogen Receptor Status', 'y': 'Survival Rate (%)'})
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12))
        st.plotly_chart(fig, use_container_width=True)
    with hc2:
        pr_surv = (1 - data.groupby('Progesterone Status_Positive')['event'].mean()) * 100
        pr_surv.index = pr_surv.index.map({0: 'PR Negative', 1: 'PR Positive'})
        fig = px.bar(x=pr_surv.index, y=pr_surv.values, color_discrete_sequence=[ACCENT],
                     labels={'x': 'Progesterone Receptor Status', 'y': 'Survival Rate (%)'})
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           paper_bgcolor="white", font=dict(color=PRIMARY, size=12))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<p style="text-align:center; color:{SECONDARY}; font-size:0.75rem; margin-top:1rem;">
Model: Stratified Cox Proportional Hazards (stratified by Estrogen Receptor Status) — Test C-index 0.718 |
Trained on SEER Breast Cancer dataset (n={total_patients:,})
</p>
""", unsafe_allow_html=True)
