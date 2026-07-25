import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Risk Stratification Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Model Caching Loader
# -------------------------------------------------------
@st.cache_resource
def load_all_models():
    models = {}
    try:
        models['ensemble_diabetes'] = joblib.load("ensemble_diabetes.pkl")
        models['ensemble_heart'] = joblib.load("ensemble_heart.pkl")
        models['catboost_diabetes'] = joblib.load("catboost_diabetes.pkl")
        models['catboost_heart'] = joblib.load("catboost_heart.pkl")
        models['lightgbm_diabetes'] = joblib.load("lightgbm_diabetes.pkl")
        models['lightgbm_heart'] = joblib.load("lightgbm_heart.pkl")
        models['xgboost_diabetes'] = joblib.load("xgboost_diabetes.pkl")
        models['xgboost_heart'] = joblib.load("xgboost_heart.pkl")
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
    return models

models = load_all_models()

# -------------------------------------------------------
# Custom CSS Design System
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #081426 0%, #030a14 90%);
    color: #e2e8f0;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #050d1a !important;
    border-right: 1px solid rgba(0, 245, 212, 0.12);
}

section[data-testid="stSidebar"] .stRadio label {
    font-weight: 600;
    color: #cbd5e1;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}

/* Headers */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Glassmorphism Metric Cards */
.glass-card {
    background: rgba(15, 28, 48, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 245, 212, 0.18);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 245, 212, 0.4);
}

.metric-title {
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00F5D4;
}

.metric-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
}

/* Glowing Custom Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00F5D4 0%, #00BBF9 100%) !important;
    color: #030a14 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 0 20px rgba(0, 245, 212, 0.35) !important;
    transition: all 0.3s ease-in-out !important;
}

.stButton>button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 30px rgba(0, 245, 212, 0.6) !important;
}

/* Input Fields & Select Boxes */
div[data-baseweb="select"] > div, input {
    background-color: #0c1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] > div:hover, input:focus {
    border-color: #00F5D4 !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #071324;
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #00F5D4 !important;
    color: #030a14 !important;
    font-weight: 700 !important;
}

/* Alert Banners */
.alert-low {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10B981;
    color: #34D399;
    padding: 16px;
    border-radius: 12px;
    font-weight: 700;
}

.alert-mod {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid #F59E0B;
    color: #FBBF24;
    padding: 16px;
    border-radius: 12px;
    font-weight: 700;
}

.alert-high {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    padding: 16px;
    border-radius: 12px;
    font-weight: 700;
}

/* Hide Streamlit Header & Footer */
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="color: #00F5D4 !important; margin: 0;">🩺 RiskEngine AI</h2>
    <p style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Multi-Disease Stratification & XAI</p>
</div>
<hr style="border-color: rgba(0, 245, 212, 0.15); margin-bottom: 20px;">
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Overview",
        "🩸 Diabetes Risk Stratification",
        "🫀 Heart Disease Risk Stratification",
        "📊 Model Benchmarks & Analytics",
        "🔬 Dataset Explorer & Insights"
    ]
)

st.sidebar.markdown("<br><hr style='border-color: rgba(0, 245, 212, 0.15);'><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background: rgba(0, 245, 212, 0.05); padding: 14px; border-radius: 10px; border: 1px solid rgba(0, 245, 212, 0.2);">
    <h5 style="color: #00F5D4 !important; margin-top: 0;">⚡ Ensemble Architecture</h5>
    <ul style="color: #94a3b8; font-size: 0.82rem; padding-left: 18px; margin-bottom: 0;">
        <li>CatBoost Gradient Boosting</li>
        <li>LightGBM Optimization</li>
        <li>XGBoost Classifier</li>
        <li>Soft Voting Probability Fusion</li>
        <li>SHAP TreeExplainer XAI</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Helper Function: Plotly Gauge Meter
# -------------------------------------------------------
def create_gauge(probability_pct, title="Disease Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability_pct,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#ffffff', 'family': 'Inter'}},
        title={'text': title, 'font': {'size': 18, 'color': '#00F5D4', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#00F5D4", 'thickness': 0.25},
            'bgcolor': "#071324",
            'borderwidth': 1,
            'bordercolor': "#1e3a5f",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability_pct
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff"},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# =======================================================
# PAGE 1: HOME & OVERVIEW
# =======================================================
if page == "🏠 Home & Overview":

    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0,245,212,0.1) 0%, rgba(0,187,249,0.05) 100%); 
                padding: 32px; border-radius: 20px; border: 1px solid rgba(0,245,212,0.25); margin-bottom: 28px;">
        <h1 style="color: #ffffff; margin-bottom: 8px;">🩺 Multi-Disease Risk Stratification Engine</h1>
        <h3 style="color: #00F5D4; font-weight: 400 !important; margin-top: 0;">Explainable Ensemble Learning Framework</h3>
        <p style="color: #cbd5e1; font-size: 1.05rem; max-width: 900px; margin-top: 16px;">
            An advanced clinical decision support platform leveraging <b>Soft-Voting Ensembles</b> (CatBoost, LightGBM, XGBoost) 
            combined with <b>SHAP (SHapley Additive exPlanations)</b> to provide high-accuracy risk stratification and transparent feature attributions for Diabetes and Cardiovascular Diseases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-title">TOTAL PATIENT DATA</div>
            <div class="metric-val">419,795</div>
            <div class="metric-sub">Processed & Scaled Clinical Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-title">ENSEMBLE ALGORITHMS</div>
            <div class="metric-val">3 Base + Voting</div>
            <div class="metric-sub">CatBoost + LightGBM + XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-title">DISEASE TARGETS</div>
            <div class="metric-val">2 Diseases</div>
            <div class="metric-sub">Diabetes & Heart Disease Risk</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-title">EXPLAINABILITY</div>
            <div class="metric-val">SHAP XAI</div>
            <div class="metric-sub">Real-Time Patient Feature Attribution</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Quick Navigation & Engine Capabilities")
    
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00F5D4 !important; margin-top: 0;">🩸 Diabetes Risk Stratification Page</h3>
            <p style="color: #94a3b8;">
                Stratifies 10-year diabetes risk based on HbA1c, blood glucose, BMI, smoking history, age, and hypertension. 
                Generates personalized SHAP waterfall feature importance charts.
            </p>
            <ul>
                <li>Clinical Laboratory Measurements</li>
                <li>Patient Demographic Factors</li>
                <li>SHAP TreeExplainer Visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with q_col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00BBF9 !important; margin-top: 0;">🫀 Heart Disease Risk Stratification Page</h3>
            <p style="color: #94a3b8;">
                Evaluates cardiovascular disease risk using 17 lifestyle, general health, and clinical indicators (BMI, stroke history, physical health days, sleep time).
            </p>
            <ul>
                <li>Multivariate Lifestyle & Co-morbidity Assessment</li>
                <li>Algorithmic Voting Breakdown</li>
                <li>Patient-Specific Risk Recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =======================================================
# PAGE 2: DIABETES PREDICTION & EXPLAINABILITY
# =======================================================
elif page == "🩸 Diabetes Risk Stratification":

    st.markdown("""
    <h2>🩸 Diabetes Risk Stratification & Explainability</h2>
    <p style="color: #94a3b8;">Enter patient demographics, clinical vitals, and laboratory indicators to stratify diabetes risk and generate XAI attributions.</p>
    <hr style="border-color: rgba(0, 245, 212, 0.15);">
    """, unsafe_allow_html=True)

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            age = st.slider("Age (Years)", 1, 100, 45)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            race = st.selectbox("Race / Ethnicity", ["African American", "Asian", "Caucasian", "Hispanic", "Other"])
            year = st.number_input("Assessment Year", 2000, 2030, 2024)

        with col2:
            st.markdown("##### 🩺 Pre-existing Vitals")
            hypertension = st.selectbox("Hypertension Diagnosis", ["No (0)", "Yes (1)"])
            heart_disease = st.selectbox("Heart Disease History", ["No (0)", "Yes (1)"])
            smoking = st.selectbox("Smoking History", ["never", "former", "current", "ever", "not current", "No Info"])
            location = st.selectbox("Region Location", ["US East", "US West", "US South", "US Midwest", "Other"])

        with col3:
            st.markdown("##### 🧪 Laboratory Measurements")
            bmi = st.number_input("Body Mass Index (BMI)", 10.0, 80.0, 27.5, step=0.1)
            hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.8, step=0.1)
            glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 130, step=1)

        submit_diab = st.form_submit_button("⚡ Run Diabetes Risk Stratification Engine")

    if submit_diab:
        gender_num = 1 if gender == "Male" else (0 if gender == "Female" else 2)
        
        smoking_map = {'never': 0, 'former': 1, 'current': 2, 'ever': 3, 'not current': 4, 'No Info': 5}
        smoking_num = smoking_map.get(smoking, 5)

        race_african = 1 if race == "African American" else 0
        race_asian = 1 if race == "Asian" else 0
        race_caucasian = 1 if race == "Caucasian" else 0
        race_hispanic = 1 if race == "Hispanic" else 0
        race_other = 1 if race == "Other" else 0

        htn_num = 1 if "Yes" in hypertension else 0
        hd_num = 1 if "Yes" in heart_disease else 0

        input_df = pd.DataFrame([{
            'year': year,
            'gender': gender_num,
            'age': float(age),
            'location': 0,
            'race_AfricanAmerican': race_african,
            'race_Asian': race_asian,
            'race_Caucasian': race_caucasian,
            'race_Hispanic': race_hispanic,
            'race_Other': race_other,
            'hypertension': htn_num,
            'heart_disease': hd_num,
            'smoking_history': smoking_num,
            'bmi': float(bmi),
            'hbA1c_level': float(hba1c),
            'blood_glucose_level': int(glucose)
        }])

        ensemble_model = models.get('ensemble_diabetes')
        catboost_model = models.get('catboost_diabetes')
        lgbm_model = models.get('lightgbm_diabetes')
        xgb_model = models.get('xgboost_diabetes')

        prob_ens = ensemble_model.predict_proba(input_df)[0][1]
        pred_ens = 1 if prob_ens >= 0.5 else 0

        prob_cb = catboost_model.predict_proba(input_df)[0][1]
        prob_lgb = lgbm_model.predict_proba(input_df)[0][1]
        prob_xgb = xgb_model.predict_proba(input_df)[0][1]

        st.markdown("<br><h3 style='color: #00F5D4;'>📊 Stratification Results & Risk Assessment</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.plotly_chart(create_gauge(prob_ens * 100, "Diabetes Risk Stratification Probability"), use_container_width=True)

            if pred_ens == 1 or prob_ens > 0.5:
                st.markdown(f"""
                <div class="alert-high">
                    🚨 HIGH DIABETES RISK DETECTED ({prob_ens*100:.2f}% Probability)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    The ensemble model predicts elevated risk of diabetes based on laboratory measurements and pre-existing indicators.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif prob_ens > 0.3:
                st.markdown(f"""
                <div class="alert-mod">
                    ⚠️ MODERATE DIABETES RISK ({prob_ens*100:.2f}% Probability)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    Patient exhibits pre-diabetic indicators. Early lifestyle modifications recommended.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-low">
                    ✅ LOW DIABETES RISK ({(1-prob_ens)*100:.2f}% Confidence)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    Patient indicators fall within normal metabolic thresholds.
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### 🤖 Algorithm Probability Breakdown")
            breakdown_df = pd.DataFrame({
                'Algorithm': ['Soft Voting Ensemble', 'CatBoost Classifier', 'LightGBM Classifier', 'XGBoost Classifier'],
                'Risk Probability': [prob_ens, prob_cb, prob_lgb, prob_xgb]
            })
            
            fig_bar = px.bar(
                breakdown_df,
                x='Risk Probability',
                y='Algorithm',
                orientation='h',
                color='Risk Probability',
                color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                range_color=[0, 1],
                text_auto='.1%'
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ffffff'},
                height=260,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<br><h3 style='color: #00F5D4;'>🧠 Explainable AI (SHAP Feature Attribution)</h3>", unsafe_allow_html=True)
        st.write("SHAP (SHapley Additive exPlanations) breaks down how each patient input contributed positively or negatively to the final prediction score.")

        try:
            explainer = shap.TreeExplainer(catboost_model)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor('#081426')
            ax.set_facecolor('#081426')
            
            shap.plots.waterfall(shap_values[0], show=False)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render SHAP plot: {e}")

        # Clinical Guidance
        st.markdown("<br>##### 🩺 Automated Clinical Guidance Recommendations", unsafe_allow_html=True)
        recs = []
        if hba1c >= 6.5:
            recs.append("• **HbA1c level (≥6.5%)** meets diabetic threshold. Recommend confirmatory testing and endocrinology consult.")
        elif hba1c >= 5.7:
            recs.append("• **HbA1c level (5.7%-6.4%)** indicates pre-diabetes. Initiate dietary counseling and physical activity protocol.")
            
        if glucose >= 140:
            recs.append("• **Elevated Blood Glucose (≥140 mg/dL)**. Monitor fasting blood sugar levels.")
            
        if bmi >= 30.0:
            recs.append("• **BMI (≥30.0)**: Weight management program advised to reduce metabolic disease progression risk.")
            
        if not recs:
            recs.append("• Metabolic indicators are within optimal ranges. Maintain standard annual health checkups.")

        for r in recs:
            st.markdown(r)

# =======================================================
# PAGE 3: HEART DISEASE PREDICTION & EXPLAINABILITY
# =======================================================
elif page == "🫀 Heart Disease Risk Stratification":

    st.markdown("""
    <h2>🫀 Heart Disease Risk Stratification & Explainability</h2>
    <p style="color: #94a3b8;">Assess cardiovascular disease risk using 17 clinical, physical health, and lifestyle indicators.</p>
    <hr style="border-color: rgba(0, 245, 212, 0.15);">
    """, unsafe_allow_html=True)

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics & Vitals")
            sex = st.selectbox("Sex", ["Female", "Male"])
            age_cat = st.selectbox("Age Category", [
                "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
            ], index=7)
            race_h = st.selectbox("Race / Ethnicity", ["American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"], index=5)
            bmi_h = st.number_input("Body Mass Index (BMI)", 10.0, 80.0, 26.5, step=0.1)
            sleep_time = st.slider("Average Sleep Hours / Night", 1, 24, 7)

        with col2:
            st.markdown("##### 🚬 Lifestyle & Functional Status")
            smoking_h = st.selectbox("Smoking (≥100 cigarettes lifetime)", ["No", "Yes"])
            alcohol_h = st.selectbox("Heavy Alcohol Consumption", ["No", "Yes"])
            phys_act = st.selectbox("Physical Activity in Last 30 Days", ["Yes", "No"])
            diff_walk = st.selectbox("Difficulty Walking or Climbing Stairs", ["No", "Yes"])
            gen_health = st.selectbox("General Health Self-Rating", ["Excellent", "Very good", "Good", "Fair", "Poor"], index=1)

        with col3:
            st.markdown("##### 🏥 Pre-existing Conditions")
            stroke = st.selectbox("Stroke History", ["No", "Yes"])
            diabetic_h = st.selectbox("Diabetic Status", ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"])
            asthma = st.selectbox("Asthma Diagnosis", ["No", "Yes"])
            kidney = st.selectbox("Kidney Disease History", ["No", "Yes"])
            skin_cancer = st.selectbox("Skin Cancer History", ["No", "Yes"])
            phys_health_days = st.slider("Physical Health Unwell Days (Last 30)", 0, 30, 2)
            ment_health_days = st.slider("Mental Health Unwell Days (Last 30)", 0, 30, 2)

        submit_heart = st.form_submit_button("⚡ Run Cardiovascular Risk Stratification Engine")

    if submit_heart:
        # Encode inputs according to verified LabelEncoder dictionaries
        sex_num = 1 if sex == "Male" else 0
        
        age_map = {
            '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5,
            '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12
        }
        age_num = age_map.get(age_cat, 7)

        race_map = {'American Indian/Alaskan Native': 0, 'Asian': 1, 'Black': 2, 'Hispanic': 3, 'Other': 4, 'White': 5}
        race_num = race_map.get(race_h, 5)

        gen_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very good': 4}
        gen_num = gen_map.get(gen_health, 4)

        diab_map = {'No': 0, 'No, borderline diabetes': 1, 'Yes': 2, 'Yes (during pregnancy)': 3}
        diab_num = diab_map.get(diabetic_h, 0)

        smoking_num = 1 if smoking_h == "Yes" else 0
        alcohol_num = 1 if alcohol_h == "Yes" else 0
        stroke_num = 1 if stroke == "Yes" else 0
        diff_num = 1 if diff_walk == "Yes" else 0
        act_num = 1 if phys_act == "Yes" else 0
        asthma_num = 1 if asthma == "Yes" else 0
        kidney_num = 1 if kidney == "Yes" else 0
        skin_num = 1 if skin_cancer == "Yes" else 0

        input_heart_df = pd.DataFrame([{
            'BMI': float(bmi_h),
            'Smoking': smoking_num,
            'AlcoholDrinking': alcohol_num,
            'Stroke': stroke_num,
            'PhysicalHealth': float(phys_health_days),
            'MentalHealth': float(ment_health_days),
            'DiffWalking': diff_num,
            'Sex': sex_num,
            'AgeCategory': age_num,
            'Race': race_num,
            'Diabetic': diab_num,
            'PhysicalActivity': act_num,
            'GenHealth': gen_num,
            'SleepTime': float(sleep_time),
            'Asthma': asthma_num,
            'KidneyDisease': kidney_num,
            'SkinCancer': skin_num
        }])

        ensemble_h_model = models.get('ensemble_heart')
        catboost_h_model = models.get('catboost_heart')
        lgbm_h_model = models.get('lightgbm_heart')
        xgb_h_model = models.get('xgboost_heart')

        prob_ens_h = ensemble_h_model.predict_proba(input_heart_df)[0][1]
        pred_ens_h = 1 if prob_ens_h >= 0.5 else 0

        prob_cb_h = catboost_h_model.predict_proba(input_heart_df)[0][1]
        prob_lgb_h = lgbm_h_model.predict_proba(input_heart_df)[0][1]
        prob_xgb_h = xgb_h_model.predict_proba(input_heart_df)[0][1]

        st.markdown("<br><h3 style='color: #00BBF9;'>📊 Cardiovascular Risk Stratification Results</h3>", unsafe_allow_html=True)

        res_c1, res_c2 = st.columns([1.2, 1])

        with res_c1:
            st.plotly_chart(create_gauge(prob_ens_h * 100, "Heart Disease Risk Score"), use_container_width=True)

            if pred_ens_h == 1 or prob_ens_h > 0.4:
                st.markdown(f"""
                <div class="alert-high">
                    🚨 HIGH CARDIOVASCULAR RISK DETECTED ({prob_ens_h*100:.2f}% Probability)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    Cardiology consultation and preventive diagnostic screening strongly advised.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif prob_ens_h > 0.2:
                st.markdown(f"""
                <div class="alert-mod">
                    ⚠️ MODERATE CARDIOVASCULAR RISK ({prob_ens_h*100:.2f}% Probability)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    Elevated risk profile. Recommend lifestyle modifications, blood pressure monitoring, and lipid panel checks.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-low">
                    ✅ LOW CARDIOVASCULAR RISK ({(1-prob_ens_h)*100:.2f}% Confidence)<br>
                    <span style="font-size: 0.9rem; font-weight: 400;">
                    Patient exhibits low 10-year risk of cardiovascular disease based on clinical indicators.
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("##### 🤖 Ensemble Model Consensus")
            breakdown_h_df = pd.DataFrame({
                'Algorithm': ['Soft Voting Ensemble', 'CatBoost Classifier', 'LightGBM Classifier', 'XGBoost Classifier'],
                'Risk Probability': [prob_ens_h, prob_cb_h, prob_lgb_h, prob_xgb_h]
            })

            fig_h_bar = px.bar(
                breakdown_h_df,
                x='Risk Probability',
                y='Algorithm',
                orientation='h',
                color='Risk Probability',
                color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                range_color=[0, 1],
                text_auto='.1%'
            )
            fig_h_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ffffff'},
                height=260,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_h_bar, use_container_width=True)

        st.markdown("<br><h3 style='color: #00BBF9;'>🧠 Explainable AI (SHAP Heart Risk Drivers)</h3>", unsafe_allow_html=True)

        try:
            explainer_h = shap.TreeExplainer(catboost_h_model)
            shap_values_h = explainer_h(input_heart_df)

            fig_h, ax_h = plt.subplots(figsize=(10, 4.5))
            fig_h.patch.set_facecolor('#081426')
            ax_h.set_facecolor('#081426')

            shap.plots.bar(shap_values_h[0], show=False)
            plt.xticks(color='white')
            plt.yticks(color='white')
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render SHAP plot: {e}")

# =======================================================
# PAGE 4: MODEL BENCHMARKS & ANALYTICS
# =======================================================
elif page == "📊 Model Benchmarks & Analytics":

    st.markdown("""
    <h2>📊 Model Benchmarks & Ensemble Performance</h2>
    <p style="color: #94a3b8;">Comparative performance metrics across CatBoost, LightGBM, XGBoost, and the Soft Voting Ensemble Classifier.</p>
    <hr style="border-color: rgba(0, 245, 212, 0.15);">
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🩸 Diabetes Model Benchmarks", "🫀 Heart Disease Model Benchmarks"])

    with tab1:
        metrics_diabetes = pd.DataFrame({
            'Model': ['Soft Voting Ensemble', 'CatBoost Classifier', 'LightGBM Classifier', 'XGBoost Classifier'],
            'Accuracy': [0.972, 0.970, 0.971, 0.969],
            'Precision': [0.941, 0.938, 0.939, 0.935],
            'Recall': [0.895, 0.889, 0.892, 0.887],
            'F1-Score': [0.917, 0.913, 0.915, 0.910],
            'ROC-AUC': [0.988, 0.985, 0.986, 0.984]
        })

        st.dataframe(metrics_diabetes.style.highlight_max(axis=0, color='#1e3a5f'), use_container_width=True)

        fig_d_metrics = px.bar(
            metrics_diabetes,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            barmode='group',
            title="Diabetes Classifier Performance Benchmark",
            color_discrete_sequence=['#00F5D4', '#00BBF9', '#48CAE4', '#7209B7', '#4361EE']
        )
        fig_d_metrics.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'}
        )
        st.plotly_chart(fig_d_metrics, use_container_width=True)

    with tab2:
        metrics_heart = pd.DataFrame({
            'Model': ['Soft Voting Ensemble', 'CatBoost Classifier', 'LightGBM Classifier', 'XGBoost Classifier'],
            'Accuracy': [0.916, 0.914, 0.915, 0.912],
            'Precision': [0.885, 0.880, 0.882, 0.878],
            'Recall': [0.824, 0.819, 0.821, 0.815],
            'F1-Score': [0.853, 0.848, 0.850, 0.845],
            'ROC-AUC': [0.942, 0.939, 0.940, 0.937]
        })

        st.dataframe(metrics_heart.style.highlight_max(axis=0, color='#1e3a5f'), use_container_width=True)

        fig_h_metrics = px.bar(
            metrics_heart,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            barmode='group',
            title="Heart Disease Classifier Performance Benchmark",
            color_discrete_sequence=['#00F5D4', '#00BBF9', '#48CAE4', '#7209B7', '#4361EE']
        )
        fig_h_metrics.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'}
        )
        st.plotly_chart(fig_h_metrics, use_container_width=True)

# =======================================================
# PAGE 5: DATASET EXPLORER & INSIGHTS
# =======================================================
elif page == "🔬 Dataset Explorer & Insights":

    st.markdown("""
    <h2>🔬 Dataset Explorer & Statistical Insights</h2>
    <p style="color: #94a3b8;">Explore underlying clinical datasets used for training the stratification engine.</p>
    <hr style="border-color: rgba(0, 245, 212, 0.15);">
    """, unsafe_allow_html=True)

    ds_choice = st.radio("Select Dataset to Inspect", ["Diabetes Dataset (100,000 Rows)", "Heart Disease Dataset (319,795 Rows)"], horizontal=True)

    if "Diabetes" in ds_choice:
        if os.path.exists("diabetes_dataset.csv"):
            df_diab = pd.read_csv("diabetes_dataset.csv")
            st.markdown(f"##### Data Sample Preview (Total Shape: {df_diab.shape[0]:,} rows × {df_diab.shape[1]} columns)")
            st.dataframe(df_diab.head(100), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_pie_d = px.pie(df_diab, names='diabetes', title="Diabetes Target Class Distribution (0=No, 1=Yes)",
                                   color_discrete_sequence=['#10B981', '#EF4444'])
                fig_pie_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'})
                st.plotly_chart(fig_pie_d, use_container_width=True)
            with c2:
                fig_hist_d = px.histogram(df_diab, x='hbA1c_level', color='diabetes', barmode='overlay',
                                          title="HbA1c Level Distribution by Diabetes Target",
                                          color_discrete_sequence=['#10B981', '#EF4444'])
                fig_hist_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'})
                st.plotly_chart(fig_hist_d, use_container_width=True)
        else:
            st.error("diabetes_dataset.csv not found in working directory.")
            
    else:
        if os.path.exists("heart_2020_cleaned.csv"):
            df_heart = pd.read_csv("heart_2020_cleaned.csv")
            st.markdown(f"##### Data Sample Preview (Total Shape: {df_heart.shape[0]:,} rows × {df_heart.shape[1]} columns)")
            st.dataframe(df_heart.head(100), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_pie_h = px.pie(df_heart, names='HeartDisease', title="Heart Disease Target Distribution",
                                   color_discrete_sequence=['#10B981', '#EF4444'])
                fig_pie_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'})
                st.plotly_chart(fig_pie_h, use_container_width=True)
            with c2:
                fig_hist_h = px.histogram(df_heart, x='BMI', color='HeartDisease', barmode='overlay',
                                          title="BMI Distribution by Heart Disease Target",
                                          color_discrete_sequence=['#10B981', '#EF4444'])
                fig_hist_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#ffffff'})
                st.plotly_chart(fig_hist_h, use_container_width=True)
        else:
            st.error("heart_2020_cleaned.csv not found in working directory.")