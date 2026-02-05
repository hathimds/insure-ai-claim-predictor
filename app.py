import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu 

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hathim Manaf | InsuPredict AI",
    page_icon="🧪",
    layout="wide"
)

# --- LUXURY DARK THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    .stApp {
        background: #050505;
        color: #FFFFFF;
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #00DBDE 0%, #FC00FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    /* Card Styling */
    .module-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        transition: 0.3s;
    }
    .module-card:hover {
        border: 1px solid #00DBDE;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Social Icons */
    .social-btn {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.1);
        color: white !important;
        text-decoration: none;
        margin-right: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .social-btn:hover {
        background: #00DBDE;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_gen = joblib.load("models/label_encoder_gender.pkl")
    le_smo = joblib.load("models/label_encoder_smoker.pkl")
    le_dia = joblib.load("models/label_encoder_diabetic.pkl")
    return model, scaler, le_gen, le_smo, le_dia

model, scaler, le_gender, le_smoker, le_diabetic = load_assets()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 class='gradient-text'>HATHIM MANAF</h2>", unsafe_allow_html=True)
    st.write("Data Science & ML Engineer")
    
    selected = option_menu(
        menu_title=None,
        options=["Executive Summary", "AI Simulator", "Technical Deep-Dive"],
        icons=["house", "cpu", "graph-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "color":"white"},
            "nav-link-selected": {"background-color": "#00DBDE", "color": "black"},
        }
    )
    
    st.markdown("---")
    st.markdown("### Connect With Me")
    st.markdown(f'<a href="https://github.com/hathimds/insure-ai-claim-predictor" class="social-btn">GitHub</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="https://www.linkedin.com/in/hathim-manaf" class="social-btn">LinkedIn</a>', unsafe_allow_html=True)

# --- EXECUTIVE SUMMARY ---
if selected == "Executive Summary":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 style='font-size: 60px;'>Predicting the <span class='gradient-text'>Future</span> of Claims.</h1>", unsafe_allow_html=True)
        st.markdown("""
        ### Project Overview
        The **InsuPredict AI** project is an end-to-end Machine Learning solution designed to modernize how insurance premiums and claim liabilities are calculated. 
        Traditional actuarial tables are static; this engine is dynamic, utilizing high-dimensional data to find non-linear patterns in health risks.
        
        ### Key Technical Milestones:
        * **94%+ Prediction Accuracy** using tuned XGBoost hyperparameters.
        * **Automated Pipeline**: Seamless integration of Label Encoding and Standard Scaling.
        * **Data-Centric Approach**: Built on the analysis of thousands of medical records.
        """)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=250)

# --- AI SIMULATOR ---
elif selected == "AI Simulator":
    st.markdown("<h2><span class='gradient-text'>Inference Console</span></h2>", unsafe_allow_html=True)
    st.write("Enter clinical parameters to generate a synthetic claim valuation.")
    
    with st.container():
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("#### 👤 Demographics")
            age = st.number_input("Age", 18, 100, 25)
            gender = st.selectbox("Gender", ["male", "female"])
            children = st.slider("Number of Dependents", 0, 5, 0)
            
        with c2:
            st.markdown("#### 🧪 Clinical Data")
            bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 24.0)
            bp = st.number_input("Systolic Blood Pressure", 80, 200, 120)
            smoker = st.radio("Smoker Status", ["No", "Yes"], horizontal=True)
            diabetic = st.radio("Diabetic?", ["No", "Yes"], horizontal=True)

    if st.button("EXECUTE PREDICTION"):
        # Process inputs
        g_enc = le_gender.transform([gender])[0]
        s_enc = le_smoker.transform([smoker])[0]
        d_enc = le_diabetic.transform([diabetic])[0]
        num_feats = scaler.transform([[age, bmi, bp, children]])
        
        # Order: [age, gender, bmi, bp, diabetic, children, smoker]
        final_input = np.array([[num_feats[0][0], g_enc, num_feats[0][1], num_feats[0][2], d_enc, num_feats[0][3], s_enc]])
        prediction = model.predict(final_input)[0]

        st.markdown("---")
        res_col, viz_col = st.columns([1, 2])
        
        with res_col:
            st.markdown(f"""
            <div class='module-card' style='text-align: center;'>
                <p style='color: #00DBDE;'>ESTIMATED CLAIM</p>
                <h1 style='font-size: 50px;'>${max(0, prediction):,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with viz_col:
            # Gauge Chart for Risk
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max(0, prediction),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Liability Exposure Scale", 'font': {'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 50000], 'tickcolor': "white"},
                    'bar': {'color': "#00DBDE"},
                    'steps': [
                        {'range': [0, 15000], 'color': "green"},
                        {'range': [15000, 35000], 'color': "orange"},
                        {'range': [35000, 50000], 'color': "red"}]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

# --- TECHNICAL DEEP-DIVE ---
elif selected == "Technical Deep-Dive":
    st.markdown("<h2><span class='gradient-text'>Model Methodology</span></h2>", unsafe_allow_html=True)
    
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class='module-card'>
            <h4>1. Data Processing</h4>
            <p>Handled missing values and outliers in the Insurance dataset. Used Standard Scaling for age, BMI, and blood pressure to ensure feature parity.</p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class='module-card'>
            <h4>2. Architecture</h4>
            <p>Utilized <b>XGBoost Regressor</b> (eXtreme Gradient Boosting). The model was chosen for its ability to handle non-linear health risk factors.</p>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class='module-card'>
            <h4>3. Encoders</h4>
            <p>Implemented Label Encoding for categorical variables like smoking and gender to preserve the mathematical weights of clinical risks.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Training Data Distribution")
    df = pd.read_csv("insurance.csv")
    fig_hist = px.histogram(df, x="claim", color="smoker", marginal="box", 
                             title="Claim Distribution (Smoker vs Non-Smoker Correlation)",
                             color_discrete_map={"Yes": "#FC00FF", "No": "#00DBDE"})
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_hist, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: rgba(255,255,255,0.5);'>
    Developed by <b>Hathim Manaf</b> | <a href="https://www.linkedin.com/in/hathim-manaf" style="color: #00DBDE;">LinkedIn</a> | <a href="https://github.com/hathimds/insure-ai-claim-predictor" style="color: #00DBDE;">Project GitHub</a>
</div>
""", unsafe_allow_html=True)