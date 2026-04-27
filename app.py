import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. SET THE STAGE (PREMIUM UI) ---
st.set_page_config(page_title="ARCH.AI | Admission Predictor", layout="wide", page_icon="🚀")

# Sleek Dark Mode CSS
st.markdown("""
    <style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    .stSidebar { background-color: #111111; }
    h1, h2, h3 { color: #f97316 !important; font-family: 'Inter', sans-serif; }
    .stButton>button {
        background-color: #f97316; color: #000000; font-weight: bold; border: none; width: 100%; border-radius: 5px;
    }
    .stButton>button:hover { background-color: #ffffff; color: #f97316; }
    .stMetric { background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #f97316; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD YOUR BRAINS ---
@st.cache_resource
def load_all_assets():
    reg_model = joblib.load('Models/masters_regression.pkl')
    m_scaler = joblib.load('Models/masters_scaler.pkl')
    u_dict = joblib.load('Models/masters_university_means.pkl')
    f_list = joblib.load('Models/feature_list.pkl')
    return reg_model, m_scaler, u_dict, f_list

model, scaler, univ_means, feature_columns = load_all_assets()

# --- 3. THE INTERFACE ---
st.title("🚀 ARCH.AI | Predictive Engine")
st.markdown("Enter your academic telemetry in the sidebar to calculate your admission trajectory.")

st.sidebar.header("Applicant Telemetry")

uni_list = sorted(list(univ_means.keys()))
default_idx = uni_list.index("Massachusetts Institute of Technology") if "Massachusetts Institute of Technology" in uni_list else 0
target_uni = st.sidebar.selectbox("Target University", options=uni_list, index=default_idx)

gpa = st.sidebar.slider("Undergraduate GPA", 2.0, 4.0, 3.8, 0.1)
gre = st.sidebar.slider("GRE Total Score", 260, 340, 325, 1)

st.sidebar.subheader("Experience & Materials")
res_yrs = st.sidebar.number_input("Research Years", 0, 10, 2)
pubs = st.sidebar.number_input("Publications", 0, 10, 1)
conf = st.sidebar.number_input("Conference Papers", 0, 10, 0)
sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, 4.5, 0.5)
lor_avg = st.sidebar.slider("LOR Avg Strength", 1.0, 5.0, 4.0, 0.5)
lor_cnt = st.sidebar.number_input("Number of LORs", 1, 5, 3)

# --- 4. THE EXECUTION ---
if st.sidebar.button("EXECUTE PREDICTION"):
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    input_df['undergrad_gpa'] = gpa
    input_df['gre_total'] = gre
    input_df['research_experience_years'] = res_yrs
    input_df['publications_count'] = pubs
    input_df['conference_papers'] = conf
    input_df['sop_strength'] = sop
    input_df['lor_avg_strength'] = lor_avg
    input_df['lor_count'] = lor_cnt
    
    input_df['applied_university'] = univ_means.get(target_uni, 0.5)

    input_df['academic_strength'] = gre * gpa
    input_df['application_strength'] = sop + (lor_avg * 2) + lor_cnt
    input_df['research_power'] = res_yrs + (pubs * 3) + (conf * 2)
    input_df['overall_profile_score'] = (input_df['academic_strength'] / 1000) + \
                                         input_df['application_strength'] + \
                                         input_df['research_power']

    num_cols = [
        'undergrad_gpa', 'gre_total', 'gre_verbal', 'gre_quantitative',
        'gre_analytical_writing', 'gmat_total', 'gmat_verbal', 'gmat_quant',
        'toefl_score', 'ielts_score', 'sop_strength', 'sop_word_count',
        'lor_count', 'lor_avg_strength', 'lor_from_professor', 'lor_from_industry',
        'research_experience_years', 'publications_count', 'conference_papers',
        'work_experience_years', 'internships_count', 'work_industry_relevance',
        'applied_university', 'academic_strength', 'application_strength', 
        'research_power', 'overall_profile_score'
    ]
    
    cols_to_scale = [c for c in num_cols if c in feature_columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    prob = model.predict(input_df)[0]
    
    # --- 5. PREMIUM RESULTS DISPLAY ---
    st.markdown("---")
    
    # Create two columns for a wider dashboard layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("System Status")
        # Logic to only show balloons on high probability
        if prob >= 0.75:
            st.success("🟢 Target Reached: High Admission Probability")
            st.balloons() # Only triggers if 75% or higher
        elif prob >= 0.45:
            st.warning("🟡 Target Viable: Moderate Admission Probability")
        else:
            st.error("🔴 Target Risk: Low Admission Probability")
            
        st.markdown(f"""
        **Profile Breakdown:**
        * **Target:** {target_uni}
        * **Academic Score:** {input_df['academic_strength'][0]:.1f}
        * **Research Power:** {input_df['research_power'][0]:.1f}
        """)

    with col2:
        # Build the Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            number = {'suffix': "%", 'font': {'size': 50, 'color': '#ffffff'}},
            title = {'text': "Admission Probability", 'font': {'size': 24, 'color': '#f97316'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#f97316"},
                'bgcolor': "#1a1a1a",
                'borderwidth': 2,
                'bordercolor': "#333333",
                'steps': [
                    {'range': [0, 45], 'color': '#3a0c0c'},
                    {'range': [45, 75], 'color': '#4a3600'},
                    {'range': [75, 100], 'color': '#0d3a14'}],
            }
        ))
        fig.update_layout(paper_bgcolor="#0a0a0a", font={'color': "#ffffff"}, margin=dict(t=50, b=0, l=20, r=20), height=300)
        st.plotly_chart(fig, use_container_width=True)