import streamlit as st
import pandas as pd
from utils.navigation import render_sidebar
from utils.data_loader import DATA_DIR
from utils.matchmaker import match_therapist_risk_adjusted, get_match_reasoning

render_sidebar()

st.title("🧩 Intake Matchmaker")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(145deg, #1A1F26 0%, #11151A 100%);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.risk-high { border-left: 5px solid #FF4B4B; }
.risk-medium { border-left: 5px solid #FACA15; }
.risk-low { border-left: 5px solid #22C55E; }

.t-card {
    background: #1A1F26;
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.t-card:hover {
    transform: translateY(-3px);
    border-color: #3B82F6;
    box-shadow: 0 10px 15px rgba(0,0,0,0.4);
}
.t-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #E2E8F0;
    margin-bottom: 4px;
}
.t-match {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    color: #4ADE80;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
}
.t-spec {
    color: #94A3B8;
    font-size: 0.9rem;
    margin-bottom: 8px;
}
.t-reason {
    background: rgba(59, 130, 246, 0.1);
    border-left: 3px solid #3B82F6;
    padding: 12px;
    margin-top: 15px;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #BFDBFE;
}
</style>
""", unsafe_allow_html=True)

import os

# Load necessary data
therapists_df = pd.read_csv(os.path.join(DATA_DIR, "therapists.csv"))

# Initialize state
if "pending_match" not in st.session_state:
    st.session_state["pending_match"] = None
if "assigned_success" not in st.session_state:
    st.session_state["assigned_success"] = None

if st.session_state["assigned_success"]:
    st.success(st.session_state["assigned_success"])
    st.session_state["assigned_success"] = None

if not st.session_state["pending_match"]:
    st.markdown("### New Patient Intake Form")
    with st.form("intake_form"):
        full_name = st.text_input("Full Name", key="intake_name")
        phone = st.text_input("Phone Number", key="intake_phone")
        city = st.text_input("City", key="intake_city")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=30, key="intake_age")
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary", "Other"], key="intake_gender")
            pref_gender = st.selectbox("Therapist Gender Preference", ["Any", "Female Therapist", "Male Therapist"], key="intake_pref")
        with col2:
            insurance = st.selectbox("Insurance", ["Insured", "Uninsured", "Out-of-Pocket"], key="intake_ins")
            phq9 = st.slider("Initial PHQ-9 (Depression)", 0, 27, 10, key="intake_phq9")
            gad7 = st.slider("Initial GAD-7 (Anxiety)", 0, 21, 10, key="intake_gad7")

        statement = st.text_area("Patient Intake Statement / Goals", key="intake_statement")
        submit = st.form_submit_button("Find Match & Register Patient", type="primary")

    if submit:
        if not full_name:
            st.error("❌ Please provide a name.")
        elif not statement:
            st.error("❌ Please provide an intake statement.")
        else:
            with st.spinner("Analyzing risk and matching with best therapist..."):
                match_res = match_therapist_risk_adjusted(
                    intake_text=statement,
                    therapists_df=therapists_df,
                    pref_gender=pref_gender,
                    age=age,
                    gender=gender,
                    insurance=insurance,
                    phq9=phq9,
                    gad7=gad7
                )
                
                # Store in session state instead of assigning immediately
                st.session_state["pending_match"] = {
                    "full_name": full_name,
                    "phone": phone,
                    "city": city,
                    "age": age,
                    "gender": gender,
                    "pref_gender": pref_gender,
                    "insurance": insurance,
                    "phq9": phq9,
                    "gad7": gad7,
                    "statement": statement,
                    "match_res": match_res,
                    "risk": match_res['risk']
                }
                st.rerun()
            
if st.session_state["pending_match"]:
    pm = st.session_state["pending_match"]
    match_res = pm["match_res"]
    risk = pm["risk"]
    
    st.markdown("---")
    
    # Custom CSS logic for risk color class
    rclass = "risk-high" if risk['level'] == 'HIGH' else ("risk-medium" if risk['level'] == 'MEDIUM' else "risk-low")
    color = "#FF4B4B" if risk['level'] == 'HIGH' else ("#FACA15" if risk['level'] == 'MEDIUM' else "#22C55E")
    
    st.markdown(f"""
    <div class='metric-card {rclass}'>
        <h4 style='margin-top:0; color:#E2E8F0;'>🚨 Baseline Risk Analysis</h4>
        <div style='font-size:1.1rem; margin-bottom:12px;'>
            Calculated Flight Risk: <strong style='color:{color};'>{risk['level']} ({risk['score']} / 100)</strong>
        </div>
    """, unsafe_allow_html=True)
    
    if risk['factors']:
        for f, s in risk['factors']:
            st.markdown(f"- {f} (+{s} pts)")
            
    st.markdown("</div>", unsafe_allow_html=True)
            
    if match_res['override_active']:
        st.warning("⚡ **Risk Override Active:** Due to high baseline dropout risk, semantic matching was adjusted to prioritize therapists with the highest historical retention rates.")
    
    colTitle, colCancel = st.columns([3, 1])
    with colTitle:
        st.markdown(f"<h3 style='margin-top: 30px;'>🤝 Top 5 Recommended Therapists for {pm['full_name']}</h3>", unsafe_allow_html=True)
    with colCancel:
        st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
        if st.button("Cancel / New Search"):
            st.session_state["pending_match"] = None
            st.rerun()
    
    # Show Top 5
    top_5 = match_res['matches'].head(5)
    
    for idx, top_match in top_5.iterrows():
        with st.container():
            st.markdown("<div class='t-card'>", unsafe_allow_html=True)
            colA, colB = st.columns([3, 1])
            
            with colA:
                st.markdown(f"<div class='t-name'>{top_match['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='t-match'>🎯 Match Quality: {top_match['match_pct']}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='t-spec'><strong>Specialty:</strong> {top_match['specialty']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='t-spec'><strong>Clinical Approach:</strong> {top_match['style_keywords']}</div>", unsafe_allow_html=True)
                
                reason = get_match_reasoning(pm['statement'], top_match['style_keywords'])
                st.markdown(f"<div class='t-reason'><strong>🤖 SHAP Feature Reasoning:</strong><br>{reason}</div>", unsafe_allow_html=True)
                
            with colB:
                st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
                if st.button("Assign this DR", key=f"assign_{top_match['therapist_id']}", type="primary", use_container_width=True):
                    # Save to Database
                    import datetime
                    patients_path = os.path.join(DATA_DIR, "patients_intake.csv")
                    patients_df = pd.read_csv(patients_path)
                    
                    if not patients_df.empty:
                        last_id = patients_df["patient_id"].str.extract(r'P(\d+)').astype(float).max().values[0]
                        new_id = f"P{int(last_id) + 1:03d}"
                    else:
                        new_id = "P001"
                        
                    clean_phone = pm['phone'].strip().replace('+', '')
                    if len(clean_phone) == 10:
                        clean_phone = f"+91{clean_phone}"
                    elif len(clean_phone) == 12 and clean_phone.startswith("91"):
                        clean_phone = f"+{clean_phone}"
                    else:
                        clean_phone = f"+{clean_phone}"
                        
                    new_row = pd.DataFrame([{
                        "patient_id": new_id,
                        "full_name": pm['full_name'],
                        "age": pm['age'],
                        "gender": pm['gender'],
                        "insurance_status": pm['insurance'],
                        "baseline_phq9": pm['phq9'],
                        "baseline_gad7": pm['gad7'],
                        "therapist_pref_gender": pm['pref_gender'],
                        "intake_text": pm['statement'],
                        "assigned_therapist_id": top_match['therapist_id'],
                        "assigned_therapist_name": top_match['name'],
                        "intake_date": datetime.date.today().isoformat(),
                        "phone": clean_phone,
                        "city": pm['city'],
                        "baseline_flight_risk_score": risk['score']
                    }])
                    
                    patients_df = pd.concat([patients_df, new_row], ignore_index=True)
                    patients_df.to_csv(patients_path, index=False)
                    
                    st.cache_data.clear()
                    st.session_state["pending_match"] = None
                    st.session_state["assigned_success"] = f"✅ Successfully registered {pm['full_name']} and assigned to {top_match['name']}!"
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
