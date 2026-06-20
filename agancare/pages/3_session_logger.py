"""
pages/3_session_logger.py
A dedicated page for logging individual therapy sessions.
"""
import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

from utils.navigation import render_sidebar
from utils.data_loader import load_patients, load_sessions, DATA_DIR

st.set_page_config(page_title="Session Logger — AganCare", page_icon="📝", layout="wide")
render_sidebar()

st.markdown("## 📝 Session Logger")
st.markdown("<p style='color:#8B949E;margin-top:-10px'>Log new therapy sessions and update patient status.</p>", unsafe_allow_html=True)
st.divider()

patients_df = load_patients()
sessions_df = load_sessions()

if patients_df.empty:
    st.warning("No patients found in the database. Please use the Matchmaker to add patients first.")
    st.stop()

# ── Select Patient ──────────────────────────────────────────────────────────
# Format dropdown options as: P001 - John Doe (Assigned to: Dr. Smith)
patient_options = []
for _, row in patients_df.iterrows():
    patient_options.append(f"{row['patient_id']} - {row['full_name']} ({row['assigned_therapist_name']})")

selected_option = st.selectbox("Select Patient", options=patient_options)
selected_pid = selected_option.split(" - ")[0]

# Get patient's existing sessions
p_sessions = sessions_df[sessions_df["patient_id"] == selected_pid].sort_values("session_date")
last_session_date = None
session_num = 1

if not p_sessions.empty:
    last_session = p_sessions.iloc[-1]
    last_session_date = pd.to_datetime(last_session["session_date"]).date()
    session_num = len(p_sessions) + 1
    
    st.info(f"📅 **Last Session:** {last_session_date.strftime('%b %d, %Y')} (Session {session_num - 1}) • **Current Status:** {last_session['status']}")
else:
    st.info(f"📅 **New Patient:** This will be Session 1.")

st.markdown("### Log New Session Details")

with st.form("session_log_form"):
    c1, c2, c3 = st.columns(3)
    
    session_date = c1.date_input("Session Date", value=datetime.date.today())
    no_shows = c2.number_input("No-Shows since last session", min_value=0, max_value=10, value=0)
    status = c3.selectbox("Patient Status Update", options=["Active", "Graduated", "Dropped"])
    
    st.markdown("#### Session Rating Scale (SRS)")
    s1, s2, s3 = st.columns(3)
    srs_rel = s1.slider("Relationship (I felt heard, understood, and respected)", 0.0, 10.0, 8.0, 0.1)
    srs_goals = s2.slider("Goals/Topics (We worked on what I wanted)", 0.0, 10.0, 8.0, 0.1)
    srs_app = s3.slider("Approach/Method (The therapist's approach is a good fit)", 0.0, 10.0, 8.0, 0.1)
    
    st.markdown("#### Clinical Notes")
    feedback = st.text_area("Session Notes / Feedback (Optional)", placeholder="e.g. Patient showed improvement in anxiety management...")
    
    submitted = st.form_submit_button("💾 Save Session to Database", type="primary", use_container_width=True)
    
    if submitted:
        srs_overall = round(np.mean([srs_rel, srs_goals, srs_app]), 1)
        
        days_since = 0
        if last_session_date:
            days_since = (session_date - last_session_date).days
            if days_since < 0:
                st.error("Session Date cannot be earlier than the last session date.")
                st.stop()
                
        therapist_id = patients_df[patients_df["patient_id"] == selected_pid].iloc[0]["assigned_therapist_id"]
        did_dropout = 1 if status == "Dropped" else 0
        
        new_row = {
            "patient_id": selected_pid,
            "session_number": session_num,
            "session_date": session_date.isoformat(),
            "days_since_last_session": days_since,
            "no_show_count": no_shows,
            "srs_relationship": srs_rel,
            "srs_goals": srs_goals,
            "srs_approach": srs_app,
            "srs_overall": srs_overall,
            "optional_feedback": feedback.replace("\n", " "),
            "phq9_this_session": "",
            "therapist_id": therapist_id,
            "did_dropout": did_dropout,
            "status": status
        }
        
        path = os.path.join(DATA_DIR, "session_logs.csv")
        df = pd.read_csv(path)
        
        # Append
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(path, index=False)
        
        st.cache_data.clear()
        st.success(f"Session {session_num} successfully logged for {selected_pid}!")
