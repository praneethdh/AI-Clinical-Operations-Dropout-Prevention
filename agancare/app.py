"""
app.py — AganCare Sentinel
Main entry point: Overview Dashboard.
Run with: streamlit run app.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from utils.data_loader import load_patients, load_sessions, load_therapists, get_high_risk_patients, mark_patient_dropout
from utils.charts import therapist_load_chart, insurance_donut

st.set_page_config(
    page_title="AganCare Sentinel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

  /* Sidebar branding */
  section[data-testid="stSidebar"] > div:first-child {
    background: #0D1117;
    border-right: 1px solid #21262D;
  }

  .brand-block {
    background: linear-gradient(135deg, #162120 0%, #0D1117 100%);
    border: 1px solid #2D6A4F;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
  }
  .brand-title {
    font-size: 1.25rem; font-weight: 700; color: #52B788; letter-spacing: -0.02em;
  }
  .brand-sub { font-size: 0.75rem; color: #8B949E; margin-top: 2px; }

  .stat-card {
    background: #161B22; border: 1px solid #21262D; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
  }
  .stat-num { font-size: 2rem; font-weight: 700; color: #52B788; }
  .stat-label { font-size: 0.8rem; color: #8B949E; margin-top: 4px; }
  .stat-num-warn { font-size: 2rem; font-weight: 700; color: #E63946; }
  .stat-num-amber { font-size: 2rem; font-weight: 700; color: #F4A261; }

  .section-header {
    font-size: 1rem; font-weight: 600; color: #C9D1D9;
    border-bottom: 1px solid #21262D; padding-bottom: 6px; margin: 1rem 0 0.7rem;
  }

  .alert-pill {
    display: inline-block; background: #3D1A1C; color: #E63946;
    border: 1px solid #E63946; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 700;
  }
  .ok-pill {
    display: inline-block; background: #162120; color: #52B788;
    border: 1px solid #52B788; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 700;
  }
  .amber-pill {
    display: inline-block; background: #3B2A15; color: #F4A261;
    border: 1px solid #F4A261; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 700;
  }

  .dropout-row {
    background: #161B22; border: 1px solid #21262D; border-radius: 8px;
    padding: 0.7rem 1rem; margin-bottom: 0.4rem;
    display: flex; justify-content: space-between; align-items: center;
  }
</style>
""", unsafe_allow_html=True)

from utils.navigation import render_sidebar
render_sidebar()

# ── Load Data ─────────────────────────────────────────────────────────────────
patients_df = load_patients()
sessions_df = load_sessions()
therapists_df = load_therapists()
high_risk_df = get_high_risk_patients(sessions_df, patients_df)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Clinic Overview Dashboard")
st.markdown("<p style='color:#8B949E;margin-top:-10px'>Real-time operational snapshot of AganCare Mental Health Clinic.</p>",
            unsafe_allow_html=True)
st.divider()

# ── Top KPI Row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total_patients = len(patients_df)
latest_sessions = sessions_df.sort_values('session_number').groupby('patient_id').tail(1)
active_patients = len(latest_sessions[latest_sessions["status"] == "Active"])
total_dropouts = sessions_df[sessions_df["did_dropout"] == 1]["patient_id"].nunique()
dropout_rate = round(total_dropouts / total_patients * 100, 1)
active_high_risk = high_risk_df[high_risk_df["status"] == "Active"] if not high_risk_df.empty else pd.DataFrame()
flagged_count = len(active_high_risk[active_high_risk["is_flagged"] == True]) if not active_high_risk.empty else 0
total_sessions = len(sessions_df)
avg_srs = round(sessions_df["srs_overall"].mean(), 1)

with k1:
    st.markdown(f"""<div class='stat-card'>
      <div class='stat-num'>{total_patients}</div>
      <div class='stat-label'>Total Patients</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class='stat-card'>
      <div class='stat-num'>{active_patients}</div>
      <div class='stat-label'>Active in Therapy</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class='stat-card'>
      <div class='stat-num-warn'>{dropout_rate}%</div>
      <div class='stat-label'>Dropout Rate</div>
    </div>""", unsafe_allow_html=True)

with k4:
    num_class = "stat-num-warn" if flagged_count >= 5 else "stat-num-amber" if flagged_count > 0 else "stat-num"
    st.markdown(f"""<div class='stat-card'>
      <div class='{num_class}'>{flagged_count}</div>
      <div class='stat-label'>🚨 At-Risk Flagged</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class='stat-card'>
      <div class='stat-num'>{avg_srs}</div>
      <div class='stat-label'>Avg SRS Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row ────────────────────────────────────────────────────────────────
c_left, c_mid, c_right = st.columns([1.2, 1.2, 1], gap="medium")

with c_left:
    st.markdown("<div class='section-header'>👨‍⚕️ Therapist Patient Load</div>", unsafe_allow_html=True)
    st.plotly_chart(therapist_load_chart(patients_df), use_container_width=True,
                    config={"displayModeBar": False})

with c_mid:
    st.markdown("<div class='section-header'>💳 Insurance Distribution</div>", unsafe_allow_html=True)
    st.plotly_chart(insurance_donut(patients_df), use_container_width=True,
                    config={"displayModeBar": False})
    uninsured = patients_df[patients_df["insurance_status"] == "Uninsured"]
    if len(uninsured) > 0:
        st.markdown(f"<div style='font-size:0.8rem;color:#8B949E;text-align:center'>"
                    f"⚠️ {len(uninsured)} uninsured patients — highest dropout risk group</div>",
                    unsafe_allow_html=True)

with c_right:
    st.markdown("<div class='section-header'>📋 Clinic Stats</div>", unsafe_allow_html=True)
    stats = {
        "Total Sessions": total_sessions,
        "Avg Sessions/Patient": round(total_sessions / total_patients, 1),
        "Therapists on Roster": len(therapists_df),
        "Cities Served": patients_df["city"].nunique(),
        "High PHQ-9 (≥15)": len(patients_df[patients_df["baseline_phq9"] >= 15]),
        "High GAD-7 (≥10)": len(patients_df[patients_df["baseline_gad7"] >= 10]),
    }
    for label, val in stats.items():
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid #21262D;font-size:0.85rem'>
          <span style='color:#8B949E'>{label}</span>
          <span style='color:#E6EDF3;font-weight:600'>{val}</span>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Pagination State ──────────────────────────────────────────────────────────
if "risk_page" not in st.session_state:
    st.session_state.risk_page = 0
if "dropout_page" not in st.session_state:
    st.session_state.dropout_page = 0

# ── Manual Dropout Form has been moved to Dropout Prevention Hub ──────────────

st.divider()

@st.experimental_fragment
def render_risk_list(high_risk_df):
    st.markdown("<div class='section-header'>🚨 Risk Overview (Active Patients)</div>", unsafe_allow_html=True)
    search_risk = st.text_input("🔍 Search Risk List", key="search_risk", placeholder="Enter patient name...")
    
    filtered_risk = high_risk_df[high_risk_df["status"] == "Active"].copy()
    if not filtered_risk.empty and search_risk:
        filtered_risk = filtered_risk[filtered_risk["full_name"].str.contains(search_risk, case=False, na=False)]
        
    if filtered_risk.empty:
        st.markdown("<span class='ok-pill'>✓ All Clear</span> No active patients found at risk.", unsafe_allow_html=True)
    else:
        items_per_page = 5
        total_risk = len(filtered_risk)
        max_pages = max(0, (total_risk - 1) // items_per_page)
        st.session_state.risk_page = min(st.session_state.risk_page, max_pages)
        
        start_idx = st.session_state.risk_page * items_per_page
        end_idx = start_idx + items_per_page
        
        for _, row in filtered_risk.iloc[start_idx:end_idx].iterrows():
            # Pill color logic
            if row["risk_score"] >= 60:
                badge = "alert-pill"
            elif row["risk_score"] >= 35:
                badge = "amber-pill"
            else:
                badge = "ok-pill"
                
            label = f"{int(row['risk_score'])}/100"
            st.markdown(f"""
            <div class='dropout-row'>
              <div>
                <div style='font-weight:600;color:#E6EDF3;font-size:0.9rem'>{row['full_name']}</div>
                <div style='font-size:0.75rem;color:#8B949E'>{row['therapist']} · Sess {int(row['session_number'])}</div>
              </div>
              <span class='{badge}'>{label}</span>
            </div>
            """, unsafe_allow_html=True)
            
        if total_risk > items_per_page:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("⬅️ Prev", key="risk_prev") and st.session_state.risk_page > 0:
                    st.session_state.risk_page -= 1
                    st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;color:#8B949E;font-size:0.8rem;padding-top:8px'>Page {st.session_state.risk_page + 1} of {max_pages + 1}</div>", unsafe_allow_html=True)
            with c3:
                if st.button("Next ➡️", key="risk_next") and st.session_state.risk_page < max_pages:
                    st.session_state.risk_page += 1
                    st.rerun()

@st.experimental_fragment
def render_dropout_list(sessions_df, patients_df):
    st.markdown("<div class='section-header'>📅 Recent Dropout History</div>", unsafe_allow_html=True)
    dropouts_df = sessions_df[sessions_df["did_dropout"] == 1].copy()
    
    if dropouts_df.empty:
        st.markdown("No recorded dropouts yet.")
    else:
        merged = dropouts_df.merge(patients_df[["patient_id", "full_name"]], on="patient_id", how="left")
        merged = merged.sort_values("session_date", ascending=False)
        
        search_drop = st.text_input("🔍 Search Dropouts", key="search_drop", placeholder="Enter patient name...")
        if search_drop:
            merged = merged[merged["full_name"].str.contains(search_drop, case=False, na=False)]
            
        if merged.empty:
            st.markdown("No dropouts match your search.")
        else:
            items_per_page = 5
            total_dropouts = len(merged)
            max_pages = max(0, (total_dropouts - 1) // items_per_page)
            st.session_state.dropout_page = min(st.session_state.dropout_page, max_pages)
            
            start_idx = st.session_state.dropout_page * items_per_page
            end_idx = start_idx + items_per_page
            
            for _, row in merged.iloc[start_idx:end_idx].iterrows():
                st.markdown(f"""
                <div class='dropout-row'>
                  <div>
                    <div style='font-weight:600;color:#E6EDF3;font-size:0.9rem'>{row['full_name']}</div>
                    <div style='font-size:0.75rem;color:#8B949E'>
                      Dropped after Session {int(row['session_number'])} · {row['session_date']}
                    </div>
                  </div>
                  <span style='color:#E63946;font-size:0.85rem'>Dropped</span>
                </div>
                """, unsafe_allow_html=True)
                
            if total_dropouts > items_per_page:
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    if st.button("⬅️ Prev", key="drop_prev") and st.session_state.dropout_page > 0:
                        st.session_state.dropout_page -= 1
                        st.rerun()
                with c2:
                    st.markdown(f"<div style='text-align:center;color:#8B949E;font-size:0.8rem;padding-top:8px'>Page {st.session_state.dropout_page + 1} of {max_pages + 1}</div>", unsafe_allow_html=True)
                with c3:
                    if st.button("Next ➡️", key="drop_next") and st.session_state.dropout_page < max_pages:
                        st.session_state.dropout_page += 1
                        st.rerun()

col_flags, col_recent = st.columns([1, 1], gap="medium")

with col_flags:
    render_risk_list(high_risk_df)

with col_recent:
    render_dropout_list(sessions_df, patients_df)

st.divider()
st.markdown("""
<div style='text-align:center;color:#8B949E;font-size:0.75rem;padding:0.5rem'>
  AganCare Sentinel Prototype · AI-assisted clinical operations · Human approval required for all interventions
</div>
""", unsafe_allow_html=True)
