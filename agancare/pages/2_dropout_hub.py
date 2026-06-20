"""
pages/2_dropout_hub.py
Dropout Prevention Hub — ML-powered core feature.
Uses: Random Forest for risk probability + Free API for AI drafting + Twilio for WhatsApp.
Includes: Unilateral Graduation safeguard + 3 ranked intervention options.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_patients, load_sessions, get_patient_sessions,
    get_high_risk_patients, get_model_metrics,
    train_dropout_model, encode_gender, encode_insurance,
    mark_patient_dropout
)
from utils.charts import srs_trend_chart
from utils.message_templates import draft_intervention_message, explain_risk_reasoning
from utils.twilio_whatsapp import render_send_button

st.set_page_config(page_title="Dropout Hub — AganCare", page_icon="🛡️", layout="wide")

from utils.navigation import render_sidebar
render_sidebar()

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .risk-badge-high {
    background: #3D1A1C; color: #E63946; border: 1px solid #E63946;
    border-radius: 20px; padding: 2px 12px; font-size: 0.8rem; font-weight: 700;
  }
  .risk-badge-med {
    background: #2E2010; color: #F4A261; border: 1px solid #F4A261;
    border-radius: 20px; padding: 2px 12px; font-size: 0.8rem; font-weight: 700;
  }
  .risk-badge-grad {
    background: #0D2A3D; color: #74C0FC; border: 1px solid #74C0FC;
    border-radius: 20px; padding: 2px 12px; font-size: 0.8rem; font-weight: 700;
  }
  .action-card {
    background: #161B22; border: 1px solid #2D6A4F;
    border-radius: 10px; padding: 1.2rem; margin-top: 0.5rem;
  }
  .action-card-grad {
    background: #161B22; border: 1px solid #74C0FC;
    border-radius: 10px; padding: 1.2rem; margin-top: 0.5rem;
  }
  .ai-label {
    font-size: 0.75rem; color: #8B949E; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .reason-chip {
    display: inline-block; background: #1F2937; color: #D1D5DB;
    border-radius: 6px; padding: 3px 9px; font-size: 0.78rem; margin: 2px;
  }
  .ml-badge {
    display: inline-block; background: #162120; color: #52B788;
    border: 1px solid #2D6A4F; border-radius: 6px;
    padding: 3px 10px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.04em;
  }
  .prob-score {
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🛡️ Dropout Prevention Hub")
st.markdown(
    "<p style='color:#8B949E;margin-top:-10px'>"
    "ML-powered dropout prediction. Review risk, choose intervention, send via WhatsApp.</p>",
    unsafe_allow_html=True,
)
st.divider()

patients_df = load_patients()
sessions_df = load_sessions()
high_risk_df = get_high_risk_patients(sessions_df, patients_df)
model_metrics = get_model_metrics()

# ── Summary Metrics ────────────────────────────────────────────────────────────
total_active = sessions_df[sessions_df["status"] == "Active"]["patient_id"].nunique()
total_dropout = sessions_df[sessions_df["did_dropout"] == 1]["patient_id"].nunique()
flagged = len(high_risk_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Patients", total_active)
m2.metric("Total Dropouts", total_dropout, delta=f"-{total_dropout}", delta_color="inverse")
m3.metric("🚨 Flagged Today", flagged, delta="needs attention" if flagged > 0 else "all clear")
m4.metric(
    "Model ROC-AUC",
    f"{model_metrics['roc_auc']:.2f}",
    delta="Random Forest",
)

# Show ML model info
with st.expander("📊 ML Model Performance Details", expanded=False):
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("ROC-AUC", f"{model_metrics['roc_auc']:.4f}")
    mc2.metric("Dropout Precision", f"{model_metrics['dropout_precision']:.1%}")
    mc3.metric("Dropout Recall", f"{model_metrics['dropout_recall']:.1%}")
    mc4.metric("Dropout F1", f"{model_metrics['dropout_f1']:.4f}")

    st.markdown(f"""
    <div style='font-size:0.82rem;color:#8B949E;margin-top:0.5rem'>
      Trained on <strong>{model_metrics['training_samples']}</strong> session records
      ({model_metrics['dropout_count']} dropout events) ·
      Flagging threshold: <strong>P<sub>dropout</sub> ≥ {model_metrics['threshold']:.0%}</strong> ·
      Algorithm: RandomForest (100 trees, balanced class weights)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Top Feature Importances:**")
    importances = model_metrics.get("feature_importances", {})
    sorted_feats = sorted(importances.items(), key=lambda x: -x[1])[:7]
    for feat, imp in sorted_feats:
        bar_w = int(imp * 500)
        st.markdown(
            f"<div style='display:flex;align-items:center;margin:2px 0;font-size:0.82rem'>"
            f"<span style='width:140px;color:#C9D1D9'>{feat}</span>"
            f"<div style='background:#21262D;border-radius:3px;height:8px;flex:1;margin:0 8px'>"
            f"<div style='background:#52B788;width:{bar_w}px;max-width:100%;height:8px;border-radius:3px'></div>"
            f"</div>"
            f"<span style='color:#8B949E;width:50px;text-align:right'>{imp:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

with st.expander("🧪 Manual Risk Simulator (Test the ML Model)", expanded=False):
    st.markdown("Enter hypothetical patient data to see how the Random Forest model evaluates dropout risk. This bypasses the CSV and queries the model directly in real-time.")
    with st.form("sim_form"):
        sc1, sc2, sc3 = st.columns(3)
        sim_age = sc1.number_input("Age", 18, 80, 22)
        sim_gender = sc2.selectbox("Gender", ["Male", "Female"])
        sim_insurance = sc3.selectbox("Insurance", ["Insured", "Out-of-Pocket", "Uninsured"], index=2)
        
        sc4, sc5, sc6 = st.columns(3)
        sim_phq9 = sc4.slider("Baseline PHQ-9 (Depression)", 0, 27, 18)
        sim_gad7 = sc5.slider("Baseline GAD-7 (Anxiety)", 0, 21, 15)
        sim_no_shows = sc6.number_input("No Shows", 0, 10, 1)
        
        st.markdown("#### Session Rating Scale (SRS)")
        sc7, sc8 = st.columns(2)
        sim_srs_prev = sc7.slider("Previous SRS Overall", 0.0, 10.0, 8.5)
        sim_srs_curr = sc8.slider("Current SRS Overall", 0.0, 10.0, 6.0)
        
        sim_goals_prev = sc7.slider("Previous SRS Goals", 0.0, 10.0, 8.0)
        sim_goals_curr = sc8.slider("Current SRS Goals", 0.0, 10.0, 5.0)
        
        sim_srs_rel = sc7.slider("Current SRS Relationship", 0.0, 10.0, 6.5)
        sim_srs_app = sc8.slider("Current SRS Approach", 0.0, 10.0, 6.0)
        
        sim_submit = st.form_submit_button("Evaluate Patient Risk", type="primary")
        
    if sim_submit:
        # Build features exactly as the model expects
        sim_features = pd.DataFrame([{
            "age": sim_age,
            "gender": encode_gender(sim_gender),
            "insurance": encode_insurance(sim_insurance),
            "phq9": sim_phq9,
            "gad7": sim_gad7,
            "session_number": 3,
            "days_since_last": 7,
            "no_shows": sim_no_shows,
            "srs_overall": sim_srs_curr,
            "srs_goals": sim_goals_curr,
            "srs_relationship": sim_srs_rel,
            "srs_approach": sim_srs_app,
            "srs_drop": sim_srs_prev - sim_srs_curr,
            "goals_drop": sim_goals_prev - sim_goals_curr,
        }])
        
        # We can safely call train_dropout_model() because it's cached via @st.cache_resource
        model, _ = train_dropout_model()
        
        # Get probability of class 1 (Dropout)
        sim_prob = model.predict_proba(sim_features)[0, 1]
        sim_score = round(sim_prob * 100, 1)
        
        st.markdown(f"### Predicted ML Dropout Risk: **{sim_score}%**")
        
        if sim_score >= (model_metrics['threshold'] * 100):
            st.error(f"🚨 **FLAGGED AS HIGH RISK** (Threshold is {model_metrics['threshold']*100:.0f}%)")
            
            # Show the generated rule-based message
            sim_reasons = []
            if (sim_goals_prev - sim_goals_curr) >= 1.5:
                sim_reasons.append("Goals score dropped")
            if sim_no_shows >= 1:
                sim_reasons.append("no-show")
                
            sim_msg = draft_intervention_message(
                patient_name="Alex (Simulator)", 
                risk_reasons=" | ".join(sim_reasons), 
                latest_feedback="", 
                therapist_name="Dr. Tester", 
                session_number=3, 
                srs_overall=sim_srs_curr, 
                no_shows=sim_no_shows
            )
            st.markdown("**Rule-Based Suggested Intervention Message:**")
            st.info(sim_msg)
        else:
            st.success(f"✅ **LOW RISK** (Below {model_metrics['threshold']*100:.0f}% threshold)")

st.divider()


# ── Risk Analysis Hub ──────────────────────────────────────────────────────────
st.markdown("### 🚨 Patient Risk Analysis")

if high_risk_df.empty:
    st.success("✅ No patients with enough data to analyze yet.")
    st.stop()

# Only show Active patients
filtered_df = high_risk_df[high_risk_df["status"] == "Active"]

if filtered_df.empty:
    st.info("No Active patients available for analysis.")
    st.stop()

# Sort by risk score descending
filtered_df = filtered_df.sort_values(by="risk_score", ascending=False)

# 2. Selectbox
options = []
for idx, row in filtered_df.iterrows():
    options.append(f"{row['full_name']} — Risk: {row['risk_score']}%")

selected_option = st.selectbox("Select Patient to Analyze", options=options)

# 3. Selected Patient Detail View
selected_name = selected_option.split(" — ")[0].strip()
patient_row = filtered_df[filtered_df["full_name"] == selected_name].iloc[0]
patient_id = patient_row["patient_id"]
patient_sessions = get_patient_sessions(sessions_df, patient_id)
is_grad = patient_row.get("is_graduation_candidate", False)

st.markdown(f"#### {patient_row['full_name']}")

# ML Probability
has_data = patient_row.get("has_enough_data", False)
if has_data:
    prob_color = "#74C0FC" if is_grad else ("#E63946" if patient_row["risk_score"] >= 60 else "#F4A261")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-top:-10px;margin-bottom:8px'>"
        f"<span style='font-size:0.85rem;color:#8B949E'>{patient_row['therapist']} · Session {int(patient_row['session_number'])}</span>"
        f"<span class='ml-badge'>ML Prediction</span>"
        f"<span style='font-size:1.2rem;font-weight:800;color:{prob_color}'>"
        f"P<sub>dropout</sub> = {patient_row['risk_score']}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-top:-10px;margin-bottom:8px'>"
        f"<span style='font-size:0.85rem;color:#8B949E'>{patient_row['therapist']} · Session {int(patient_row['session_number'])}</span>"
        f"<span class='ml-badge' style='background-color:#4B5563;'>Need 2+ Sessions</span>"
        f"<span style='font-size:1.0rem;font-weight:600;color:#9CA3AF'>"
        f"Insufficient Data for ML Prediction</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

if is_grad:
    st.markdown(
        "<div style='background:#0D2A3D;border:1px solid #74C0FC;border-radius:8px;"
        "padding:0.8rem 1rem;margin-bottom:0.8rem'>"
        "<div style='font-weight:700;color:#74C0FC;font-size:0.95rem'>"
        "🎓 Possible Unilateral Graduation — Flight to Health</div>"
        "<div style='font-size:0.82rem;color:#8B949E;margin-top:4px'>"
        "This patient has high SRS scores (≥8) but missed sessions. "
        "They may have self-graduated. Recommend sending a graduation congratulations "
        "and closure survey instead of a crisis intervention.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# SHAP Reason Summary & Chips
reasons = patient_row["reasons"]

# Generate a plain English summary of the SHAP reasons
st.markdown("<div class='ai-label'>🧠 AI Model Reasoning (SHAP Analysis)</div>", unsafe_allow_html=True)
if not has_data:
    summary = "The ML model requires at least two recorded sessions to measure a trend in SRS scores. Please log a second session to enable predictive analytics."
elif reasons:
    factors = [f"<b>{r.lower()}</b>" for r in reasons]
    if len(factors) == 1:
        factors_str = factors[0]
    elif len(factors) == 2:
        factors_str = f"{factors[0]} and {factors[1]}"
    else:
        factors_str = ", ".join(factors[:-1]) + f", and {factors[-1]}"
        
    summary = f"Based on the patient's data, the Random Forest model calculates a <b>{patient_row['risk_score']}%</b> chance of dropout. The top contributing factors are {factors_str}."
else:
    summary = f"Based on the patient's data, the Random Forest model calculates a <b>{patient_row['risk_score']}%</b> chance of dropout based on a general pattern of disengagement."

st.markdown(f"<p style='font-size:0.9rem; color:#C9D1D9; margin-bottom: 12px;'>{summary}</p>", unsafe_allow_html=True)

chips = " ".join([f"<span class='reason-chip'>{r}</span>" for r in reasons if r])
st.markdown(f"<div style='margin-bottom:0.8rem'>{chips}</div>", unsafe_allow_html=True)

st.divider()

# 4. Graph & Intervention Side-by-Side
col_graph, col_action = st.columns([1.2, 1.5], gap="large")

with col_graph:
    st.markdown("##### 📈 SRS Trend History")
    st.plotly_chart(
        srs_trend_chart(patient_sessions, patient_row["full_name"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    
    feedback = patient_row.get("latest_feedback", "")
    if feedback and len(str(feedback)) > 3:
        st.markdown("<div class='ai-label'>Patient's last feedback</div>", unsafe_allow_html=True)
        st.markdown(f'> *"{feedback}"* ')

with col_action:
    # ── AI Action Card ─────────────────────────────────────────────────────────
    card_class = "action-card-grad" if is_grad else "action-card"
    st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)

    if not has_data:
        st.markdown("##### ⏳ Awaiting More Data")
        st.caption("Intervention tools will unlock once the patient has logged at least 2 sessions.")
    elif is_grad:
        st.markdown("##### 🎓 Graduation Action Card")
        st.caption("This patient may have self-graduated. Send congratulations instead of an intervention.")
    else:
        st.markdown("##### 🤖 AI Intervention Suggestion")
        st.caption("Choose an intervention strategy. AI drafts the message. You review and edit. Twilio sends the real WhatsApp.")

    if not has_data:
        st.info("Log a second session via 'Log a Session' to unlock predictive messaging.")
    elif is_grad:
        first_name = patient_row["full_name"].split()[0]
        grad_draft = (
            f"Hi {first_name}, I noticed we haven't connected recently. "
            f"I want you to know how proud I am of the progress you've made. "
            f"If you feel you're in a good place, I'd love to formally celebrate "
            f"that milestone. Would you like to schedule a brief graduation session, "
            f"or would you prefer I send you a short closure survey? Either way, "
            f"my door is always open. 😊"
        )
        if f"grad_draft_{patient_id}" not in st.session_state:
            st.session_state[f"grad_draft_{patient_id}"] = grad_draft

        st.markdown("<div class='ai-label'>Graduation message — edit before sending</div>", unsafe_allow_html=True)
        editable = st.text_area("Edit graduation message", value=st.session_state[f"grad_draft_{patient_id}"], height=130, key=f"grad_edit_{patient_id}", label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("**📲 Send via Twilio WhatsApp Sandbox**")
        phone = str(patient_row.get("phone", ""))
        render_send_button(patient_row["full_name"], phone, editable, f"twilio_grad_{patient_id}")
        
    else:
        first_name = patient_row["full_name"].split()[0]
        st.markdown("<div class='ai-label'>Select intervention strategy (ranked by effectiveness)</div>", unsafe_allow_html=True)
        intervention = st.radio("Choose intervention", ["📹 Suggest Telehealth Switch (Top Recommendation)", "🎬 Send Psychoeducation Video", "💬 Draft Warm Check-In (Rule-Based)"], index=0, key=f"intervention_{patient_id}", label_visibility="collapsed")
        
        draft_key = f"draft_{patient_id}"
        if intervention.startswith("📹"):
            if patient_row["no_shows"] > 0:
                template_draft = f"Hi {first_name}, I noticed you missed your last session. Would it be easier to switch to a quick video call this week? I can send you a one-click link — no travel needed. Let me know what works best for you!"
            else:
                template_draft = f"Hi {first_name}, I want to make sure our sessions stay convenient for you. Would it be easier to do a quick video call for our next meeting? I can send you a one-click link — no travel needed. Let me know!"
                
            st.markdown("<div style='font-size:0.78rem;color:#52B788;margin:4px 0 8px'>💡 <strong>Why this is the top suggestion:</strong> Removes logistical friction (travel, scheduling) and lowers social pressure — the #1 reason high-risk patients avoid sessions.</div>", unsafe_allow_html=True)
            if draft_key not in st.session_state or st.session_state.get(f"last_intervention_{patient_id}") != intervention:
                st.session_state[draft_key] = template_draft
                st.session_state[f"last_intervention_{patient_id}"] = intervention
        elif intervention.startswith("🎬"):
            template_draft = f"Hi {first_name}, it's completely normal to feel stuck at this stage of therapy. Here's a 2-minute video that explains why progress sometimes feels invisible — it's called 'the therapy plateau' and almost everyone goes through it. I think it'll really help. How are you feeling this week?"
            st.markdown("<div style='font-size:0.78rem;color:#52B788;margin:4px 0 8px'>💡 <strong>Why this works:</strong> Normalizes the patient's distress so they don't think treatment is 'failing'. Reduces shame-based avoidance.</div>", unsafe_allow_html=True)
            if draft_key not in st.session_state or st.session_state.get(f"last_intervention_{patient_id}") != intervention:
                st.session_state[draft_key] = template_draft
                st.session_state[f"last_intervention_{patient_id}"] = intervention
        elif intervention.startswith("💬"):
            if st.button("✨ Generate Rule-Based Draft", key=f"gen_msg_{patient_id}", type="primary", use_container_width=True):
                with st.spinner("AI is drafting..."):
                    draft = draft_intervention_message(patient_row["full_name"], " | ".join(patient_row["reasons"]), str(patient_row.get("latest_feedback", "")), patient_row["therapist"], int(patient_row["session_number"]), patient_row["srs_overall"], int(patient_row.get("no_shows", 0)))
                    st.session_state[draft_key] = draft
                    st.session_state[f"last_intervention_{patient_id}"] = intervention
            st.markdown("<div style='font-size:0.78rem;color:#52B788;margin:4px 0 8px'>💡 <strong>Why this is ranked #3:</strong> Most personalized, but requires an extra human administrative step. Best when the other two options don't fit the situation.</div>", unsafe_allow_html=True)
            
        draft_text = st.session_state.get(draft_key, "")
        if draft_text:
            st.markdown("<div class='ai-label'>Review and edit before sending</div>", unsafe_allow_html=True)
            editable = st.text_area("Edit message", value=draft_text, height=110, key=f"edit_{patient_id}", label_visibility="collapsed")
            st.markdown("---")
            st.markdown("**📲 Send via Twilio WhatsApp Sandbox**")
            phone = str(patient_row.get("phone", ""))
            render_send_button(patient_row["full_name"], phone, editable, f"twilio_send_{patient_id}")

    st.markdown("</div>", unsafe_allow_html=True)

