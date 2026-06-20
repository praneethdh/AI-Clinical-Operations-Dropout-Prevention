import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <style>
          .brand-block {
            background: linear-gradient(135deg, #1C2A22 0%, #0D1612 100%);
            padding: 1.5rem; border-radius: 12px; border: 1px solid #2D6A4F;
            margin-bottom: 2rem; text-align: center;
          }
          .brand-title { font-size: 1.2rem; font-weight: 800; color: #52B788; letter-spacing: 0.5px; }
          .brand-sub { font-size: 0.75rem; color: #8B949E; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
          /* Hide Streamlit's default ugly file-based navigation */
          [data-testid="stSidebarNav"] {display: none !important;}
        </style>
        <div class='brand-block'>
          <div class='brand-title'>🧠 AganCare Sentinel</div>
          <div class='brand-sub'>Clinical Operations AI · Prototype v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Navigate**")
        st.page_link("app.py", label="Dashboard", icon="📊")
        st.page_link("pages/1_matchmaker.py", label="Intake Matchmaker", icon="🧩")
        st.page_link("pages/3_session_logger.py", label="Log a Session", icon="📝")
        st.page_link("pages/2_dropout_hub.py", label="Dropout Prevention Hub", icon="🛡️")

        st.divider()
        st.markdown("<div style='font-size:0.75rem;color:#8B949E'>Built for AganCare · Prototype · Not for clinical use</div>",
                    unsafe_allow_html=True)
