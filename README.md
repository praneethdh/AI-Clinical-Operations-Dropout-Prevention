---
title: AganCare Sentinel
emoji: 🧠
colorFrom: green
colorTo: teal
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🧠 AganCare Sentinel: AI Clinical Operations Platform
**Streamlit • Scikit-Learn • SHAP • Twilio WhatsApp API**

An advanced AI-powered clinical operations platform designed to optimize therapist matching and proactively predict/prevent patient dropout in mental health clinics. 

---

## 🔗 Live Application Link
🚀 **Hugging Face Spaces Demo: [Launch Application Here]**

---

## 🚀 Architecture
This system is built as a highly interactive, state-driven Python web application:

* **Frontend & Backend:** Built entirely with **Streamlit**, leveraging advanced state management (`st.session_state`) for seamless multi-step clinical workflows and dynamic UI rendering.
* **Database / Data Layer:** Powered by **Pandas**, reading and appending dynamically to normalized CSV datasets containing patient intakes, session logs, and therapist profiles.
* **AI Models:**
  * **TF-IDF & Cosine Similarity (NLP):** Drives the semantic Intake Matchmaker to perfectly align patient goals with a therapist's clinical approach.
  * **Random Forest Classifier (Machine Learning):** Analyzes historical Session Rating Scale (SRS) scores, missed sessions, and demographics to calculate real-time patient dropout probabilities.
  * **SHAP (Explainable AI):** Unpacks the Random Forest and NLP algorithms to provide transparent, human-readable reasoning for all AI recommendations.

---

## 🌟 Key Features

* **🧩 Intake Matchmaker:** A dynamic 2-step pipeline that calculates baseline "flight risk", processes semantic text matching, and recommends the Top 5 therapists complete with SHAP-based feature reasoning. Includes a *Risk Override* feature that prioritizes high-retention therapists for high-risk patients.
* **🛡️ Dropout Prevention Hub:** An analytical command center that monitors active patients. If a patient's ML risk score crosses a critical threshold, they are flagged for immediate intervention.
* **💬 AI Intervention & WhatsApp Integration:** Generates clinical intervention strategies (e.g., suggesting a Telehealth switch to reduce friction) and sends the drafted messages directly to the patient's phone via the **Twilio WhatsApp Sandbox API**.
* **📊 Sentinel Dashboard:** A real-time executive overview of clinic health, tracking total active patients, global dropout rates, and AI model performance metrics.
* **Modern UI:** Features dark-mode aesthetics, custom glassmorphism metric cards, and fluid interactive expanders to make clinical data easily digestible.

---

## 🛠️ Getting Started

### Prerequisites
* Python 3.12+
* Twilio Account (Free Sandbox)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/agancare-sentinel.git
cd agancare-sentinel

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.streamlit/secrets.toml` file in the root directory and add your Twilio Sandbox credentials:
```toml
[twilio]
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_SANDBOX_NUMBER = "whatsapp:+14155238886"
```

### 3. Run the Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`.

---

## 📂 Project Structure
```text
├── dataset/              # Core CSV database (patients, sessions, therapists)
├── pages/                # Streamlit Application Pages
│   ├── 1_matchmaker.py   # Intake and Therapist Matching UI
│   ├── 2_dropout_hub.py  # ML Dropout Prediction and Intervention UI
│   └── 3_session_logger.py # Form to log new patient sessions
├── utils/                # Backend logic and helpers
│   ├── data_loader.py    # Caching, feature engineering, and RF model training
│   ├── matchmaker.py     # NLP similarity and Baseline Risk math
│   ├── navigation.py     # Global sidebar routing
│   └── twilio_whatsapp.py # Twilio API integration
├── .streamlit/           # Theme and secrets configuration
├── app.py                # Main Dashboard Entrypoint
├── requirements.txt      # Project dependencies
└── README.md             # You are here
```

---

## 📈 System Objectives
* **Proactive Visibility:** Move from reactive crisis-management to predictive clinical care.
* **Risk Mitigation:** Identify high-risk patients *before* they miss a session using ML-driven SRS trend analysis.
* **Optimized Empathy:** Assist clinical directors in maintaining the therapeutic alliance through automated, warm, and timely WhatsApp check-ins.

---

## 📄 License
Distributed under the MIT License.
