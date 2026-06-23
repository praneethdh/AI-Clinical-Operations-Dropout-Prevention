# AganCare Sentinel: AI-Powered Clinical Operations & Patient Retention Platform

## About The Project

AganCare Sentinel is an advanced AI-powered clinical operations platform developed to improve patient retention and optimize therapist allocation within mental health clinics. Patient dropout remains one of the biggest challenges in behavioral healthcare, often resulting in interrupted treatment outcomes and reduced clinic efficiency. This project addresses that challenge by combining machine learning, natural language processing, explainable AI, and automated patient engagement workflows into a single intelligent platform.

The system uses a multi-model AI architecture that analyzes patient intake information, therapy goals, session attendance patterns, and Session Rating Scale (SRS) scores to identify patients who may be at risk of discontinuing treatment. A TF-IDF and Cosine Similarity based recommendation engine performs semantic matching between patient needs and therapist expertise, ensuring that patients are paired with clinicians best suited to their goals. A Random Forest Classifier continuously evaluates patient engagement data and predicts dropout probabilities in real time. To improve trust and transparency, SHAP (SHapley Additive Explanations) is integrated to provide human-readable explanations behind every AI recommendation and risk prediction.

The platform also incorporates automated intervention workflows through the Twilio WhatsApp API, enabling clinics to proactively engage high-risk patients with personalized outreach messages. Built entirely using Streamlit, Pandas, Scikit-Learn, and SHAP, the application provides a modern dashboard-driven experience that helps clinics transition from reactive patient management to proactive, data-driven clinical care.

## 🔗 Live Space Deployment

**Live Web Application:**
https://huggingface.co/spaces/praneeth-dh/agancare-sentinel

## Library Requirements

### Core Libraries

* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* Twilio
* Scipy

### NLP Libraries

* Scikit-Learn TF-IDF Vectorizer
* Cosine Similarity

### Visualization Libraries

* Plotly
* Matplotlib

## Getting Started

Follow these instructions to set up and run the project locally.

## Installation Steps

### Option 1: Installation from GitHub

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/agancare-sentinel.git
```

#### 2. Navigate to Project Directory

```bash
cd agancare-sentinel
```

#### 3. Create a Virtual Environment

```bash
python -m venv venv
```

#### 4. Activate Virtual Environment

**Windows**

```powershell
.\venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

#### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 6. Configure Twilio Credentials

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
[twilio]
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_SANDBOX_NUMBER = "whatsapp:+14155238886"
```

#### 7. Run the Application

```bash
streamlit run app.py
```

#### 8. Access the Application

Open:

```text
http://localhost:8501
```

## 💻 How to Use

### Step 1: Open Dashboard

Launch the application and access the Sentinel Dashboard for a real-time overview of clinic performance and patient retention metrics.

### Step 2: Use Intake Matchmaker

Enter patient intake details, therapy goals, and preferences. The AI engine evaluates clinical requirements and recommends the most suitable therapists.

### Step 3: Review Therapist Recommendations

Analyze therapist rankings, compatibility scores, and SHAP-based explanations showing why each therapist was recommended.

### Step 4: Monitor Patient Risk

Navigate to the Dropout Prevention Hub to monitor active patients and their predicted dropout probabilities.

### Step 5: Analyze AI Explanations

Review key risk factors contributing to patient disengagement through Explainable AI visualizations.

### Step 6: Generate Interventions

Allow the platform to generate personalized intervention strategies based on identified risk drivers.

### Step 7: Send WhatsApp Outreach

Use the integrated Twilio WhatsApp Sandbox API to send supportive follow-up messages directly to patients.

## Key Features

### 🧩 Intake Matchmaker

* Semantic therapist-patient matching
* TF-IDF and Cosine Similarity based recommendations
* Top 5 therapist ranking system
* Risk-aware matching override mechanism
* Explainable recommendation reasoning

### 🛡️ Dropout Prevention Hub

* Real-time patient monitoring
* Random Forest dropout prediction
* Automated risk categorization
* Early intervention alerts

### 💬 Automated Patient Engagement

* Twilio WhatsApp integration
* AI-generated intervention messages
* Personalized patient outreach
* Reduced communication delays

### 📊 Sentinel Dashboard

* Active patient tracking
* Dropout rate monitoring
* Model performance analytics
* Clinical operations overview

### 🔍 Explainable AI

* SHAP visualizations
* Transparent risk predictions
* Human-readable recommendation insights

## API Key Setup

This project requires Twilio credentials for WhatsApp messaging.

### Create a Twilio Account

Visit the official Twilio website and create a free Sandbox account.

### Configure Secrets

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
[twilio]
TWILIO_ACCOUNT_SID="your_account_sid"
TWILIO_AUTH_TOKEN="your_auth_token"
TWILIO_SANDBOX_NUMBER="whatsapp:+14155238886"
```

**Important:** Never commit secrets or API credentials to GitHub.

## Project Structure

```text
├── dataset/
│   ├── patients.csv
│   ├── sessions.csv
│   └── therapists.csv
│
├── pages/
│   ├── 1_matchmaker.py
│   ├── 2_dropout_hub.py
│   └── 3_session_logger.py
│
├── utils/
│   ├── data_loader.py
│   ├── matchmaker.py
│   ├── navigation.py
│   └── twilio_whatsapp.py
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
├── app.py
├── requirements.txt
└── README.md
```

## System Objectives

* Improve patient retention outcomes
* Reduce treatment dropout rates
* Enhance therapist-patient alignment
* Enable proactive clinical interventions
* Increase transparency through Explainable AI
* Support data-driven clinical decision-making

## Contributing

Contributions are welcome and greatly appreciated.

### Report Bugs

Open an issue describing the problem and reproduction steps.

### Contribute Code

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes

```bash
git commit -m "Add Amazing Feature"
```

4. Push to the branch

```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request

### Suggestions

Feature requests and enhancement ideas are always welcome.

If you find this project useful, consider giving it a ⭐ on GitHub.

## License

This project is licensed under the MIT License.


## Acknowledgements

Special thanks to the open-source community and the technologies that made this project possible:

* Streamlit
* Scikit-Learn
* SHAP
* Twilio
* Pandas
* NumPy
* Plotly
* Python
* Hugging Face Spaces
