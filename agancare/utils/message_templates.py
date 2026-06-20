"""
utils/message_templates.py

Rule-based message templates and risk explanations to replace the Anthropic AI integration.
"""

def draft_intervention_message(
    patient_name: str,
    risk_reasons: str,
    latest_feedback: str,
    therapist_name: str,
    session_number: int,
    srs_overall: float,
    no_shows: int = 0
) -> str:
    """
    Draft a warm, short WhatsApp check-in message based on predefined rules.
    """
    first_name = patient_name.split()[0]
    
    if no_shows >= 1:
        return f"Hi {first_name}, we noticed you missed your last session. We want to support you through this. Would it be easier to switch to a quick video call, or is there another time that works better for you?"
        
    elif "Goals score dropped" in risk_reasons:
        return f"Hi {first_name}, I was reviewing your recent feedback and want to make sure we are focusing on the right goals for you. Would you be open to adjusting our approach in our next session?"
        
    else:
        return f"Hi {first_name}, just checking in to see how you are doing before our next session. Your progress is important to us, and we want to make sure you're feeling supported."


def explain_risk_reasoning(
    patient_name: str,
    risk_reasons: str,
    srs_history: list,
    latest_feedback: str,
) -> str:
    """
    Plain-English clinical explanation using a rule-based template.
    """
    srs_str = " -> ".join([str(s) for s in srs_history[-3:]]) if len(srs_history) > 0 else "None"
    
    explanation = (
        f"**Risk Analysis for {patient_name}**\n\n"
        f"**Recent SRS Trend:** {srs_str}\n"
        f"The patient's recent scores suggest a decline in engagement or satisfaction. "
    )
    
    if "Goals score dropped" in risk_reasons:
        explanation += "A drop in the Goals score often indicates the patient feels therapy isn't addressing their current needs. "
    if "no-show" in risk_reasons.lower():
        explanation += "Missed sessions are a strong behavioral indicator of potential dropout. "
    if "SRS dropped" in risk_reasons:
        explanation += "A sharp decline in overall SRS points to an immediate issue with the therapeutic alliance. "
        
    explanation += (
        "\n\n**Recommendation:** "
        "Review the recent feedback and consider directly addressing the alliance or goals at the start of the next session. "
        "Offering a low-friction telehealth alternative may also improve attendance."
    )
    
    return explanation
