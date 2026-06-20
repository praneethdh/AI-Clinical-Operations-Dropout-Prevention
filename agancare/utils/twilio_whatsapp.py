"""
utils/twilio_whatsapp.py

Twilio Sandbox for WhatsApp — FREE for prototyping.
No paid Meta Business API needed.

Setup (one-time, free):
1. Sign up at https://www.twilio.com (free account)
2. Go to Messaging → Try it Out → Send a WhatsApp message
3. Note your Sandbox number: +14155238886
4. Each recipient must join the sandbox first by sending:
   "join <your-sandbox-word>" to +14155238886 on WhatsApp
5. Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SANDBOX_NUMBER to .streamlit/secrets.toml

How it works in the demo:
- Therapist types/edits message in dashboard
- Clicks "Send via WhatsApp"  
- Patient's phone buzzes with a REAL WhatsApp message
- Patient replies → reply shows in Twilio console (and can be routed back)
"""
import streamlit as st


def get_twilio_credentials() -> dict:
    """
    Read Twilio creds from Streamlit secrets.
    Returns dict with keys: account_sid, auth_token, sandbox_number
    Returns None values if not configured (shows setup instructions instead).
    """
    try:
        tw = st.secrets.get("twilio", {})
        return {
            "account_sid": tw.get("TWILIO_ACCOUNT_SID", st.secrets.get("TWILIO_ACCOUNT_SID", "")),
            "auth_token": tw.get("TWILIO_AUTH_TOKEN", st.secrets.get("TWILIO_AUTH_TOKEN", "")),
            "sandbox_number": tw.get("TWILIO_SANDBOX_NUMBER", st.secrets.get("TWILIO_SANDBOX_NUMBER", "whatsapp:+14155238886")),
        }
    except Exception:
        return {"account_sid": "", "auth_token": "", "sandbox_number": "whatsapp:+14155238886"}


def send_whatsapp_message(to_phone: str, message_body: str) -> dict:
    """
    Send a real WhatsApp message via Twilio Sandbox.
    
    Args:
        to_phone: Patient phone number in E.164 format e.g. "+919876543210"
        message_body: The message text to send
    
    Returns:
        dict with keys: success (bool), message_sid (str), error (str)
    """
    creds = get_twilio_credentials()

    if not creds["account_sid"] or not creds["auth_token"]:
        return {
            "success": False,
            "message_sid": "",
            "error": "TWILIO_NOT_CONFIGURED",
        }

    try:
        from twilio.rest import Client
        client = Client(creds["account_sid"], creds["auth_token"])

        # Ensure phone number is E.164 formatted (must start with +)
        clean_phone = to_phone.strip()
        if not clean_phone.startswith("+"):
            clean_phone = f"+{clean_phone}"
            
        # Format the recipient number for WhatsApp
        to_whatsapp = f"whatsapp:{clean_phone}"
        
        from_num = creds["sandbox_number"]
        if not from_num.startswith("whatsapp:"):
            from_num = f"whatsapp:{from_num}"
        from_whatsapp = from_num

        msg = client.messages.create(
            body=message_body,
            from_=from_whatsapp,
            to=to_whatsapp,
        )
        return {
            "success": True,
            "message_sid": msg.sid,
            "error": "",
        }
    except Exception as e:
        return {
            "success": False,
            "message_sid": "",
            "error": str(e),
        }


def render_send_button(patient_name: str, phone: str, message_body: str, button_key: str):
    """
    Renders the full Twilio WhatsApp send UI inside the dropout hub action card.
    Shows:
    - Real Send button (if Twilio configured)
    - Setup instructions (if not configured)
    - Success/error feedback
    """
    creds = get_twilio_credentials()
    first_name = patient_name.split()[0]

    if creds["account_sid"] and creds["auth_token"]:
        # ── Twilio IS configured — show real send button ──────────────────
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#8B949E'>📱 Confirm 10-Digit Recipient Number (+91)</div>",
                unsafe_allow_html=True,
            )
            # Extract just the 10 digits from whatever is in the CSV
            default_10_digit = phone.replace("+91", "").replace("+", "").strip()
            if len(default_10_digit) == 12 and default_10_digit.startswith("91"):
                default_10_digit = default_10_digit[2:]
                
            input_number = st.text_input(
                "Phone",
                value=default_10_digit,
                max_chars=10,
                key=f"phone_input_{button_key}",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
            if st.button(
                "📲 Send WhatsApp",
                key=button_key,
                type="primary",
                use_container_width=True,
            ):
                if len(input_number) != 10 or not input_number.isdigit():
                    st.error("❌ Please enter exactly 10 digits.")
                elif not message_body.strip():
                    st.warning("Message is empty — please generate or type a message first.")
                else:
                    final_phone = f"+91{input_number}"
                    with st.spinner(f"Sending WhatsApp to {first_name}..."):
                        result = send_whatsapp_message(final_phone, message_body)

                    if result["success"]:
                        st.success(
                            f"✅ WhatsApp sent to {first_name}! "
                            f"Message SID: `{result['message_sid']}`"
                        )
                    else:
                        st.error(f"❌ Send failed: {result['error']}")
                        st.info(
                            "💡 Make sure the patient has joined the Twilio Sandbox by sending "
                            f"`join <your-sandbox-word>` to +14155238886 on WhatsApp."
                        )
    else:
        # ── Twilio NOT configured — show setup instructions ────────────────
        with st.expander("📲 Set up WhatsApp Sending (Free Twilio Sandbox)", expanded=False):
            st.markdown("""
**One-time setup — completely free:**

**Step 1:** Sign up at [twilio.com](https://www.twilio.com) (free account, no credit card)

**Step 2:** Go to **Messaging → Try it Out → Send a WhatsApp message**

**Step 3:** Note your credentials:
- Account SID (starts with `AC...`)
- Auth Token
- Sandbox number: `+14155238886`

**Step 4:** Patient joins sandbox by sending this WhatsApp to `+14155238886`:
```
join <your-sandbox-word>
```

**Step 5:** Add to `.streamlit/secrets.toml`:
```toml
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN  = "your_auth_token"
TWILIO_SANDBOX_NUMBER = "whatsapp:+14155238886"
```

**Step 6:** Restart the app — the Send button will appear automatically.
""")
        st.info(
            "⚙️ Twilio not configured yet. Add credentials to `secrets.toml` to enable real WhatsApp sending. "
            "See setup instructions above."
        )
