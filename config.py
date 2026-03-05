import streamlit as st

# ─── SETTRADE CREDENTIALS ─────────────────────────────────────────────────────
APP_ID     = st.secrets.get("APP_ID",     "VMxlV5Hz3BvMkitL")
APP_SECRET = st.secrets.get("APP_SECRET", "OecHOLQUlbnHevImrX68VPzEOCxqKaBbuatzq88LOmg=")
BROKER_ID  = st.secrets.get("BROKER_ID",  "023")
APP_CODE   = st.secrets.get("APP_CODE",   "ALGO_EQ")

# ─── AI API ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")  # ← ใส่ใน .streamlit/secrets.toml
