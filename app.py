"""
app.py — EDA Agent powered by Claude (Anthropic)
--------------------------------------------------
Upload a dataset and get a full AI-powered analysis instantly.
Uses Claude for narrative insights + Pandas/Plotly for stats & charts.

Run:  python3 -m streamlit run app.py
"""

import os
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DAILY_LIMIT   = int(os.getenv("DAILY_LIMIT", "10"))
MAX_FILE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
APP_TITLE     = os.getenv("APP_TITLE", "🔍 EDA Agent — Powered by Claude")
USE_AI        = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

# ── Lazy imports (choose engine based on API key availability) ────────────────
if USE_AI:
    from eda_agent import EDAAgent
else:
    from eda_engine import EDAEngine

from rate_limiter import RateLimiter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "result"     not in st.session_state:
    st.session_state.result     = None
if "filename"   not in st.session_state:
    st.session_state.filename   = ""

session_id   = st.session_state.session_id
rate_limiter = RateLimiter()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 EDA Agent")
    if USE_AI:
        st.markdown("**AI-powered** analysis via **Claude** (Anthropic).")
        remaining = rate_limiter.get_remaining(session_id, DAILY_LIMIT)
        st.metric("Analyses remaining today", remaining)
    else:
        st.markdown("**Free mode** — no API key detected. Using rule-based engine.")
        st.info("Set `ANTHROPIC_API_KEY` in `.env` to enable Claude AI insights.")

    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("CSV · Excel (.xlsx / .xls) · TSV · JSON")
    st.markdown("---")
    st.markdown("**What it analyses**")
    st.markdown(
        "- 📋 Dataset overview\n"
        "- 🧹 Missing values & duplicates\n"
        "- 📊 Descriptive statistics\n"
        "- 📈 Distributions & skewness\n"
        "- ⚠️ Outlier detection (IQR)\n"
        "- 🔗 Correlation analysis\n"
        "- 🏷️ Categorical breakdowns\n"
        "- 💡 Smart recommendations"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
if USE_AI:
    st.markdown(
        "Upload your dataset and get a **full AI-powered EDA report** — "
        "statistics computed by Pandas, narrative written by **Claude**."
    )
    st.info("🤖 AI mode active — Claude will write the analysis narrative.", icon="✨")
else:
    st.markdown(
        "Upload your dataset and get a **full data analysis report** — "
        "completely free, no account or API key needed."
    )
    st.info("⚡ Free mode — powered by Python + Pandas. Add `ANTHROPIC_API_KEY` for AI insights.", icon="🆓")

# ── Upload ────────────────────────────────────────────────────────────────────
col_upload, col_question = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset",
        type=["csv", "xlsx", "xls", "tsv", "json"],
        help=f"Max file size: {MAX_FILE_MB} MB",
    )

with col_question:
    st.markdown("**Ask a specific question** *(optional)*")
    user_question = st.text_area(
        "question",
        placeholder=(
            "e.g. Which columns have the most missing data?\n"
            "or: Are there strong correlations between columns?"
        ),
        height=130,
        label_visibility="collapsed",
    )

# ── Load & preview ────────────────────────────────────────────────────────────
df = None
if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"❌ File is {size_mb:.1f} MB — max allowed is {MAX_FILE_MB} MB.")
        st.stop()

    try:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(uploaded_file)
        elif ext == "tsv":
            df = pd.read_csv(uploaded_file, sep="\t")
        elif ext == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported format.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    st.success(
        f"✅ **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",             f"{df.shape[0]:,}")
    m2.metric("Columns",          df.shape[1])
    m3.metric("Missing values",   f"{df.isnull().sum().sum():,}")
    m4.metric("Numeric columns",  len(df.select_dtypes(include="number").columns))

    with st.expander("👀 Preview first 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Rate-limit check (AI mode only)
    if USE_AI and rate_limiter.get_remaining(session_id, DAILY_LIMIT) <= 0:
        st.warning(f"⚠️ Daily limit of {DAILY_LIMIT} analyses reached for this session.")
        st.stop()

    btn_label = "🚀 Run AI-Powered EDA!" if USE_AI else "🚀 Run EDA Analysis — Free!"
    st.markdown("")
    if st.button(btn_label, type="primary", use_container_width=True):
        # Consume rate-limit slot (AI mode)
        if USE_AI and not rate_limiter.check_and_increment(session_id, DAILY_LIMIT):
            st.warning("Daily limit reached. Try again tomorrow.")
            st.stop()

        st.session_state.result   = None
        st.session_state.filename = uploaded_file.name
        msgs = []

        with st.status("⚙️ Analysing your data...", expanded=True) as status_box:
            if USE_AI:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                engine  = EDAAgent(api_key=api_key)
            else:
                engine = EDAEngine()

            result = engine.analyze(
                df,
                user_question=user_question,
                filename=uploaded_file.name,
                progress_callback=lambda m: msgs.append(m),
            )
            for m in msgs:
                st.write(m)
            status_box.update(label="✅ Done!", state="complete", expanded=False)

        st.session_state.result = result
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result   = st.session_state.result
    filename = st.session_state.filename

    st.markdown("---")
    st.header(f"📊 EDA Report — {filename}")

    with st.expander("📄 Full Analysis Report", expanded=True):
        st.markdown(result["report"])

    charts = result.get("charts", [])
    if charts:
        st.markdown("---")
        st.header("📈 Visualisations")
        for i in range(0, len(charts), 2):
            cols = st.columns(2, gap="medium")
            for j, col_widget in enumerate(cols):
                if i + j < len(charts):
                    with col_widget:
                        st.subheader(charts[i + j]["title"])
                        st.plotly_chart(charts[i + j]["fig"], use_container_width=True)

    st.markdown("---")
    col_r, col_d = st.columns(2)
    with col_r:
        if st.button("🔄 Analyse another dataset", use_container_width=True):
            st.session_state.result = None
            st.rerun()
    with col_d:
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name=f"eda_{filename.rsplit('.', 1)[0]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.markdown("---")
if USE_AI:
    st.caption("AI-powered · Claude by Anthropic · Built with Pandas + Plotly + Streamlit")
else:
    st.caption("100% free · No API · Built with Python + Pandas + Plotly + Streamlit")
