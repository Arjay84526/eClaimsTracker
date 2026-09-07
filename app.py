import io
import os
from datetime import datetime, timedelta
import msoffcrypto
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="eClaimsTracker | IMAP Lying-In Clinic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Clean Theme Overrides */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Card Container */
    .clay-card {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        padding: 14px 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .clay-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Streamlit Button Styling */
    div.stButton > button {
        background: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1rem;
        font-weight: 600;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: background 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #1d4ed8;
    }
    
    /* Auto-fit the first column (No.) */
    div[data-testid="stDataFrame"] table tr th:first-child,
    div[data-testid="stDataFrame"] table tr td:first-child {
        width: 1% !important;
        white-space: nowrap !important;
        text-align: center !important;
    }

    /* Right-Alignment Helper */
    .right-aligned-column {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        text-align: right;
    }
    .right-aligned-column h3 {
        text-align: right;
        width: 100%;
    }
    .right-aligned-column .stTextInput {
        text-align: right;
        width: 100%;
    }
    .right-aligned-column div.stTextInput input {
        text-align: right;
    }
    .right-aligned-column div.stButton > button,
    .right-aligned-column div[data-testid="stDownloadButton"] button {
        margin-left: auto;
        margin-right: 0;
    }

    /* Force side-by-side expander containers to match height */
    [data-testid="stExpander"] {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stExpander"] > div:last-child {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    [data-testid="stExpanderDetails"] {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- APP TITLE ---
st.markdown(
    """
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #0f172a; margin-bottom: 2px; letter-spacing: -0.025em;">PhilHealth eClaims Manager</h1>
        <p style="font-size: 0.95rem; color: #475569;">IMAP Lying-In Clinic Inc. - Bilar Branch | eClaims Tracker & RTH Resolution Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 2. ICD / RVS CODE DICTIONARY & DISCREPANCY PARSER ---
ICD_RVS_MAPPING = {
    "NSDBF": "Normal Spontaneous Delivery (NSD)",
    "NSD1L": "Normal Spontaneous Delivery (NSD)",
    "FP001": "Family Planning (Subdermal Implants)",
    "58300": "Family Planning (IUD Insertion)",
    "99460": "Newborn Care Package (NCP)",
    "MCP01": "Maternity Care Package (MCP)",
    "PPC01": "Postpartum Care Package",
    "ANV01": "Antenatal Care Package (Trimester 1)",
    "ANV02": "Antenatal Care Package (Trimester 2)",
    "ANV03": "Antenatal Care Package (Trimester 3)",
    "AND03": "Antenatal / Delivery Care",
    "ANS02": "Antenatal Care Services",
}

def translate_icd_rvs(code):
    if pd.isna(code) or not str(code).strip():
        return "Uncategorized Package"
    clean_code = str(code).strip().upper()
    return ICD_RVS_MAPPING.get(clean_code, f"Code: {clean_code}")

def categorize_discrepancy(disc_text):
    if pd.isna(disc_text) or not str(disc_text).strip():
        return "❓ Pending Details from PhilHealth"
    
    text = str(disc_text).upper()
    tags = []
    
    if "FORM 2" in text or "CLAIM FORM 2" in text or "ACCOMPLISHED" in text:
        tags.append("📝 Claim Form Issue")
    if "ANTENATAL" in text or "RECORD" in text or "CHART" in text:
        tags.append("📄 Antenatal / Clinical Chart")
    if "JUSTIFICATION" in text or "LETTER" in text:
        tags.append("✉️ Missing Justification")
    if "OTHER DOCUMENTS" in text or "REQUIRED CLAIM FORM" in text:
        tags.append("📎 Missing Attachment / Document")
        
    return " | ".join(tags) if tags else "📋 General Discrepancy Notice"

# Working days calculation helper (Excludes Weekends)
def calc_working_days(start_date, end_date):
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan
    try:
        s = pd.Timestamp(start_date).date()
        e = pd.Timestamp(end_date).date()
        if s > e:
            return 0
        return len(pd.bdate_range(start=s, end=e)) - 1
    except Exception:
        return np.nan

def eval_bank_withdrawal_status(row):
    """
    Evaluates bank withdrawal readiness for claims.
    Includes both 'With Voucher' and 'With Cheque' statuses.
    - 5 to 12 working days: Ready for Bank Run (Thursday/Friday)
    - > 12 working days: Already Collected / Withdrawn
    - < 5 working days: Too Early
    """
    status = str(row.get("STATUS", "")).strip().title()
    if status not in ["With Voucher", "With Cheque"]:
        return "N/A"
    
    wd_thu = row.get("Working Days Thursday")
    wd_fri = row.get("Working Days Friday")
    
    if pd.isna(wd_thu) or pd.isna(wd_fri):
        return "N/A"
    
    if 5 <= wd_thu <= 12:
        return "🏦 Ready by Thursday"
    elif 5 <= wd_fri <= 12:
        return "🏦 Ready by Friday"
    elif wd_thu > 12:
        return "✅ Collected / Withdrawn"
    else:
        return "⏳ Too Early (<5 Days)"

def compute_bank_readiness_columns(df):
    """Calculates Thursday / Friday Working Day Age and Bank Withdrawal Status based on a 5-12 day window."""
    today = pd.Timestamp(datetime.today().date())
    days_to_thu = (3 - today.weekday()) % 7
    days_to_fri = (4 - today.weekday()) % 7
    next_thu = today + pd.Timedelta(days=days_to_thu)
    next_fri = today + pd.Timedelta(days=days_to_fri)

    if "VOUCHER DATE" in df.columns:
        df["Working Days Today"] = df["VOUCHER DATE"].apply(lambda v: calc_working_days(v, today))
        df["Working Days Thursday"] = df["VOUCHER DATE"].apply(lambda v: calc_working_days(v, next_thu))
        df["Working Days Friday"] = df["VOUCHER DATE"].apply(lambda v: calc_working_days(v, next_fri))
    else:
        df["Working Days Today"] = np.nan
        df["Working Days Thursday"] = np.nan
        df["Working Days Friday"] = np.nan

    df["Bank Withdrawal Readiness"] = df.apply(eval_bank_withdrawal_status, axis=1)
    return df


# --- 3. RESILIENT DATA INGESTION & OPTIMIZED PIPELINE ---
@st.cache_data(show_spinner=False)
def load_and_process_data_cached(file_bytes, filename):
    try:
        file_buffer = io.BytesIO(file_bytes)
        if filename.endswith(".csv"):
            preview = pd.read_csv(file_buffer, nrows=20, header=None)
        else:
            preview = pd.read_excel(file_buffer, nrows=20, header=None)

        header_row = 0
        for idx, row in preview.iterrows():
            row_str = row.astype(str).str.cat(sep=" ").upper()
            if ("PATIENT" in row_str or "NAME" in row_str) and (
                "TRANSMITTAL" in row_str or "SERIES" in row_str or "PIN" in row_str
            ):
                header_row = idx
                break

        file_buffer.seek(0)
        if filename.endswith(".csv"):
            df = pd.read_csv(file_buffer, header=header_row)
        else:
            df = pd.read_excel(file_buffer, header=header_row)

        df.columns = df.columns.astype(str).str.strip()

        col_mapping = {}
        for col in df.columns:
            u_col = col.upper()
            if "SERIES" in u_col or "LHIO" in u_col:
                col_mapping[col] = "CLAIM SERIES LHIO"
            elif "PATIENT" in u_col and "PIN" in u_col:
                col_mapping[col] = "PATIENT PIN"
            elif "MEMBER" in u_col and "PIN" in u_col:
                col_mapping[col] = "MEMBER PIN"
            elif "PATIENT" in u_col and "NAME" in u_col:
                if "PATIENT NAME" not in col_mapping.values():
                    col_mapping[col] = "PATIENT NAME"
            elif "DISCHARGE" in u_col and "DATE" in u_col:
                col_mapping[col] = "DISCHARGE DATE"
            elif "TRANSMITTAL" in u_col:
                if "TRANSMITTAL ID" not in col_mapping.values():
                    col_mapping[col] = "TRANSMITTAL ID"
            elif u_col == "CLAIM STATUS" or u_col == "STATUS":
                col_mapping[col] = "STATUS"
            elif "ICD/RVS CODE" in u_col or "ICD" in u_col or "RVS" in u_col:
                if "RAW_ICD_RVS" not in col_mapping.values():
                    col_mapping[col] = "RAW_ICD_RVS"
            elif "TRANSMITTED" in u_col and ("DATE" in u_col or "ON" in u_col):
                if "TRANSMITTED ON" not in col_mapping.values():
                    col_mapping[col] = "TRANSMITTED ON"
            elif "VOUCHER" in u_col and "DATE" in u_col:
                col_mapping[col] = "VOUCHER DATE"
            elif "VOUCHER" in u_col and "NUMBER" in u_col:
                col_mapping[col] = "VOUCHER NUMBER"

        df = df.rename(columns=col_mapping)

        if "RAW_ICD_RVS" in df.columns:
            df["RAW_ICD_RVS"] = df["RAW_ICD_RVS"].astype(str).str.strip().str.upper().replace(["NAN", "NONE", "N/A", ""], "N/A")
            df["CLAIM TYPE"] = df["RAW_ICD_RVS"].apply(translate_icd_rvs)
        else:
            df["RAW_ICD_RVS"] = "N/A"
            df["CLAIM TYPE"] = "General Claim"

        if "STATUS" not in df.columns:
            df["STATUS"] = "In Process"
        if "TRANSMITTED ON" not in df.columns:
            df["TRANSMITTED ON"] = datetime.today().strftime("%m-%d-%Y")

        required_cols = [
            "CLAIM SERIES LHIO",
            "PATIENT PIN",
            "PATIENT NAME",
            "DISCHARGE DATE",
            "STATUS",
            "CLAIM TYPE",
            "TRANSMITTED ON",
        ]
        for rc in required_cols:
            if rc not in df.columns:
                if rc in ["DISCHARGE DATE", "TRANSMITTED ON"]:
                    df[rc] = datetime.today().strftime("%m-%d-%Y")
                else:
                    df[rc] = "N/A"

        for id_col in [
            "CLAIM SERIES LHIO",
            "PATIENT PIN",
            "MEMBER PIN",
            "TRANSMITTAL ID",
        ]:
            if id_col in df.columns:
                df[id_col] = (
                    df[id_col]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .replace("nan", "N/A")
                )

        today = pd.Timestamp(datetime.today().date())

        df["DISCHARGE DATE"] = pd.to_datetime(
            df["DISCHARGE DATE"], errors="coerce"
        ).fillna(today)

        df["TRANSMITTED ON"] = pd.to_datetime(
            df["TRANSMITTED ON"], errors="coerce"
        ).fillna(today)

        if "VOUCHER DATE" in df.columns:
            df["VOUCHER DATE"] = pd.to_datetime(df["VOUCHER DATE"], errors="coerce")
            df["Days Since Voucher Date"] = [
                int((today - vd).days) if pd.notna(vd) else "N/A"
                for vd in df["VOUCHER DATE"]
            ]
        else:
            df["VOUCHER DATE"] = pd.NaT
            df["Days Since Voucher Date"] = "N/A"

        df["Days Since Discharge"] = (today - df["DISCHARGE DATE"]).dt.days
        df["Days Until Deadline"] = 60 - df["Days Since Discharge"]
        df["Days Since Transmitted On"] = (today - df["TRANSMITTED ON"]).dt.days

        for amt_col in [
            "CLAIM AMOUNT",
            "1ST CASERATE AMOUNT",
            "RETURN DISCREPANCY",
            "DENIED REASON",
        ]:
            if amt_col in df.columns:
                if amt_col in ["CLAIM AMOUNT", "1ST CASERATE AMOUNT"]:
                    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

        status_upper = df["STATUS"].astype(str).str.strip().str.upper()
        claim_amt = pd.to_numeric(df.get("CLAIM AMOUNT", 0.0), errors="coerce").fillna(0.0)
        caserate_amt = pd.to_numeric(df.get("1ST CASERATE AMOUNT", 0.0), errors="coerce").fillna(0.0)
        
        return_disc_raw = df.get("RETURN DISCREPANCY", pd.Series([None] * len(df)))
        def check_disc(val):
            if pd.isna(val):
                return False
            s = str(val).strip().upper()
            if s in ["", "N/A", "NONE", "NULL", "NA", "NAN", "-", "INF", "-INF"]:
                return False
            return len(s.split()) > 0
        has_disc = return_disc_raw.apply(check_disc)

        actions = np.select(
            [
                (status_upper == "WITH CHEQUE") & (claim_amt > 0),
                (status_upper == "WITH VOUCHER"),
                (status_upper == "DENIED"),
                (status_upper == "IN PROCESS"),
                (status_upper == "RETURN") & has_disc,
                (status_upper == "RETURN")
            ],
            [
                "✅ Paid Claims",
                "🎟️ With Voucher",
                "❌ Denied",
                "⏳ In-Progress",
                "Attention RTH",
                "🚨 RTH Claims"
            ],
            default="📁 Other / Unassigned"
        )

        display_amt = np.where(claim_amt > 0, claim_amt, caserate_amt)

        df["Action Required"] = actions
        df["Display_Amount"] = display_amt
        df["Discrepancy Category"] = df.get("RETURN DISCREPANCY", pd.Series([None] * len(df))).apply(categorize_discrepancy)

        df = compute_bank_readiness_columns(df)

        search_cols_to_concat = [c for c in ["PATIENT NAME", "CLAIM SERIES LHIO", "PATIENT PIN", "TRANSMITTAL ID", "STATUS", "CLAIM TYPE", "RAW_ICD_RVS", "RETURN DISCREPANCY"] if c in df.columns]
        df["_SEARCH_INDEX"] = df[search_cols_to_concat].apply(lambda row: " ".join(row.fillna("").astype(str)), axis=1).str.upper()

        for col in ["STATUS", "CLAIM TYPE", "Action Required", "Bank Withdrawal Readiness"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    except Exception as e:
        st.error(f"Pipeline Error parsing file: {e}")
        return None


# --- 4. SESSION & PERSISTENT CACHE INITIALIZATION ---
CACHE_FILE = "philhealth_cached_claims.csv"

if "rth_notice_dates" not in st.session_state:
    st.session_state["rth_notice_dates"] = {}

if "df_data" not in st.session_state:
    if os.path.exists(CACHE_FILE):
        try:
            cached_df = pd.read_csv(
                CACHE_FILE, 
                parse_dates=["DISCHARGE DATE", "TRANSMITTED ON", "VOUCHER DATE"]
            )
            s_cols = [c for c in ["PATIENT NAME", "CLAIM SERIES LHIO", "PATIENT PIN", "TRANSMITTAL ID", "STATUS", "CLAIM TYPE", "RAW_ICD_RVS", "RETURN DISCREPANCY"] if c in cached_df.columns]
            cached_df["_SEARCH_INDEX"] = cached_df[s_cols].apply(lambda row: " ".join(row.fillna("").astype(str)), axis=1).str.upper()
            if "Discrepancy Category" not in cached_df.columns:
                cached_df["Discrepancy Category"] = cached_df.get("RETURN DISCREPANCY", pd.Series([None] * len(cached_df))).apply(categorize_discrepancy)
            if "RAW_ICD_RVS" not in cached_df.columns:
                cached_df["RAW_ICD_RVS"] = "N/A"
            
            if "Bank Withdrawal Readiness" not in cached_df.columns:
                cached_df = compute_bank_readiness_columns(cached_df)

            for col in ["STATUS", "CLAIM TYPE", "Action Required", "Bank Withdrawal Readiness"]:
                if col in cached_df.columns:
                    cached_df[col] = cached_df[col].astype("category")

            st.session_state["df_data"] = cached_df
        except Exception:
            st.session_state["df_data"] = None
    else:
        st.session_state["df_data"] = None

df = st.session_state["df_data"]

# --- 5. DATA INGESTION CENTER EXPANDER ---
with st.expander(
    "📂 Data Ingestion Center (Upload Bizbox Export)",
    expanded=(df is None or df.empty),
):
    uploaded_file = st.file_uploader(
        "Upload Bizbox Export (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"]
    )
    
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
            file_bytes = uploaded_file.getvalue()
            
            with st.status("🚀 Processing Bizbox claims export...", expanded=True) as status:
                st.write("📂 Reading file stream...")
                file_buffer = io.BytesIO(file_bytes)
                
                st.write("🔍 Detecting header row, parsing ICD/RVS codes & validating columns...")
                processed_df = load_and_process_data_cached(file_bytes, uploaded_file.name)
                
                st.write("⚡ Categorizing claims, computing Thursday/Friday Bank Run Forecast...")
                if processed_df is not None:
                    st.session_state["df_data"] = processed_df
                    processed_df.to_csv(CACHE_FILE, index=False)
                    df = processed_df
                    st.session_state["last_uploaded_filename"] = uploaded_file.name
                    st.session_state["show_upload_banner"] = True
                    status.update(label="✅ Data loaded, processed, and cached successfully!", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="❌ Failed to parse file.", state="error", expanded=True)

    if df is not None and not df.empty:
        if st.button("🗑️ Clear Saved Data Cache"):
            with st.spinner("Clearing local cache..."):
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                st.session_state["df_data"] = None
                st.session_state.pop("last_uploaded_filename", None)
                st.session_state.pop("show_upload_banner", None)
                st.success("Cache cleared successfully.")
                st.rerun()

# --- SUCCESS BANNER NOTIFICATION ---
if st.session_state.get("show_upload_banner", False):
    st.success("🎉 Bizbox Export File uploaded Successfully, the data is updated")

if df is not None and not df.empty:
    rth_df = df[df["STATUS"].astype(str).str.strip().str.title() == "Return"]
    attn_df = df[df["Action Required"] == "Attention RTH"]
    paid_df = df[df["Action Required"] == "✅ Paid Claims"]
    vouch_df = df[df["Action Required"] == "🎟️ With Voucher"]
    den_df = df[df["Action Required"] == "❌ Denied"]
    prog_df = df[df["Action Required"] == "⏳ In-Progress"]
    
    thu_ready_df = df[df["Bank Withdrawal Readiness"] == "🏦 Ready by Thursday"]
    fri_ready_df = df[df["Bank Withdrawal Readiness"] == "🏦 Ready by Friday"]
    collected_bank_df = df[df["Bank Withdrawal Readiness"] == "✅ Collected / Withdrawn"]
    bank_ready_combined = df[df["Bank Withdrawal Readiness"].isin(["🏦 Ready by Thursday", "🏦 Ready by Friday"])]

    total_records_count = len(df)

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">✅ Paid</span>
                <span style="background-color: #d1fae5; color: #065f46; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">{(len(paid_df)/max(1, total_records_count))*100:.0f}%</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #059669; margin-top: 4px; line-height: 1.1;">{len(paid_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{paid_df['Display_Amount'].sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">🎟️ Voucher</span>
                <span style="background-color: #e0f2fe; color: #0369a1; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">🏦 {len(bank_ready_combined)} Bank Ready</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #0284c7; margin-top: 4px; line-height: 1.1;">{len(vouch_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{vouch_df['Display_Amount'].sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">❌ Denied</span>
                <span style="background-color: #fee2e2; color: #991b1b; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">{(len(den_df)/max(1, total_records_count))*100:.0f}%</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #dc2626; margin-top: 4px; line-height: 1.1;">{len(den_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{den_df['Display_Amount'].sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">⏳ Process</span>
                <span style="background-color: #fef3c7; color: #92400e; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">{(len(prog_df)/max(1, total_records_count))*100:.0f}%</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #d97706; margin-top: 4px; line-height: 1.1;">{len(prog_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{prog_df['Display_Amount'].sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">🚨 RTH</span>
                <span style="background-color: #ffedd5; color: #9a3412; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">{(len(rth_df)/max(1, total_records_count))*100:.0f}%</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #ea580c; margin-top: 4px; line-height: 1.1;">{len(rth_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{pd.to_numeric(rth_df['Display_Amount'], errors='coerce').sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"""
        <div class="clay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">Attn RTH</span>
                <span style="background-color: #f3e8ff; color: #6b21a8; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">{(len(attn_df)/max(1, total_records_count))*100:.0f}%</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #9333ea; margin-top: 4px; line-height: 1.1;">{len(attn_df)}</div>
            <div style="font-size: 0.75rem; color: #1e293b; font-weight: 700; margin-top: 4px;">₱{pd.to_numeric(attn_df['Display_Amount'], errors='coerce').sum():,.2f}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 6. SIDE-BY-SIDE EXPANDERS: BANK PLANNER & ANALYTICS CHARTS ---
    b_col1, b_col2 = st.columns([1.1, 1])

    with b_col1:
        with st.expander("🏦 Thursday & Friday Bank Withdrawal Planner (5–7 Working Days)", expanded=not bank_ready_combined.empty):
            st.markdown("##### 💵 Weekly Bank Payout Forecast")
            st.markdown("Forecast of Claims hitting the optimal 5–7 working days threshold for Thursday & Friday bank runs.")
            st.caption("Note: the calculation is only based on the Previous Claims. There are dalays that are Due.")

            b1, b2, b3 = st.columns(3)
            with b1:
                thu_amt = thu_ready_df['Display_Amount'].sum()
                st.metric(
                    "🏦 Ready by Thursday", 
                    f"{len(thu_ready_df)} Claims", 
                    delta=f"₱{thu_amt:,.2f}"
                )
            with b2:
                fri_amt = fri_ready_df['Display_Amount'].sum()
                st.metric(
                    "🏦 Ready by Friday", 
                    f"{len(fri_ready_df)} Claims", 
                    delta=f"₱{fri_amt:,.2f}"
                )
            with b3:
                total_bank_amt = bank_ready_combined['Display_Amount'].sum()
                st.metric(
                    "💰 Possible Bank Run Target this Week",  
                    f"{len(bank_ready_combined)} Claims", 
                    delta=f"₱{total_bank_amt:,.2f}"
                )

            if not bank_ready_combined.empty:
                st.markdown("##### 📋 Bank Withdrawal Target Worklist")
                bank_display = bank_ready_combined[[
                    "TRANSMITTAL ID", "CLAIM SERIES LHIO", "PATIENT NAME", "CLAIM TYPE", 
                    "VOUCHER DATE", "Working Days Thursday", "Working Days Friday", "Bank Withdrawal Readiness", "Display_Amount"
                ]].copy()

                st.dataframe(
                    bank_display,
                    use_container_width=True,
                    hide_index=True,
                    height=280,
                    column_config={
                        "Display_Amount": st.column_config.NumberColumn("WITHDRAWABLE AMOUNT", format="₱%.2f"),
                        "VOUCHER DATE": st.column_config.DateColumn("VOUCHER DATE", format="MM-DD-YYYY"),
                        "Working Days Thursday": st.column_config.NumberColumn("WDAYS (THU)", format="%d Days"),
                        "Working Days Friday": st.column_config.NumberColumn("WDAYS (FRI)", format="%d Days"),
                    }
                )
                
                csv_bank = bank_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="🖨️ Download Thursday/Friday Bank Withdrawal Worklist (CSV)",
                    data=csv_bank,
                    file_name=f"Bank_Withdrawal_Worklist_{datetime.today().strftime('%m-%d-%Y')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No vouchered claims reaching 5–7 working days for this Thursday or Friday bank run.")

    with b_col2:
        with st.expander("📊 Analytics & Management Revenue Trend Charts", expanded=True):
            chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5, chart_tab6 = st.tabs([
                "📈 Revenue Timeline",
                "🩺 Clinical Breakdown",
                "🗓️ Day-of-Week",
                "📊 Claims Count Trend",
                "🏆 Highest-to-Lowest Payouts",
                "🏷️ ICD/RVS Code Count"
            ])
            
            status_color_map = {
                "✅ Paid Claims": "#059669",
                "🎟️ With Voucher": "#0284c7",
                "❌ Denied": "#dc2626",
                "⏳ In-Progress": "#d97706",
                "🚨 RTH Claims": "#ea580c",
                "Attention RTH": "#9333ea"
            }

            # --- TAB 1: DYNAMIC REVENUE TIMELINE ---
            with chart_tab1:
                st.markdown("##### 💵 Revenue Timeline")
                tf_rev_option = st.radio(
                    "📅 Interval:",
                    ["Weekly", "Monthly", "Annually"],
                    horizontal=True,
                    key="revenue_timeline_tf_toggle"
                )

                trend_df = df[df["Action Required"].isin(["✅ Paid Claims", "🎟️ With Voucher"])].copy()
                if not trend_df.empty:
                    trend_df["Anchor_Date"] = pd.to_datetime(trend_df["DISCHARGE DATE"], errors="coerce")
                    trend_df = trend_df.dropna(subset=["Anchor_Date"])
                    
                    if not trend_df.empty:
                        if tf_rev_option == "Weekly":
                            trend_df["Time_Period"] = trend_df["Anchor_Date"].dt.to_period("W").astype(str)
                        elif tf_rev_option == "Monthly":
                            trend_df["Time_Period"] = trend_df["Anchor_Date"].dt.to_period("M").astype(str)
                        else:
                            trend_df["Time_Period"] = trend_df["Anchor_Date"].dt.to_period("Y").astype(str)

                        rev_grouped = trend_df.groupby(["Time_Period", "Action Required"])["Display_Amount"].sum().reset_index()
                        
                        fig = px.bar(
                            rev_grouped,
                            x="Time_Period",
                            y="Display_Amount",
                            color="Action Required",
                            color_discrete_map=status_color_map,
                            labels={"Time_Period": tf_rev_option, "Display_Amount": "Revenue (₱)", "Action Required": "Status"}
                        )
                        fig.update_traces(hovertemplate="Period: %{x}<br>Amount: ₱%{y:,.2f}")
                        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), barmode="stack", height=280)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid date records found for revenue trend.")
                else:
                    st.info("No paid or vouchered claims available.")

            # --- TAB 2: DYNAMIC CLINICAL PACKAGE BREAKDOWN ---
            with chart_tab2:
                st.markdown("##### 🩺 Package Analytics (ICD / RVS)")
                analytics_df = df[df["Action Required"].isin(["✅ Paid Claims", "🎟️ With Voucher"])].copy()

                if not analytics_df.empty:
                    tf_option = st.radio(
                        "📅 Timeframe:",
                        ["Daily", "Weekly", "Monthly", "Annually"],
                        horizontal=True,
                        key="icd_rvs_timeframe_toggle"
                    )

                    analytics_df["Date_Col"] = pd.to_datetime(analytics_df["DISCHARGE DATE"], errors="coerce")
                    analytics_df = analytics_df.dropna(subset=["Date_Col"])

                    if tf_option == "Daily":
                        analytics_df["Time_Period"] = analytics_df["Date_Col"].dt.strftime("%Y-%m-%d")
                    elif tf_option == "Weekly":
                        analytics_df["Time_Period"] = analytics_df["Date_Col"].dt.to_period("W").astype(str)
                    elif tf_option == "Monthly":
                        analytics_df["Time_Period"] = analytics_df["Date_Col"].dt.to_period("M").astype(str)
                    else:
                        analytics_df["Time_Period"] = analytics_df["Date_Col"].dt.to_period("Y").astype(str)

                    grouped_icd = analytics_df.groupby(["Time_Period", "RAW_ICD_RVS", "CLAIM TYPE"])["Display_Amount"].agg(
                        Paid_Count="count",
                        Total_Paid_Amount="sum"
                    ).reset_index()

                    top_paid = grouped_icd.sort_values(by="Total_Paid_Amount", ascending=False)
                    fig_paid = px.bar(
                        top_paid.head(8),
                        x="Total_Paid_Amount",
                        y="CLAIM TYPE",
                        color="Time_Period",
                        orientation="h",
                        labels={"Total_Paid_Amount": "Revenue (₱)", "CLAIM TYPE": "Package"}
                    )
                    fig_paid.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), height=200)
                    st.plotly_chart(fig_paid, use_container_width=True)

                    st.dataframe(
                        top_paid[["Time_Period", "RAW_ICD_RVS", "CLAIM TYPE", "Total_Paid_Amount"]],
                        use_container_width=True,
                        hide_index=True,
                        height=120,
                        column_config={
                            "Time_Period": tf_option,
                            "RAW_ICD_RVS": "ICD/RVS Code",
                            "Total_Paid_Amount": st.column_config.NumberColumn("Total Paid", format="₱%.2f")
                        }
                    )
                else:
                    st.info("No paid or vouchered claims available.")

            # --- TAB 3: CONSOLIDATED DAY-OF-WEEK DISTRIBUTION ---
            with chart_tab3:
                st.markdown("##### 🗓️ Day-of-Week Distribution")
                if "VOUCHER DATE" in df.columns:
                    dow_df = df.copy()
                    dow_df["Voucher_Dt"] = pd.to_datetime(dow_df["VOUCHER DATE"], errors="coerce")
                    dow_df = dow_df.dropna(subset=["Voucher_Dt"])
                    
                    if not dow_df.empty:
                        dow_df["Day_Name"] = dow_df["Voucher_Dt"].dt.day_name()
                        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        
                        dow_grouped = dow_df.groupby(["Day_Name", "Action Required"])["Display_Amount"].agg(
                            Count="count", Total_Amount="sum"
                        ).reset_index()
                        
                        dow_grouped["Day_Name"] = pd.Categorical(dow_grouped["Day_Name"], categories=day_order, ordered=True)
                        dow_grouped = dow_grouped.sort_values("Day_Name")

                        fig = px.bar(
                            dow_grouped,
                            x="Day_Name",
                            y="Count",
                            color="Action Required",
                            color_discrete_map=status_color_map,
                            labels={"Day_Name": "Day of Week", "Count": "Claims"}
                        )
                        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), barmode="stack", height=280)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid voucher dates found.")
                else:
                    st.info("Voucher Date column missing.")

            # --- TAB 4: CLAIMS COUNT TRENDS (DAY, WEEK, MONTH, ANNUAL) ---
            with chart_tab4:
                st.markdown("##### 📊 Claims Count Volume Overview")
                cnt_view = st.radio(
                    "📅 Select Frequency:",
                    ["Daily", "Weekly", "Monthly", "Annually"],
                    horizontal=True,
                    key="claims_count_trend_view_toggle"
                )

                cnt_df = df[df["Action Required"].isin(["✅ Paid Claims", "🎟️ With Voucher"])].copy()
                if not cnt_df.empty:
                    cnt_df["Anchor_Date"] = pd.to_datetime(cnt_df["DISCHARGE DATE"], errors="coerce")
                    cnt_df = cnt_df.dropna(subset=["Anchor_Date"])

                    if not cnt_df.empty:
                        if cnt_view == "Daily":
                            cnt_df["Period_Key"] = cnt_df["Anchor_Date"].dt.strftime("%Y-%m-%d")
                            x_label = "Date"
                        elif cnt_view == "Weekly":
                            cnt_df["Period_Key"] = cnt_df["Anchor_Date"].dt.to_period("W").astype(str)
                            x_label = "Week"
                        elif cnt_view == "Monthly":
                            cnt_df["Period_Key"] = cnt_df["Anchor_Date"].dt.to_period("M").astype(str)
                            x_label = "Month"
                        else:
                            cnt_df["Period_Key"] = cnt_df["Anchor_Date"].dt.to_period("Y").astype(str)
                            x_label = "Year"

                        grouped_cnt = cnt_df.groupby(["Period_Key", "Action Required"])["CLAIM SERIES LHIO"].count().reset_index(name="Claim_Count")

                        fig_cnt = px.bar(
                            grouped_cnt,
                            x="Period_Key",
                            y="Claim_Count",
                            color="Action Required",
                            color_discrete_map=status_color_map,
                            labels={"Period_Key": x_label, "Claim_Count": "Claims Count", "Action Required": "Status"},
                            text_auto=True
                        )
                        fig_cnt.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), barmode="stack", height=280)
                        st.plotly_chart(fig_cnt, use_container_width=True)
                    else:
                        st.info("No valid date records found for claims count trend.")
                else:
                    st.info("No paid or vouchered claims available.")

            # --- TAB 5: HIGHEST-TO-LOWEST PAYOUT INFOGRAPHIC TRENDS ---
            with chart_tab5:
                st.markdown("##### 🏆 Claims Payout Infographic (Highest to Lowest)")
                payout_view = st.radio(
                    "📅 Frequency View:",
                    ["Daily", "Weekly", "Monthly", "Annually"],
                    horizontal=True,
                    key="payout_ranking_view_toggle"
                )

                payout_df = df[df["Action Required"].isin(["✅ Paid Claims", "🎟️ With Voucher"])].copy()
                if not payout_df.empty:
                    payout_df["Anchor_Date"] = pd.to_datetime(payout_df["DISCHARGE DATE"], errors="coerce")
                    payout_df = payout_df.dropna(subset=["Anchor_Date"])

                    if not payout_df.empty:
                        if payout_view == "Daily":
                            payout_df["Period_Key"] = payout_df["Anchor_Date"].dt.strftime("%Y-%m-%d")
                        elif payout_view == "Weekly":
                            payout_df["Period_Key"] = payout_df["Anchor_Date"].dt.to_period("W").astype(str)
                        elif payout_view == "Monthly":
                            payout_df["Period_Key"] = payout_df["Anchor_Date"].dt.to_period("M").astype(str)
                        else:
                            payout_df["Period_Key"] = payout_df["Anchor_Date"].dt.to_period("Y").astype(str)

                        ranked_payout = payout_df.groupby("Period_Key").agg(
                            Total_Payout=("Display_Amount", "sum"),
                            Claim_Count=("CLAIM SERIES LHIO", "count")
                        ).reset_index()

                        ranked_payout = ranked_payout.sort_values(by="Total_Payout", ascending=False)

                        fig_ranked = px.bar(
                            ranked_payout,
                            x="Period_Key",
                            y="Total_Payout",
                            text=ranked_payout["Total_Payout"].apply(lambda v: f"₱{v:,.0f}"),
                            labels={"Period_Key": payout_view, "Total_Payout": "Total Payout (₱)"},
                            color="Total_Payout",
                            color_continuous_scale="Blues"
                        )
                        fig_ranked.update_traces(textposition="outside")
                        fig_ranked.update_layout(
                            template="plotly_white", 
                            margin=dict(l=10, r=10, t=20, b=10), 
                            height=280,
                            coloraxis_showscale=False,
                            xaxis=dict(categoryorder="total descending")
                        )
                        st.plotly_chart(fig_ranked, use_container_width=True)

                        st.markdown("##### 📋 Ranked Summary Table")
                        st.dataframe(
                            ranked_payout.rename(columns={"Period_Key": f"Period ({payout_view})", "Total_Payout": "Total Payout", "Claim_Count": "Claims Count"}),
                            use_container_width=True,
                            hide_index=True,
                            height=150,
                            column_config={
                                "Total Payout": st.column_config.NumberColumn("Total Payout", format="₱%.2f"),
                                "Claims Count": st.column_config.NumberColumn("Claims Count", format="%d")
                            }
                        )
                    else:
                        st.info("No valid date records found for payout ranking.")
                else:
                    st.info("No paid or vouchered claims available.")

            # --- TAB 6: ICD/RVS CODE COUNT (DAILY, WEEKLY, MONTHLY, ANNUALLY) ---
            with chart_tab6:
                st.markdown("##### 🏷️ ICD/RVS Code Claim Count Breakdown")
                icd_view = st.radio(
                    "📅 Frequency:",
                    ["Daily", "Weekly", "Monthly", "Annually"],
                    horizontal=True,
                    key="icd_code_count_view_toggle"
                )

                icd_cnt_df = df[df["Action Required"].isin(["✅ Paid Claims", "🎟️ With Voucher"])].copy()
                if not icd_cnt_df.empty:
                    icd_cnt_df["Anchor_Date"] = pd.to_datetime(icd_cnt_df["DISCHARGE DATE"], errors="coerce")
                    icd_cnt_df = icd_cnt_df.dropna(subset=["Anchor_Date"])

                    if not icd_cnt_df.empty:
                        if icd_view == "Daily":
                            icd_cnt_df["Period_Key"] = icd_cnt_df["Anchor_Date"].dt.strftime("%Y-%m-%d")
                        elif icd_view == "Weekly":
                            icd_cnt_df["Period_Key"] = icd_cnt_df["Anchor_Date"].dt.to_period("W").astype(str)
                        elif icd_view == "Monthly":
                            icd_cnt_df["Period_Key"] = icd_cnt_df["Anchor_Date"].dt.to_period("M").astype(str)
                        else:
                            icd_cnt_df["Period_Key"] = icd_cnt_df["Anchor_Date"].dt.to_period("Y").astype(str)

                        icd_grouped = icd_cnt_df.groupby(["Period_Key", "RAW_ICD_RVS", "CLAIM TYPE"]).agg(
                            Claim_Count=("CLAIM SERIES LHIO", "count"),
                            Total_Revenue=("Display_Amount", "sum")
                        ).reset_index()

                        icd_grouped = icd_grouped.sort_values(by="Claim_Count", ascending=False)

                        fig_icd_cnt = px.bar(
                            icd_grouped,
                            x="Period_Key",
                            y="Claim_Count",
                            color="RAW_ICD_RVS",
                            labels={"Period_Key": icd_view, "Claim_Count": "Claims Count", "RAW_ICD_RVS": "ICD/RVS Code"},
                            text_auto=True
                        )
                        fig_icd_cnt.update_layout(
                            template="plotly_white", 
                            margin=dict(l=10, r=10, t=10, b=10), 
                            barmode="stack", 
                            height=280
                        )
                        st.plotly_chart(fig_icd_cnt, use_container_width=True)

                        st.markdown("##### 📋 ICD/RVS Code Count Table")
                        st.dataframe(
                            icd_grouped.rename(columns={"Period_Key": f"Period ({icd_view})", "RAW_ICD_RVS": "ICD/RVS Code", "CLAIM TYPE": "Package Description", "Claim_Count": "Claim Count", "Total_Revenue": "Total Revenue"}),
                            use_container_width=True,
                            hide_index=True,
                            height=150,
                            column_config={
                                "Claim Count": st.column_config.NumberColumn("Claim Count", format="%d"),
                                "Total Revenue": st.column_config.NumberColumn("Total Revenue", format="₱%.2f")
                            }
                        )
                    else:
                        st.info("No valid date records found for ICD/RVS code counts.")
                else:
                    st.info("No paid or vouchered claims available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 7. SMART ACTIONABLE RTH RESOLUTION MODULE EXPANDER (COLLAPSED BY DEFAULT) ---
    with st.expander("🎯 Smart Actionable RTH Resolution Module & 60-Day Countdown", expanded=False):
        st.markdown("##### ⚡ Active Return-to-Hospital (RTH) Action Center")
        st.caption("Manage, track, and refile returned claims before the PhilHealth 60-calendar-day deadline expires.")

        if not rth_df.empty:
            rth_action_tab1, rth_action_tab2 = st.tabs([
                "🚨 Discrepancy Action Center (Text Available)",
                "❓ Pending PhilHealth Notice (Blank Text)"
            ])

            today_date = datetime.today().date()

            with rth_action_tab1:
                if not attn_df.empty:
                    st.markdown(f"**Found {len(attn_df)} claims with explicit PhilHealth discrepancy notes.**")
                    
                    for idx, row in attn_df.iterrows():
                        claim_series = str(row.get("CLAIM SERIES LHIO", "N/A"))
                        patient_name = str(row.get("PATIENT NAME", "Unknown Patient"))
                        raw_icd_rvs = str(row.get("RAW_ICD_RVS", "N/A"))
                        claim_type = str(row.get("CLAIM TYPE", "General"))
                        discrepancy_text = str(row.get("RETURN DISCREPANCY", "No text provided"))
                        category_tag = str(row.get("Discrepancy Category", "General"))
                        caserate_val = row.get("Display_Amount", 0.0)

                        with st.container():
                            c_info, c_date, c_status = st.columns([2.5, 1.2, 1.3])

                            with c_info:
                                st.markdown(f"### 👤 {patient_name}")
                                st.markdown(f"**Series LHIO:** `{claim_series}` | **ICD/RVS Code:** `{raw_icd_rvs}` | **Package:** `{claim_type}` | **Case Rate:** ₱{caserate_val:,.2f}")
                                st.markdown(f"🏷️ **Category:** `{category_tag}`")
                                st.info(f"📌 **PhilHealth Discrepancy Note:**\n\n_{discrepancy_text}_")

                            with c_date:
                                saved_date = st.session_state["rth_notice_dates"].get(claim_series, today_date)
                                notice_date = st.date_input(
                                    "📅 RTH Notice Date:",
                                    value=saved_date,
                                    key=f"notice_date_{claim_series}_{idx}"
                                )
                                st.session_state["rth_notice_dates"][claim_series] = notice_date

                            with c_status:
                                refile_deadline = notice_date + timedelta(days=60)
                                days_remaining = (refile_deadline - today_date).days

                                if days_remaining < 0:
                                    st.error(f"❌ **EXPIRED**\n\n{abs(days_remaining)} days past 60-day limit!")
                                elif days_remaining <= 15:
                                    st.error(f"🚨 **CRITICAL**\n\n**{days_remaining} Days Left**\nDeadline: {refile_deadline.strftime('%m-%d-%Y')}")
                                elif days_remaining <= 30:
                                    st.warning(f"⚠️ **WARNING**\n\n**{days_remaining} Days Left**\nDeadline: {refile_deadline.strftime('%m-%d-%Y')}")
                                else:
                                    st.success(f"🟢 **ON TRACK**\n\n**{days_remaining} Days Left**\nDeadline: {refile_deadline.strftime('%m-%d-%Y')}")

                            st.markdown("---")
                else:
                    st.success("🎉 No active RTH claims with explicit discrepancy notices required.")

            with rth_action_tab2:
                blank_rth_df = rth_df[~rth_df.index.isin(attn_df.index)]
                if not blank_rth_df.empty:
                    st.markdown(f"**Found {len(blank_rth_df)} claims marked 'Return' but pending text details in Bizbox.**")
                    
                    pullout_data = blank_rth_df[[
                        "TRANSMITTAL ID", "CLAIM SERIES LHIO", "PATIENT NAME", "RAW_ICD_RVS", "CLAIM TYPE", "PATIENT PIN", "DISCHARGE DATE", "TRANSMITTED ON", "Display_Amount"
                    ]].copy()

                    st.dataframe(
                        pullout_data,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "RAW_ICD_RVS": st.column_config.TextColumn("ICD/RVS CODE"),
                            "Display_Amount": st.column_config.NumberColumn("AMOUNT", format="₱%.2f"),
                            "DISCHARGE DATE": st.column_config.DateColumn("DISCHARGE DATE", format="MM-DD-YYYY"),
                            "TRANSMITTED ON": st.column_config.DateColumn("TRANSMITTED ON", format="MM-DD-YYYY")
                        }
                    )
                else:
                    st.info("No pending RTH claims without text.")

            st.markdown("##### 📄 Medical Records Chart Pull-Out Worklist")
            rth_summary_export = rth_df[[
                "CLAIM SERIES LHIO", "TRANSMITTAL ID", "PATIENT NAME", "RAW_ICD_RVS", "CLAIM TYPE", "PATIENT PIN", "DISCHARGE DATE", "RETURN DISCREPANCY", "Discrepancy Category", "Display_Amount"
            ]].copy()
            
            csv_rth = rth_summary_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="🖨️ Download Medical Records RTH Pull-Out Worklist (CSV)",
                data=csv_rth,
                file_name=f"RTH_Medical_Records_Pullout_{datetime.today().strftime('%m-%d-%Y')}.csv",
                mime="text/csv"
            )

        else:
            st.success("✅ Zero Return-to-Hospital (RTH) claims found in current dataset!")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 8. GLOBAL DATA FILTER & REUSABLE RENDER HELPER ---
    st.markdown("---")
    st.markdown("### 🔍 Global Data Filter & Search Engine")
    
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    
    with f_col1:
        global_search = st.text_input(
            "Global Search (Applies across all tabs)",
            placeholder="Type patient name, series LHIO, PIN, transmittal ID, or ICD code...",
            key="global_search_query"
        ).strip().upper()
        
    with f_col2:
        available_claim_types = ["All"] + sorted(df["CLAIM TYPE"].dropna().unique().tolist()) if "CLAIM TYPE" in df.columns else ["All"]
        selected_claim_type = st.selectbox(
            "Filter by Claim Type",
            options=available_claim_types,
            key="global_claim_type_filter"
        )
        
    with f_col3:
        date_sort_order = st.selectbox(
            "Sort Discharge Date",
            options=["Newest First", "Oldest First"],
            key="global_date_sort"
        )

    filtered_df = df.copy()
    
    if global_search:
        if "_SEARCH_INDEX" in filtered_df.columns:
            mask = filtered_df["_SEARCH_INDEX"].str.contains(global_search, na=False)
            filtered_df = filtered_df[mask]
        else:
            mask = filtered_df.astype(str).apply(lambda x: x.str.upper().str.contains(global_search)).any(axis=1)
            filtered_df = filtered_df[mask]
            
    if selected_claim_type != "All":
        filtered_df = filtered_df[filtered_df["CLAIM TYPE"] == selected_claim_type]
        
    if "DISCHARGE DATE" in filtered_df.columns:
        ascending_bool = True if date_sort_order == "Oldest First" else False
        filtered_df = filtered_df.sort_values(by="DISCHARGE DATE", ascending=ascending_bool)

    def render_customizable_table_section(
        data_subset, tab_key_prefix, default_custom_cols=None
    ):
        if default_custom_cols is None:
            default_custom_cols = [
                "STATUS",
                "RAW_ICD_RVS",
                "CLAIM TYPE",
                "TRANSMITTAL ID",
                "CLAIM SERIES LHIO",
                "PATIENT NAME",
                "PATIENT PIN",
                "DISCHARGE DATE",
                "TRANSMITTED ON",
                "VOUCHER DATE",
                "Working Days Today",
                "Working Days Thursday",
                "Working Days Friday",
                "Bank Withdrawal Readiness",
                "CLAIM AMOUNT",
                "Action Required",
            ]

        saved_order_key = f"saved_column_order_{tab_key_prefix}"
        ordered_cols_key = f"ordered_columns_{tab_key_prefix}"
        col_to_move_key = f"col_to_move_{tab_key_prefix}"
        multiselect_key = f"multiselect_visible_cols_{tab_key_prefix}"

        if saved_order_key not in st.session_state:
            st.session_state[saved_order_key] = [
                c for c in default_custom_cols if c in data_subset.columns
            ]

        if ordered_cols_key not in st.session_state:
            st.session_state[ordered_cols_key] = list(
                st.session_state[saved_order_key]
            )

        with st.expander(
            "⚙️ Customize Columns & Reorder Headers", expanded=False
        ):
            all_available_cols = [c for c in data_subset.columns if c != "_SEARCH_INDEX"]

            selected_cols = st.multiselect(
                "Select visible columns:",
                options=all_available_cols,
                default=[
                    c
                    for c in st.session_state[ordered_cols_key]
                    if c in all_available_cols
                ],
                key=multiselect_key,
            )

            st.session_state[ordered_cols_key] = [
                c for c in st.session_state[ordered_cols_key] if c in selected_cols
            ]
            for c in selected_cols:
                if c not in st.session_state[ordered_cols_key]:
                    st.session_state[ordered_cols_key].append(c)

            st.markdown("##### ↕️ Reorder Columns")

            if (
                col_to_move_key not in st.session_state
                or st.session_state[col_to_move_key]
                not in st.session_state[ordered_cols_key]
            ):
                if st.session_state[ordered_cols_key]:
                    st.session_state[col_to_move_key] = st.session_state[
                        ordered_cols_key
                    ][0]

            col_to_move = st.selectbox(
                "Choose column to adjust:",
                options=st.session_state[ordered_cols_key],
                key=col_to_move_key,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            b_reset, b_left, b_right, spacer, b_save = st.columns(
                [1, 1, 1, 2.5, 1]
            )

            with b_reset:
                if st.button(
                    "🔄 Reset Table Layout",
                    key=f"btn_reset_{tab_key_prefix}",
                    use_container_width=True,
                ):
                    st.session_state[saved_order_key] = [
                        c for c in default_custom_cols if c in data_subset.columns
                    ]
                    st.session_state[ordered_cols_key] = list(
                        st.session_state[saved_order_key]
                    )
                    st.success("Layout reset to default headers!")
                    st.rerun()

            with b_left:
                if st.button(
                    "⬅️ Move Left",
                    key=f"btn_left_{tab_key_prefix}",
                    use_container_width=True,
                ):
                    if col_to_move in st.session_state[ordered_cols_key]:
                        idx = st.session_state[ordered_cols_key].index(col_to_move)
                        if idx > 0:
                            st.session_state[ordered_cols_key].insert(
                                idx - 1, st.session_state[ordered_cols_key].pop(idx)
                            )
                            st.rerun()

            with b_right:
                if st.button(
                    "Move Right ➡️",
                    key=f"btn_right_{tab_key_prefix}",
                    use_container_width=True,
                ):
                    if col_to_move in st.session_state[ordered_cols_key]:
                        idx = st.session_state[ordered_cols_key].index(col_to_move)
                        if idx < len(st.session_state[ordered_cols_key]) - 1:
                            st.session_state[ordered_cols_key].insert(
                                idx + 1, st.session_state[ordered_cols_key].pop(idx)
                            )
                            st.rerun()

            with b_save:
                if st.button(
                    "💾 Save Layout",
                    key=f"btn_save_{tab_key_prefix}",
                    use_container_width=True,
                ):
                    st.session_state[saved_order_key] = list(
                        st.session_state[ordered_cols_key]
                    )
                    st.success("Layout saved!")

        cols_to_show = st.session_state.get(
            ordered_cols_key, default_custom_cols
        )
        cols_to_show = [c for c in cols_to_show if c in data_subset.columns]

        if data_subset.empty:
            st.info("No records found matching the current criteria.")
            return

        total_amount = (
            pd.to_numeric(
                data_subset["Display_Amount"], errors="coerce"
            ).sum()
            if "Display_Amount" in data_subset.columns
            else 0.0
        )

        display_df = data_subset[cols_to_show].copy()
        display_df.insert(0, "No.", range(1, len(display_df) + 1))

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "No.": st.column_config.NumberColumn("No.", width="small"),
                "RAW_ICD_RVS": st.column_config.TextColumn("ICD/RVS CODE"),
                "CLAIM AMOUNT": st.column_config.NumberColumn("CLAIM AMOUNT", format="₱%.2f"),
                "1ST CASERATE AMOUNT": st.column_config.NumberColumn("1ST CASERATE AMOUNT", format="₱%.2f"),
                "Display_Amount": st.column_config.NumberColumn("AMOUNT", format="₱%.2f"),
                "DISCHARGE DATE": st.column_config.DateColumn("DISCHARGE DATE", format="MM-DD-YYYY"),
                "TRANSMITTED ON": st.column_config.DateColumn("TRANSMITTED ON", format="MM-DD-YYYY"),
                "VOUCHER DATE": st.column_config.DateColumn("VOUCHER DATE", format="MM-DD-YYYY"),
                "Working Days Today": st.column_config.NumberColumn("WDAYS TODAY", format="%d Days"),
                "Working Days Thursday": st.column_config.NumberColumn("WDAYS THU", format="%d Days"),
                "Working Days Friday": st.column_config.NumberColumn("WDAYS FRI", format="%d Days"),
            }
        )

        st.markdown(
            f"<div style='text-align: right; font-weight: 600; color: #1e293b; margin-top: 8px;'>Total Records: {len(data_subset)} | Combined Amount: ₱{total_amount:,.2f}</div>",
            unsafe_allow_html=True,
        )

    # --- 9. TABULAR ACTION MODULES ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "✅ Paid Claims",
        "🎟️ With Voucher",
        "❌ Denied",
        "⏳ In-Progress",
        "🚨 RTH Claims",
        "Attention RTH",
        "📁 All Records",
    ])

    paid_headers = [
        "STATUS",
        "RAW_ICD_RVS",
        "CLAIM TYPE",
        "TRANSMITTAL ID",
        "CLAIM SERIES LHIO",
        "PATIENT NAME",
        "PATIENT PIN",
        "DISCHARGE DATE",
        "TRANSMITTED ON",
        "CLAIM AMOUNT",
        "Days Since Transmitted On",
        "Days Since Voucher Date",
        "Action Required",
    ]

    vouch_headers = [
        "STATUS",
        "RAW_ICD_RVS",
        "CLAIM TYPE",
        "TRANSMITTAL ID",
        "CLAIM SERIES LHIO",
        "PATIENT NAME",
        "VOUCHER DATE",
        "Working Days Today",
        "Working Days Thursday",
        "Working Days Friday",
        "Bank Withdrawal Readiness",
        "1ST CASERATE AMOUNT",
        "Action Required",
    ]

    caserate_headers = [
        "STATUS",
        "RAW_ICD_RVS",
        "CLAIM TYPE",
        "TRANSMITTAL ID",
        "CLAIM SERIES LHIO",
        "PATIENT NAME",
        "PATIENT PIN",
        "DISCHARGE DATE",
        "TRANSMITTED ON",
        "CLAIM AMOUNT",
        "Days Since Transmitted On",
        "Days Since Voucher Date",
        "Action Required",
        "1ST CASERATE AMOUNT",
    ]

    with tab1:
        st.subheader("Paid Claims (Status: With Cheque)")
        paid_subset = filtered_df[filtered_df["Action Required"] == "✅ Paid Claims"]
        render_customizable_table_section(
            paid_subset, tab_key_prefix="paid", default_custom_cols=paid_headers
        )

    with tab2:
        st.subheader("Claims With Voucher (Status: With Voucher)")
        vouch_subset = filtered_df[filtered_df["Action Required"] == "🎟️ With Voucher"]
        render_customizable_table_section(
            vouch_subset, tab_key_prefix="vouch", default_custom_cols=vouch_headers
        )

    with tab3:
        st.subheader("Denied Claims (Status: Denied)")
        den_subset = filtered_df[filtered_df["Action Required"] == "❌ Denied"].copy()
        render_customizable_table_section(
            den_subset, tab_key_prefix="denied", default_custom_cols=caserate_headers
        )

    with tab4:
        st.subheader("In-Progress Claims (Status: In Process)")
        prog_subset = filtered_df[filtered_df["Action Required"] == "⏳ In-Progress"]
        render_customizable_table_section(
            prog_subset, tab_key_prefix="prog", default_custom_cols=caserate_headers
        )

    with tab5:
        st.subheader("RTH Claims (All Status: Return)")
        rth_subset = filtered_df[filtered_df["STATUS"].astype(str).str.strip().str.title() == "Return"].copy()
        render_customizable_table_section(
            rth_subset, tab_key_prefix="rth", default_custom_cols=caserate_headers
        )

    with tab6:
        st.subheader("Attention RTH Claims (Status: Return with Actual Discrepancy Text Only)")
        attn_subset = filtered_df[filtered_df["Action Required"] == "Attention RTH"].copy()
        render_customizable_table_section(
            attn_subset, tab_key_prefix="attn", default_custom_cols=caserate_headers
        )

    with tab7:
        st.subheader("Complete Master Claims Registry")
        render_customizable_table_section(
            filtered_df, tab_key_prefix="all", default_custom_cols=caserate_headers
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 10. EXPORT CENTER ---
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("### 📤 Export Center")
        st.markdown("#### CSV Worklist Export")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Complete Worklist (CSV)",
            data=csv_data,
            file_name=f"PhilHealth_Worklist_{datetime.today().strftime('%m-%d-%Y')}.csv",
            mime="text/csv",
        )

    with col_exp2:
        st.markdown('<div class="right-aligned-column">', unsafe_allow_html=True)
        st.markdown("### Encrypted Excel Report (Password Protected)")
        excel_password = st.text_input(
            "Encryption Password", value="PhilHealth2026!", type="password"
        )

        if st.button("Generate Encrypted Excel Report"):
            with st.status("🔒 Encrypting Excel workbook...", expanded=True) as enc_status:
                try:
                    st.write("📊 Compiling sheets and formatting rows...")
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Claims_Report")
                    excel_buffer.seek(0)

                    st.write("🔐 Applying military-grade encryption...")
                    encrypted_buffer = io.BytesIO()
                    office_file = msoffcrypto.OfficeFile(excel_buffer)
                    office_file.encrypt(excel_password, encrypted_buffer)
                    encrypted_buffer.seek(0)

                    enc_status.update(label="✅ Excel report successfully encrypted!", state="complete", expanded=False)
                    
                    st.download_button(
                        label="Download Encrypted Excel (.xlsx)",
                        data=encrypted_buffer,
                        file_name=(
                            f"Secure_PhilHealth_Report_{datetime.today().strftime('%m-%d-%Y')}.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        ),
                    )
                    st.success("Excel report encrypted successfully.")
                except Exception as e:
                    enc_status.update(label="❌ Encryption failed.", state="error", expanded=True)
                    st.error(f"Encryption failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👋 Welcome! Please expand the **Data Ingestion Center** above to upload your Bizbox claims export file.")