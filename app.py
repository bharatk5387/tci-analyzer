import io
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, Tuple, List

from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Marketing Studio — IPD Analysis Dashboard", layout="wide")


# -----------------------------
# Helpers: load file
# -----------------------------
@st.cache_data(show_spinner=False)
def load_table(file_name: str, file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    lower = file_name.lower()

    if lower.endswith(".parquet"):
        import pyarrow.parquet as pq
        import pyarrow as pa
        table = pq.read_table(pa.BufferReader(file_bytes))
        return table.to_pandas()

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        bio = io.BytesIO(file_bytes)
        if sheet_name:
            return pd.read_excel(bio, sheet_name=sheet_name)
        return pd.read_excel(bio)

    # default CSV
    bio = io.BytesIO(file_bytes)
    return pd.read_csv(bio, engine="python")


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_int_safe(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


# -----------------------------
# Success / Partial success normalization
# -----------------------------
def normalize_success_flag(status_val) -> Optional[bool]:
    """
    Returns:
      True  -> success OR partial success
      False -> fail/abandoned/unfixable/etc
      None  -> unknown
    """
    if status_val is None or (isinstance(status_val, float) and pd.isna(status_val)):
        return None

    s = str(status_val).strip().lower()

    # treat these as "eventually successful or partially successful"
    success_tokens = [
        "success", "succeeded", "pass", "passed",
        "partial success", "partial pass", "partially successful"
    ]
    fail_tokens = [
        "fail", "failed", "abandoned", "unfixable", "cancelled", "canceled"
    ]

    if any(t in s for t in success_tokens):
        return True
    if any(t in s for t in fail_tokens):
        return False

    return None


# -----------------------------
# Classification (bucket issues using intelligent analysis)
# -----------------------------
def classify_issue(text: str, analysis: str = "", additional_info: str = "") -> Tuple[str, str]:
    """
    Intelligently classify issues by analyzing multiple text sources.
    
    PRIORITIZATION:
    - 'analysis' parameter (analysis_text from the "Actual Issues" column in upload) = PRIMARY source
    - 'text' and 'additional_info' = Secondary/supplementary context
    
    Returns:
    - bucket: Input | System | Unknown
    - category: detailed label (e.g., "File & sheet structure", "Model/agent layer", etc.)
    """
    # PRIMARY SOURCE: analysis_text (from "Actual issues" column)
    # Use it first if available, then supplement with other text
    if analysis and analysis.strip():
        # Analysis_text is present - use as primary classification driver
        primary_msg = (analysis or "").lower()
        secondary_msg = " | ".join([(text or ""), (additional_info or "")]).lower()
        msg = f"{primary_msg} | {secondary_msg}"
    else:
        # Fallback: analysis_text is empty, combine all available text sources
        msg = " | ".join([
            (text or ""),
            (analysis or ""),
            (additional_info or "")
        ]).lower()
    
    if not msg.strip() or msg.strip() == "| |" or msg.strip() == "||":
        return ("Unknown", "No information provided")

    # Input/workbook/spec issues (enhanced patterns)
    input_rules = [
        ("File & sheet structure", r"(sheet|tab|header|column name|template|workbook|excel|xlsx|csv|format|structure|file structure|wrong format|incorrect format|column mismatch)"),
        ("Required fields missing", r"(missing|required|empty field|null value|blank|not provided|absent|no value|field not found|mandatory)"),
        ("Business-rule validation", r"(invalid|not allowed|out of range|frequency cap|budget|scheduled in the past|too long|too short|exceeds|below minimum|above maximum|violates rule|validation failed|business rule)"),
        ("Referential integrity", r"(not found|doesn't exist|does not exist|unknown id|asset not found|audience not found|catalog|reference error|broken reference|missing reference|id mismatch)"),
        ("Compliance & consent", r"(pii|consent|unsubscribe|legal|policy|prohibited|restricted|compliance|gdpr|privacy|regulation|unauthorized use)"),
        ("Channel/placement specs", r"(placement|aspect ratio|resolution|file size|duration|objective|optimization|creative spec|ad spec|dimension|size requirement|channel requirement)"),
        ("Tracking & analytics", r"(pixel|tag|tracking|analytics|utm|event|conversion|measurement|tracking code|analytics setup)"),
        ("Data quality & hygiene", r"(duplicate|duplicated|inconsistent|mismatch|stale|old version|data quality|bad data|corrupted|invalid data|malformed)"),
        ("Attachments & external references", r"(drive|link|url|permission|access denied|expired|attachment|external file|shared drive|file access|broken link)"),
        ("Content & creative issues", r"(image|video|creative|asset|media|content|copy|text|headline|description|visual|graphic)"),
        ("Budget & scheduling", r"(budget|spend|cost|billing|payment|schedule|timing|date|start date|end date|flight|pacing)"),
        ("Targeting & audience", r"(targeting|audience|segment|demographic|geography|location|age|gender|interest|lookalike)"),
    ]
    for cat, pat in input_rules:
        if re.search(pat, msg):
            return ("Input", cat)

    # System/platform issues (enhanced patterns)
    system_rules = [
        ("Ingestion & parsing pipeline", r"(parser|parse|ingest|pipeline|schema|deserialize|serialization|etl|data pipeline|parsing error|ingestion failed)"),
        ("Model/agent layer", r"(timeout|throttle|rate limit|5\d\d|model|llm|generation|invalid output|ai error|model failure|generation failed|prompt issue)"),
        ("Storage & data services", r"(db|database|connection|pool|cache|blob|storage|data store|query failed|database error|connection refused)"),
        ("Auth & permissions", r"(oauth|token|auth|permission|unauthorized|forbidden|403|401|authentication|authorization|credential|access token)"),
        ("Integrations & APIs", r"(api outage|integration|downstream|third party|webhook|api error|endpoint|service unavailable|external service|api call failed)"),
        ("Orchestration & workflow", r"(worker|queue|stuck|pending|job failed|orchestration|dispatch|workflow|task failed|background job|processing error)"),
        ("UX/observability", r"(unknown error|generic error|no details|telemetry loss|retry later|unexpected error|system error|internal error)"),
        ("Network & connectivity", r"(network|connectivity|connection timeout|dns|latency|packet loss|network error|unavailable|unreachable)"),
        ("Platform bugs", r"(bug|crash|exception|stack trace|null pointer|index out of bounds|runtime error|platform issue)"),
    ]
    for cat, pat in system_rules:
        if re.search(pat, msg):
            return ("System", cat)

    return ("Unknown", "Unclassified - needs review")


# -----------------------------
# TCI computation
# -----------------------------
def compute_metrics(df: pd.DataFrame, id_col: str, tries_col: str, status_col: str, 
                    reason_col: Optional[str], analysis_col: Optional[str], 
                    additional_cols: List[str], date_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()

    out.rename(columns={id_col: "ipd_id", tries_col: "tries_raw", status_col: "status_raw"}, inplace=True)
    out["tries"] = out["tries_raw"].apply(to_int_safe)
    out["is_success_or_partial"] = out["status_raw"].apply(normalize_success_flag)
    
    # Handle date column if provided
    if date_col and date_col in out.columns:
        out.rename(columns={date_col: "ipd_date"}, inplace=True)
        # Parse date with flexibility
        out["ipd_date"] = pd.to_datetime(out["ipd_date"], errors='coerce')
    else:
        out["ipd_date"] = None

    # Per-campaign TCI (only meaningful for success/partial success)
    # TCI = (1 / tries) * 100
    out["TCI"] = out.apply(
        lambda r: (100.0 / r["tries"]) if (r["is_success_or_partial"] is True and r["tries"] and r["tries"] > 0) else None,
        axis=1
    )

    # Store classification source columns
    if reason_col and reason_col in out.columns:
        out.rename(columns={reason_col: "reason_text"}, inplace=True)
    else:
        out["reason_text"] = None

    if analysis_col and analysis_col in out.columns:
        out.rename(columns={analysis_col: "analysis_text"}, inplace=True)
    else:
        out["analysis_text"] = None

    # Intelligent classification using analysis_text as PRIMARY source
    buckets, cats = [], []
    for idx, row in out.iterrows():
        # analysis_text (from the "Actual Issues" column in the uploaded file) is the PRIMARY source for bucketing
        analysis_txt = str(row.get("analysis_text", "")).strip() if row.get("analysis_text") is not None else ""
        
        # reason_text and additional columns provide supplementary context
        reason_txt = str(row.get("reason_text", "")).strip() if row.get("reason_text") is not None else ""
        additional_txt = " ".join([
            str(row.get(col, "")).strip() 
            for col in additional_cols 
            if col in row and row.get(col) is not None
        ])
        
        # Pass analysis_text as the primary parameter for classification
        b, c = classify_issue(reason_txt, analysis_txt, additional_txt)
        buckets.append(b)
        cats.append(c)

    out["issue_bucket"] = buckets
    out["issue_category"] = cats

    return out


def overall_tci_weighted(success_df: pd.DataFrame, all_df: pd.DataFrame) -> float:
    """
    Overall TCI formula:
    total successes (counted as 1 each) / total tries across ALL campaigns (including failed/abandoned) * 100
    """
    if success_df.empty:
        return 0.0
    num_success = float(len(success_df))
    # Use ALL tries from ALL interactions (success, failed, abandoned, etc.)
    denom = float(all_df["tries"].dropna().sum())
    if denom <= 0:
        return 0.0
    return (num_success / denom) * 100.0


# -----------------------------
# UI
# -----------------------------
st.title("Marketing Studio — IPD Success & Failure Analysis")

st.markdown(
    """
**What this dashboard does:**
- Analyzes **In-Platform Delivery (IPD)** campaign data from Excel/CSV with **tries**, **status**, **failure reasons**, and **date**.
- **Primary Classification Source**: Uses the **"Actual Issues"** column from your upload to intelligently categorize failures into **Input vs Platform (System)** errors.
- Tracks key metrics:
  1. **# IPDs Created Successfully**: Count of successful deliveries
  2. **# IPDs Failed with Reason**: Count of failures, segregated by Input vs Platform errors
  3. **# Retries Required Before Success**: Average attempts needed for successful IPDs
  4. **Failed Customer Interactions**: (# of failed tries) / (total # of tries) as percentage
- **Error Categorization**: 15+ detailed categories for root cause analysis
- **Visual Analytics**: Colorful pie charts showing Input vs Platform error distribution
- **Trend Analysis**: Track performance metrics over time
"""
)

with st.sidebar:
    st.header("Upload")
    upload = st.file_uploader("Upload .xlsx / .csv / .parquet", type=["xlsx", "xls", "csv", "parquet"])
    sheet = None
    if upload and upload.name.lower().endswith((".xlsx", ".xls")):
        sheet = st.text_input("Excel sheet name (optional)", value="")
        if sheet.strip() == "":
            sheet = None

if not upload:
    st.info("Upload a file to begin.")
    st.stop()

df_raw = load_table(upload.name, upload.getvalue(), sheet_name=sheet)
df_raw = clean_cols(df_raw)

# Column mapping
st.sidebar.header("Column mapping")
cols = [""] + list(df_raw.columns)

def pick(label: str, default_guess: Optional[str], key: str) -> str:
    idx = 0
    if default_guess and default_guess in cols:
        idx = cols.index(default_guess)
    return st.sidebar.selectbox(label, cols, index=idx, key=key)

def guess_col(patterns: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df_raw.columns}
    for p in patterns:
        for cl, orig in low.items():
            if p in cl:
                return orig
    return None

id_guess = guess_col(["elm id", "lm campaign", "campaign id", "campaign_id", "ipd", "id", "interaction id", "interaction_id"])
tries_guess = guess_col(["tries", "attempt", "rounds of testing", "rounds", "runs"])
status_guess = guess_col(["final outcome", "status", "outcome", "result"])
reason_guess = guess_col(["reason", "error", "issue", "failure", "notes", "comment", "reported issue"])
# Look for "Actual Issues" column specifically - this is PRIMARY for classification
analysis_guess = guess_col(["actual issues", "actual issue"])
date_guess = guess_col(["date", "created", "timestamp", "time", "tried date", "attempt date", "test date", "submission date"])

st.sidebar.markdown("### Required columns")
id_col = pick("IPD/Campaign ID column", id_guess, "id_col")
tries_col = pick("Number of tries/attempts column", tries_guess, "tries_col")
status_col = pick("Final status column", status_guess, "status_col")
date_col = st.sidebar.selectbox("Date column (for trend analysis)", cols, index=(cols.index(date_guess) if date_guess in cols else 0), key="date_col")

st.sidebar.markdown("### Issue classification columns")
st.sidebar.caption("Map the 'Actual Issues' column for Input vs Platform error segregation")
reason_col = st.sidebar.selectbox("Reported issue/reason column (optional)", cols, index=(cols.index(reason_guess) if reason_guess in cols else 0), key="reason_col")
analysis_col = st.sidebar.selectbox("📌 'Actual Issues' column (PRIMARY)", cols, index=(cols.index(analysis_guess) if analysis_guess in cols else 0), key="analysis_col")
st.sidebar.caption("🎯 The 'Actual Issues' column drives Input vs Platform classification")

st.sidebar.markdown("### Additional context columns")
st.sidebar.caption("Select any other columns with relevant information")
additional_cols = st.sidebar.multiselect(
    "Additional columns for classification",
    options=[c for c in df_raw.columns if c not in [id_col, tries_col, status_col, reason_col, analysis_col]],
    default=[],
    key="additional_cols"
)

if id_col == "" or tries_col == "" or status_col == "":
    st.error("Please map IPD/Campaign ID, Tries, and Status columns in the sidebar.")
    st.stop()

reason_col_final = reason_col if reason_col != "" else None
analysis_col_final = analysis_col if analysis_col != "" else None
date_col_final = date_col if date_col != "" else None

# Warn if "Actual Issues" column is not mapped
if not analysis_col_final:
    st.warning("⚠️ 'Actual Issues' column not mapped. Error classification will be limited. Please select the 'Actual Issues' column in the sidebar for accurate Input vs Platform segregation.")

df = compute_metrics(df_raw, id_col, tries_col, status_col, reason_col_final, analysis_col_final, additional_cols, date_col_final)

# Show what columns are being used for classification
classification_sources = []
if analysis_col_final:
    classification_sources.append(f"🎯 **PRIMARY: analysis_text** (from '{analysis_col_final}' column)")
if reason_col_final:
    classification_sources.append(f"📝 **Secondary: reason_text** (from '{reason_col_final}' column)")
if additional_cols:
    classification_sources.append(f"➕ **Supplementary Context**: {', '.join(additional_cols)}")

if classification_sources:
    with st.expander("📊 How Input vs Platform Classification Works", expanded=False):
        st.markdown("**Input vs Platform Error Segregation:**")
        st.markdown("- **Input Errors**: Issues with data, specifications, or user-provided content")
        st.markdown("- **Platform Errors**: System, infrastructure, or technical issues")
        st.markdown("")
        st.markdown("**Classification is driven by:**")
        st.markdown(f"The `analysis_text` column (from **'{analysis_col_final}'** in your upload) is the **PRIMARY source** for categorizing errors.")
        st.markdown("")
        st.markdown("**Data sources used:**")
        for source in classification_sources:
            st.markdown(f"- {source}")
        st.caption("💡 The 'Actual Issues' column content determines Input vs Platform classification. 15+ detailed categories help identify root causes.")

# Calculate key metrics
success_df = df[df["is_success_or_partial"] == True].copy()
failed_df = df[df["is_success_or_partial"] == False].copy()

# Metric 1: # IPDs Created Successfully
ipds_successful = len(success_df)

# Metric 2: # IPDs Failed (segregated by Input vs Platform)
ipds_failed_total = len(failed_df)
ipds_failed_input = len(failed_df[failed_df["issue_bucket"] == "Input"])
ipds_failed_platform = len(failed_df[failed_df["issue_bucket"] == "System"])
ipds_failed_unknown = len(failed_df[failed_df["issue_bucket"] == "Unknown"])

# Metric 3: # Retries Required Before Success (average tries for successful IPDs)
avg_retries_success = success_df["tries"].mean() if not success_df.empty else 0.0

# Metric 4: Failed Customer Interactions = (failed tries / total tries)
# For failed IPDs, all tries are considered "failed tries"
failed_tries = failed_df["tries"].dropna().sum()
total_tries = df["tries"].dropna().sum()
failed_customer_interactions = (failed_tries / total_tries * 100) if total_tries > 0 else 0.0

# Display 4 Key Metrics
st.subheader("📊 Key Performance Metrics")

# Top row - main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("✅ IPDs Created Successfully", f"{ipds_successful:,}")
    st.caption("Successfully delivered campaigns")

with col2:
    st.metric("❌ IPDs Failed with Reason", f"{ipds_failed_total:,}")
    st.caption(f"🔴 Input: {ipds_failed_input} | 🔵 Platform: {ipds_failed_platform}")

with col3:
    st.metric("🔄 Avg Retries Before Success", f"{avg_retries_success:.1f}")
    st.caption("Average attempts for successful IPDs")

with col4:
    st.metric("⚠️ Failed Customer Interactions", f"{failed_customer_interactions:.1f}%")
    st.caption(f"{int(failed_tries):,} failed tries / {int(total_tries):,} total")

# Add breakdown info box
st.info(
    f"""
    📈 **Total IPDs**: {len(df):,} | ✅ **Success Rate**: {(ipds_successful/len(df)*100):.1f}% | 
    ❌ **Failure Breakdown**: Input Errors: {ipds_failed_input:,} ({(ipds_failed_input/ipds_failed_total*100) if ipds_failed_total > 0 else 0:.1f}%) | 
    Platform Errors: {ipds_failed_platform:,} ({(ipds_failed_platform/ipds_failed_total*100) if ipds_failed_total > 0 else 0:.1f}%) | 
    Unknown: {ipds_failed_unknown:,}
    """,
    icon="📊"
)

# Performance Trends Over Time
if date_col_final and "ipd_date" in df.columns and df["ipd_date"].notna().any():
    st.subheader("📈 Performance Trends Over Time")
    
    # Use ALL IPDs with valid dates
    trend_df = df[df["ipd_date"].notna()].copy()
    
    if not trend_df.empty:
        # Group by date and calculate daily metrics
        trend_df["date_only"] = trend_df["ipd_date"].dt.date
        
        daily_stats = []
        for date, group in trend_df.groupby("date_only"):
            success_count = len(group[group["is_success_or_partial"] == True])
            failed_count = len(group[group["is_success_or_partial"] == False])
            total_count = len(group)
            total_tries = group["tries"].dropna().sum()
            failed_tries = group[group["is_success_or_partial"] == False]["tries"].dropna().sum()
            
            # Failed Customer Interactions % for this day
            failed_customer_pct = (failed_tries / total_tries * 100) if total_tries > 0 else 0
            
            daily_stats.append({
                "Date": date,
                "Successful IPDs": success_count,
                "Failed IPDs": failed_count,
                "Failed Customer Interactions %": failed_customer_pct,
                "Total Tries": int(total_tries)
            })
        
        daily_metrics = pd.DataFrame(daily_stats)
        
        # Create dual-axis trend chart
        fig_trend = px.line(
            daily_metrics, 
            x="Date", 
            y="Failed Customer Interactions %",
            title="Failed Customer Interactions Trend by Date",
            markers=True,
            hover_data={"Successful IPDs": True, "Failed IPDs": True, "Total Tries": True, "Failed Customer Interactions %": ":.1f"}
        )
        
        fig_trend.update_traces(
            line_color='#FF6B6B',
            line_width=3,
            marker=dict(size=8, color='#4ECDC4', line=dict(width=2, color='white'))
        )
        
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Failed Customer Interactions %",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.caption("**Formula**: Failed Customer Interactions % = (failed tries on that day) / (total tries on that day) × 100")
        
        # Summary stats
        trend_col1, trend_col2, trend_col3 = st.columns(3)
        with trend_col1:
            st.metric("Best Day", f"{daily_metrics['Failed Customer Interactions %'].min():.1f}%", 
                     delta=None, help="Lowest failure rate")
        with trend_col2:
            st.metric("Worst Day", f"{daily_metrics['Failed Customer Interactions %'].max():.1f}%", 
                     delta=None, help="Highest failure rate")
        with trend_col3:
            st.metric("Avg Daily Failure Rate", f"{daily_metrics['Failed Customer Interactions %'].mean():.1f}%", 
                     delta=None, help="Average across all days")
    else:
        st.info("No valid date data available for trend analysis.")
else:
    st.info("💡 Select a date column in the sidebar to see performance trends over time.")

# Error Segregation & Distribution
st.subheader("🔍 Error Segregation: Input vs Platform (click slice to filter)")

bucket_mode = st.radio("View errors by", ["Input vs Platform vs Unknown", "Detailed Error Categories"], horizontal=True)

dim = "issue_bucket" if bucket_mode.startswith("Input vs Platform") else "issue_category"

# Pie counts only for rows that have issue text; if none, show Unknown
# Rename "System" to "Platform" for clearer labeling
pie_src = df.copy()
pie_src[dim] = pie_src[dim].fillna("Unknown")
pie_src[dim] = pie_src[dim].replace("System", "Platform")
pie_agg = pie_src.groupby(dim, as_index=False).size().rename(columns={"size": "count"})

# Custom color palette for better visual distinction
# Use specific colors for Input (red), Platform (blue), Unknown (gray)
if dim == "issue_bucket":
    color_map = {
        "Input": "#FF6B6B",      # Red for input errors
        "Platform": "#4ECDC4",   # Teal/blue for platform errors
        "Unknown": "#999999"      # Gray for unknown
    }
    fig = px.pie(pie_agg, names=dim, values="count", 
                 title="Error Distribution: Input vs Platform vs Unknown", 
                 hole=0.35,
                 color=dim,
                 color_discrete_map=color_map)
else:
    color_sequence = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
        '#FF9AA2', '#C7CEEA', '#FFDAC1', '#B5EAD7', '#E2F0CB'
    ]
    fig = px.pie(pie_agg, names=dim, values="count", 
                 title="Detailed Error Category Distribution", 
                 hole=0.35,
                 color_discrete_sequence=color_sequence)

clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=450)

selected_label = None
if clicked and len(clicked) > 0:
    selected_label = clicked[0].get("label")

if selected_label:
    st.success(f"✅ Filtered to: {selected_label} errors")
    # Need to handle the Platform -> System mapping for filtering
    filter_label = "System" if selected_label == "Platform" else selected_label
    df_filtered = df[df[dim].fillna("Unknown") == filter_label].copy()
else:
    st.info("💡 Click a pie slice to filter the IPD list below.")
    df_filtered = df.copy()

# IPD Details Table
st.subheader("📋 IPD Details Table")
st.caption("🎯 Error classification based on **analysis_text** (from 'Actual Issues' column) | Input vs Platform segregation")

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    show_only_failed = st.checkbox("Show only Failed IPDs", value=False)
with col_filter2:
    show_only_input_errors = st.checkbox("Show only Input errors", value=False)

if show_only_failed:
    df_filtered = df_filtered[df_filtered["is_success_or_partial"] == False].copy()
if show_only_input_errors:
    df_filtered = df_filtered[df_filtered["issue_bucket"] == "Input"].copy()

# Rename columns for better display
display_df = df_filtered.copy()
if "issue_bucket" in display_df.columns:
    display_df["error_type"] = display_df["issue_bucket"].replace("System", "Platform")
else:
    display_df["error_type"] = "Unknown"

table_cols = [
    "ipd_id", "ipd_date", "tries", "status_raw", "is_success_or_partial", 
    "error_type", "issue_category", "analysis_text", "reason_text", "TCI"
]
# Reorder to show analysis_text before reason_text since it's primary for classification
table_cols = [c for c in table_cols if c in display_df.columns and display_df[c].notna().any()]

# Sort: successful first, then by tries (more retries = more problematic)
display_df["success_sort"] = display_df["is_success_or_partial"].apply(lambda x: 0 if x is True else 1)
display_df = display_df.sort_values(["success_sort", "tries"], ascending=[True, False])

st.dataframe(display_df[table_cols], use_container_width=True, height=480)

# Export
st.subheader("📥 Export Data")
export_df = display_df[table_cols].drop(columns=["success_sort"], errors="ignore")
export_csv = export_df.to_csv(index=False).encode("utf-8")

col_exp1, col_exp2 = st.columns([3, 1])
with col_exp1:
    st.download_button(
        "📥 Download Filtered IPD Data (CSV)", 
        data=export_csv, 
        file_name="ipd_analysis_export.csv", 
        mime="text/csv",
        help="Download the currently filtered IPD data"
    )
with col_exp2:
    st.metric("Rows to Export", f"{len(export_df):,}")
