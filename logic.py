# logic.py
import io
import re
from typing import Optional, Tuple, List, Dict
import pandas as pd

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

def normalize_success_flag(status_val) -> Optional[bool]:
    if status_val is None or (isinstance(status_val, float) and pd.isna(status_val)):
        return None
    s = str(status_val).strip().lower()
    success_tokens = [
        "success", "succeeded", "pass", "passed",
        "partial success", "partial pass", "partially successful", "partial", "partial pass"
    ]
    fail_tokens = [
        "fail", "failed", "abandoned", "unfixable", "cancelled", "canceled"
    ]
    if any(t in s for t in success_tokens):
        return True
    if any(t in s for t in fail_tokens):
        return False
    return None

def classify_issue(text: str) -> Tuple[str, str]:
    msg = (text or "").lower()
    if not msg.strip():
        return ("Unknown", "Unknown")

    input_rules = [
        ("File & sheet structure", r"(sheet|tab|header|column|template|workbook|excel|xlsx|csv|format|structure)"),
        ("Required fields missing", r"(missing|required|empty|null|blank)"),
        ("Business-rule validation", r"(invalid|not allowed|out of range|frequency cap|budget|scheduled in the past|too long|too short)"),
        ("Referential integrity", r"(not found|doesn.t exist|does not exist|unknown id|asset not found|audience not found|catalog)"),
        ("Compliance & consent", r"(pii|consent|unsubscribe|legal|policy|prohibited|restricted)"),
        ("Channel/placement specs", r"(placement|aspect ratio|resolution|file size|duration|objective|optimization)"),
        ("Tracking & analytics", r"(pixel|tag|tracking|analytics|utm|event)"),
        ("Data quality & hygiene", r"(duplicate|duplicated|inconsistent|mismatch|stale|old version)"),
        ("Attachments & external references", r"(drive|link|url|permission|access denied|expired)"),
    ]
    for cat, pat in input_rules:
        if re.search(pat, msg):
            return ("Input", cat)

    system_rules = [
        ("Ingestion & parsing pipeline", r"(parser|parse|ingest|pipeline|schema|deserialize|serialization)"),
        ("Model/agent layer", r"(timeout|throttle|rate limit|5\d\d|model|llm|generation|invalid output)"),
        ("Storage & data services", r"(db|database|connection|pool|cache|blob|storage)"),
        ("Auth", r"(oauth|token|auth|permission|unauthorized|forbidden|403|401)"),
        ("Integrations", r"(api outage|integration|downstream|third party|webhook)"),
        ("Orchestration", r"(worker|queue|stuck|pending|job failed|orchestration|dispatch)"),
        ("UX/observability", r"(unknown error|generic error|no details|telemetry loss|retry later)"),
    ]
    for cat, pat in system_rules:
        if re.search(pat, msg):
            return ("System", cat)

    return ("Unknown", "Unknown")


def _read_uploaded_bytes(file_name: str, file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    lower = file_name.lower()
    if lower.endswith(".parquet"):
        import pyarrow.parquet as pq, pyarrow as pa
        table = pq.read_table(pa.BufferReader(file_bytes))
        return table.to_pandas()
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        bio = io.BytesIO(file_bytes)
        return pd.read_excel(bio, sheet_name=sheet_name)
    bio = io.BytesIO(file_bytes)
    return pd.read_csv(bio, engine="python")


def compute_metrics_from_file(file_name: str, file_bytes: bytes, sheet_name: Optional[str],
                              id_col: str, tries_col: str, status_col: str, reason_col: Optional[str] = None) -> Dict:
    """
    Returns dict shaped for Builder.io consumption.
    - id_col, tries_col, status_col, reason_col should match column names in the uploaded file.
    """
    df = _read_uploaded_bytes(file_name, file_bytes, sheet_name=sheet_name)
    # Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]

    # Validate required columns
    missing = [c for c in [id_col, tries_col, status_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in uploaded file: {missing}")

    out = df.copy()
    out = out[[id_col, tries_col, status_col] + ([reason_col] if reason_col and reason_col in out.columns else [])].copy()
    out = out.rename(columns={id_col: "campaign_id", tries_col: "tries_raw", status_col: "status_raw"})
    if reason_col and reason_col in out.columns:
        out = out.rename(columns={reason_col: "reason_text"})
    else:
        out["reason_text"] = None

    out["tries"] = out["tries_raw"].apply(to_int_safe)
    out["is_success_or_partial"] = out["status_raw"].apply(normalize_success_flag)
    # per-campaign TCI = (1 / tries) * 100 for successful/partial success only
    out["TCI"] = out.apply(lambda r: (100.0 / r["tries"]) if (r["is_success_or_partial"] is True and r["tries"] and r["tries"] > 0) else None, axis=1)

    # Issue classification
    buckets, cats = [], []
    for t in out["reason_text"].fillna("").astype(str).tolist():
        b, c = classify_issue(t)
        buckets.append(b)
        cats.append(c)
    out["issue_bucket"] = buckets
    out["issue_category"] = cats

    # Rollup values for response
    snapshot = out[["campaign_id", "tries", "status_raw", "is_success_or_partial", "TCI", "issue_bucket", "issue_category", "reason_text"]].copy()

    succeeded = snapshot[snapshot["is_success_or_partial"] == True].copy()
    num_success = int(len(succeeded))
    denom = float(succeeded["tries"].dropna().sum()) if num_success > 0 else 0.0
    product_tci = (num_success / denom * 100.0) if denom > 0 else 0.0
    avg_campaign_tci = float(succeeded["TCI"].dropna().mean()) if not succeeded.empty else 0.0

    # issue buckets summary
    bucket_summary = (
        out.groupby("issue_bucket")
           .size()
           .reset_index(name="count")
           .to_dict(orient="records")
    )

    # prepare campaigns list for JSON
    campaigns_json = snapshot.fillna("").to_dict(orient="records")

    resp = {
        "productTCI": product_tci,
        "avgCampaignTCI": avg_campaign_tci,
        "totalCampaigns": int(len(out)),
        "successfulCampaigns": num_success,
        "campaigns": campaigns_json,
        "issueBuckets": bucket_summary,
    }
    return resp
