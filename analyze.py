"""
Copper Wire Rod — Process Quality Analysis Agent
================================================
Runs via GitHub Actions. Reads CSVs/XLSXs from /data, writes report to /reports.
"""

import os
import sys
import glob
import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_FILE = Path("reports/agent_log.txt")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Industry spec limits (IS/IEC/ASTM benchmarks) ──────────────────────────────
# Format: parameter_keyword -> (LSL, USL)
# Extend/override via data/spec_limits.csv if present
DEFAULT_SPECS = {
    "furnace_temp":        (1085, 1130),   # °C  — copper melt
    "melt_temp":           (1085, 1130),
    "casting_speed":       (5.0,  12.0),   # m/min
    "rod_exit_temp":       (400,   700),   # °C
    "cooling_water_flow":  (20,    80),    # m³/h
    "cooling_water_temp":  (15,    40),    # °C
    "rod_diameter":        (7.8,   8.2),   # mm  (8 mm rod)
    "oxygen_content":      (0,     5),     # ppm
    "conductivity":        (100,  101.5),  # %IACS
    "tensile_strength":    (195,  250),    # MPa
    "elongation":          (25,    40),    # %
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_spec_overrides(data_dir: Path) -> dict:
    spec_file = data_dir / "spec_limits.csv"
    specs = dict(DEFAULT_SPECS)
    if spec_file.exists():
        try:
            df = pd.read_csv(spec_file)
            for _, row in df.iterrows():
                key = str(row.get("parameter", "")).strip().lower()
                lsl = float(row.get("LSL", np.nan))
                usl = float(row.get("USL", np.nan))
                if key and not np.isnan(lsl) and not np.isnan(usl):
                    specs[key] = (lsl, usl)
            log.info(f"Loaded {len(df)} spec overrides from spec_limits.csv")
        except Exception as e:
            log.warning(f"Could not load spec_limits.csv: {e}")
    return specs


def find_spec(col: str, specs: dict):
    col_lower = col.lower().replace(" ", "_")
    for key, limits in specs.items():
        if key in col_lower:
            return limits
    return None


def load_file(fp: Path) -> pd.DataFrame:
    suffix = fp.suffix.lower()
    if suffix == ".csv":
        # Try comma first, then semicolon/tab
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(fp, sep=sep)
                if df.shape[1] > 1:
                    log.info(f"Loaded {fp.name} with sep='{sep}' → {df.shape}")
                    return df
            except Exception:
                pass
        raise ValueError(f"Could not parse {fp.name} as CSV")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(fp, parse_dates=True)
        log.info(f"Loaded {fp.name} → {df.shape}")
        return df
    else:
        raise ValueError(f"Unsupported file type: {fp.suffix}")


def detect_timestamp_col(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if any(k in col.lower() for k in ["time", "date", "timestamp", "ts"]):
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None


def merge_files(dfs: list, filenames: list) -> pd.DataFrame:
    if len(dfs) == 1:
        return dfs[0]
    merged = pd.concat(dfs, ignore_index=True)
    ts_col = detect_timestamp_col(merged)
    if ts_col:
        merged[ts_col] = pd.to_datetime(merged[ts_col], errors="coerce")
        # Check overlaps / gaps
        merged.sort_values(ts_col, inplace=True)
        merged.reset_index(drop=True, inplace=True)
    log.info(f"Merged {len(dfs)} files → {merged.shape[0]} rows")
    return merged


# ── Analysis functions ─────────────────────────────────────────────────────────

def audit_data_quality(df: pd.DataFrame, numeric_cols: list):
    results = []
    ts_col = detect_timestamp_col(df)

    for col in numeric_cols:
        series = df[col].copy()
        total = len(series)
        missing = series.isna().sum()
        missing_pct = 100.0 * missing / total if total else 0
        clean = series.dropna()

        # IQR outliers
        if len(clean) >= 4:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            iqr_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
            iqr_outliers = int(iqr_mask.sum())
        else:
            iqr_outliers = 0

        # Z-score outliers
        if len(clean) >= 4:
            z = np.abs(stats.zscore(clean))
            z_outliers = int((z > 3).sum())
        else:
            z_outliers = 0

        # Frozen sensor (>5 identical consecutive readings)
        rle = (series != series.shift()).cumsum()
        max_run = rle.value_counts().max() if len(series) > 0 else 0
        frozen = int(max_run > 5)

        # Status
        flags = []
        if missing_pct > 5:
            flags.append("HIGH_MISSING")
        if iqr_outliers > 0 or z_outliers > 0:
            flags.append("OUTLIERS")
        if frozen:
            flags.append("FROZEN_SENSOR")
        status = "🔴 CRITICAL" if "HIGH_MISSING" in flags or "FROZEN_SENSOR" in flags else \
                 "🟡 WARNING" if flags else "🟢 OK"

        results.append({
            "Parameter":    col,
            "Total Rows":   total,
            "Missing (%)":  f"{missing_pct:.1f}%",
            "Outliers IQR": iqr_outliers,
            "Outliers Z":   z_outliers,
            "Frozen":       "YES ⚠" if frozen else "No",
            "Status":       status,
        })

    # Timestamp checks
    ts_notes = []
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
        diffs = ts.diff().dropna()
        if not diffs.empty:
            mode_interval = diffs.mode()[0]
            irregular = int((diffs != mode_interval).sum())
            if irregular:
                ts_notes.append(f"⚠ {irregular} irregular sampling intervals (mode={mode_interval})")
        non_mono = int((ts.diff().dropna() < pd.Timedelta(0)).sum())
        if non_mono:
            ts_notes.append(f"🔴 {non_mono} non-monotonic (backward) timestamps")

    return results, ts_notes


def compute_cpk(series: pd.Series, lsl, usl):
    clean = series.dropna()
    if len(clean) < 10:
        return np.nan, np.nan
    mu, sigma = clean.mean(), clean.std()
    if sigma == 0:
        return np.nan, np.nan
    cpu = (usl - mu) / (3 * sigma)
    cpl = (mu - lsl) / (3 * sigma)
    cp  = (usl - lsl) / (6 * sigma)
    cpk = min(cpu, cpl)
    return round(cp, 3), round(cpk, 3)


def trend_direction(series: pd.Series) -> str:
    clean = series.dropna()
    if len(clean) < 10:
        return "Insufficient data"
    x = np.arange(len(clean))
    slope, _, r, _, _ = stats.linregress(x, clean.values)
    if abs(r) < 0.3:
        return "Stable"
    threshold = clean.std() * 0.05
    if slope > threshold:
        return "⬆ Drifting Up"
    elif slope < -threshold:
        return "⬇ Drifting Down"
    return "Stable"


def statistical_summary(df: pd.DataFrame, numeric_cols: list, specs: dict):
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        spec = find_spec(col, specs)
        cp, cpk = compute_cpk(series, *spec) if spec else (np.nan, np.nan)
        risk = ""
        if not np.isnan(cpk) if isinstance(cpk, float) else False:
            if cpk < 1.0:
                risk = "🔴 CRITICAL"
            elif cpk < 1.33:
                risk = "🟡 WARNING"
            else:
                risk = "🟢 OK"

        results.append({
            "Parameter":  col,
            "Mean":       round(series.mean(), 3),
            "Std Dev":    round(series.std(), 3),
            "Min":        round(series.min(), 3),
            "Max":        round(series.max(), 3),
            "Cp":         cp if not (isinstance(cp, float) and np.isnan(cp)) else "—",
            "Cpk":        cpk if not (isinstance(cpk, float) and np.isnan(cpk)) else "—",
            "Trend":      trend_direction(series),
            "Risk Flag":  risk,
        })
    return results


def collect_outlier_detail(df: pd.DataFrame, numeric_cols: list):
    ts_col = detect_timestamp_col(df)
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        outlier_rows = df[mask]
        for idx, row in outlier_rows.iterrows():
            ts = row[ts_col] if ts_col else idx
            val = row[col]
            dev = val - series.mean()
            rows.append({
                "Timestamp":  ts,
                "Parameter":  col,
                "Value":      round(val, 4),
                "Deviation":  round(dev, 4),
                "Type":       "IQR",
            })
    # Limit to 100 rows in appendix
    return rows[:100]


# ── Report builder ─────────────────────────────────────────────────────────────

def build_markdown_report(
    filenames, date_range, dq_results, ts_notes,
    stat_results, outlier_detail, total_rows
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("# Copper Wire Rod — Process Quality Report")
    lines.append(f"**Generated:** {now}  |  **Files Analyzed:** {', '.join(filenames)}  |  **Period Covered:** {date_range}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1 ──
    lines.append("## 1. 📋 Data Health Summary")
    if ts_notes:
        lines.append("")
        lines.append("**⚠ Timestamp Integrity Notes:**")
        for note in ts_notes:
            lines.append(f"- {note}")
    lines.append("")
    dq_headers = ["Parameter","Total Rows","Missing (%)","Outliers IQR","Outliers Z","Frozen","Status"]
    dq_rows = [[r[h] for h in dq_headers] for r in dq_results]
    lines.append(tabulate(dq_rows, headers=dq_headers, tablefmt="github"))
    lines.append("")

    # ── Section 2 ──
    lines.append("## 2. 📊 Statistical Summary")
    lines.append("")
    st_headers = ["Parameter","Mean","Std Dev","Min","Max","Cp","Cpk","Trend","Risk Flag"]
    st_rows = [[r[h] for h in st_headers] for r in stat_results]
    lines.append(tabulate(st_rows, headers=st_headers, tablefmt="github"))
    lines.append("")

    # ── Section 3 ──
    lines.append("## 3. 🚨 Quality Risk Highlights")
    lines.append("")
    critical = [r["Parameter"] for r in stat_results if r["Risk Flag"] == "🔴 CRITICAL"]
    critical += [r["Parameter"] for r in dq_results if "CRITICAL" in r["Status"]]
    critical = list(dict.fromkeys(critical))

    warning = [r["Parameter"] for r in stat_results if r["Risk Flag"] == "🟡 WARNING"]
    warning += [r["Parameter"] for r in dq_results if "WARNING" in r["Status"]]
    warning = list(dict.fromkeys(warning))

    ok = [r["Parameter"] for r in stat_results if r["Risk Flag"] == "🟢 OK"]

    lines.append(f"- 🔴 **Critical:** {', '.join(critical) if critical else 'None'}")
    lines.append(f"- 🟡 **Warning:**  {', '.join(warning) if warning else 'None'}")
    lines.append(f"- 🟢 **OK:**       {', '.join(ok) if ok else 'None'}")
    lines.append("")

    # ── Section 4 ──
    lines.append("## 4. ✅ Recommended Actions")
    lines.append("")
    actions = []
    for r in stat_results:
        p, cpk, trend = r["Parameter"], r["Cpk"], r["Trend"]
        if r["Risk Flag"] == "🔴 CRITICAL":
            actions.append(f"**Investigate & correct** `{p}` immediately — Cpk={cpk} (<1.0 indicates process incapability) — 🔴 Urgent")
        elif r["Risk Flag"] == "🟡 WARNING":
            actions.append(f"**Monitor closely** `{p}` — Cpk={cpk} (<1.33 is below world-class threshold) — 🟡 Within 24 hrs")
        if "Up" in trend:
            actions.append(f"**Check for upward drift** in `{p}` — review process settings or sensor calibration — 🟡 Monitor")
        elif "Down" in trend:
            actions.append(f"**Check for downward drift** in `{p}` — possible equipment wear or raw material shift — 🟡 Monitor")
    for r in dq_results:
        if "HIGH_MISSING" in r["Missing (%)"]:
            pass
        if "FROZEN" in r["Frozen"]:
            actions.append(f"**Verify sensor / instrument** for `{r['Parameter']}` — frozen readings detected — 🔴 Urgent")
        if float(r["Missing (%)"].replace("%","")) > 5:
            actions.append(f"**Investigate data gaps** for `{r['Parameter']}` — {r['Missing (%)']} missing — 🟡 Check historian/SCADA")

    if not actions:
        actions.append("No critical actions required. Continue routine SPC monitoring.")

    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a}")
    lines.append("")

    # ── Section 5 ──
    lines.append("## 5. 📁 Appendix")
    lines.append("")
    lines.append(f"**Merged Dataset:** {total_rows} total rows | **Period:** {date_range}")
    lines.append("")
    if outlier_detail:
        lines.append("### Outlier Detail Table (IQR-based, max 100 records)")
        od_headers = ["Timestamp","Parameter","Value","Deviation","Type"]
        od_rows = [[r[h] for h in od_headers] for r in outlier_detail]
        lines.append(tabulate(od_rows, headers=od_headers, tablefmt="github"))
    else:
        lines.append("_No outliers detected._")
    lines.append("")

    return "\n".join(lines)


def export_xlsx(stat_results, dq_results, outlier_detail, out_path: Path):
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(stat_results).to_excel(writer, sheet_name="Statistical Summary", index=False)
            pd.DataFrame(dq_results).to_excel(writer, sheet_name="Data Health", index=False)
            if outlier_detail:
                pd.DataFrame(outlier_detail).to_excel(writer, sheet_name="Outlier Detail", index=False)
        log.info(f"XLSX report written to {out_path}")
    except Exception as e:
        log.warning(f"XLSX export failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path("data")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Locate input files
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    input_files = []
    for pat in patterns:
        input_files.extend(data_dir.glob(pat))
    # Exclude spec_limits.csv
    input_files = [f for f in input_files if f.name.lower() != "spec_limits.csv"]

    if not input_files:
        log.error("No CSV/XLSX files found in /data directory. Exiting.")
        sys.exit(1)

    log.info(f"Found {len(input_files)} input file(s): {[f.name for f in input_files]}")

    # Load files
    dfs = []
    for fp in input_files:
        try:
            dfs.append(load_file(fp))
        except Exception as e:
            log.error(f"Failed to load {fp.name}: {e}\n{traceback.format_exc()}")

    if not dfs:
        log.error("All file loads failed. Exiting.")
        sys.exit(1)

    # Merge
    merged = merge_files(dfs, [f.name for f in input_files])
    total_rows = len(merged)

    # Date range
    ts_col = detect_timestamp_col(merged)
    if ts_col:
        merged[ts_col] = pd.to_datetime(merged[ts_col], errors="coerce")
        ts_valid = merged[ts_col].dropna()
        date_range = f"{ts_valid.min().date()} to {ts_valid.max().date()}" if len(ts_valid) else "Unknown"
    else:
        date_range = "Unknown (no timestamp column detected)"

    # Numeric columns only
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        log.error("No numeric columns found in merged data. Exiting.")
        sys.exit(1)
    log.info(f"Numeric parameters detected: {numeric_cols}")

    # Load specs
    specs = load_spec_overrides(data_dir)

    # Analysis
    log.info("Running data quality audit...")
    dq_results, ts_notes = audit_data_quality(merged, numeric_cols)

    log.info("Computing statistical summary...")
    stat_results = statistical_summary(merged, numeric_cols, specs)

    log.info("Collecting outlier details...")
    outlier_detail = collect_outlier_detail(merged, numeric_cols)

    # Build report
    log.info("Building Markdown report...")
    filenames = [f.name for f in input_files]
    report_md = build_markdown_report(
        filenames, date_range, dq_results, ts_notes,
        stat_results, outlier_detail, total_rows
    )

    # Write Markdown
    md_path = reports_dir / "quality_report.md"
    md_path.write_text(report_md, encoding="utf-8")
    log.info(f"Markdown report saved → {md_path}")

    # Write XLSX
    xlsx_path = reports_dir / "quality_report.xlsx"
    export_xlsx(stat_results, dq_results, outlier_detail, xlsx_path)

    log.info("✅ Agent completed successfully.")
    print("\n" + report_md)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        sys.exit(2)
