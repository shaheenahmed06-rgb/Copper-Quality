# 🔵 Copper Wire Rod — Process Quality Analysis Agent

Automated SPC and anomaly detection agent for continuous casting lines (Conform / Properzi-type).  
Runs as a **GitHub Actions** workflow or standalone Python script.

---

## 📁 Repository Structure

```
├── analyze.py                          # Main analysis agent
├── generate_sample_data.py             # Optional: generate synthetic test data
├── requirements.txt
├── data/
│   ├── spec_limits.csv                 # (Optional) Override default spec limits
│   └── *.csv / *.xlsx                  # ← Place your historian/SCADA exports here
├── reports/
│   ├── quality_report.md               # Generated Markdown report
│   ├── quality_report.xlsx             # Generated Excel summary
│   └── agent_log.txt                   # Run log
└── .github/
    └── workflows/
        └── quality_analysis.yml        # GitHub Actions workflow
```

---

## 🚀 Quick Start

### Option A — GitHub Actions (Recommended)

1. **Fork / clone** this repository.
2. Drop your `.csv` or `.xlsx` historian exports into the `data/` folder.
3. Push to `main` — the workflow triggers automatically on any `data/**` change.
4. Find the generated report under **Actions → Run → Artifacts** (`quality-reports-N`).
5. The report is also printed directly in the **Job Summary** tab.

### Option B — Local Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Generate synthetic test data
python generate_sample_data.py

# 3. Run agent
python analyze.py

# 4. View output
cat reports/quality_report.md
```

---

## ⚙️ Configuration

### Spec Limits Override

Edit `data/spec_limits.csv` to override any default industry benchmark:

| parameter | LSL | USL | unit | reference |
|-----------|-----|-----|------|-----------|
| furnace_temp | 1085 | 1130 | degC | IS 613 |
| rod_diameter | 7.85 | 8.15 | mm | Plant spec |

Column matching is **keyword-based** (case-insensitive substring).  
E.g., a column named `"Furnace Temperature (°C)"` will match spec key `furnace_temp`.

---

## 📊 Output Report Sections

| Section | Contents |
|---------|----------|
| 1. Data Health | Missing %, outlier counts, frozen sensor flags |
| 2. Statistical Summary | Mean, Std Dev, Cp/Cpk, trend direction per parameter |
| 3. Quality Risk Highlights | 🔴 Critical / 🟡 Warning / 🟢 OK classification |
| 4. Recommended Actions | Prioritized, parameter-specific action items |
| 5. Appendix | Outlier detail table, dataset coverage confirmation |

---

## 🔬 Default Spec Limits (IS / IEC / ASTM)

| Parameter | LSL | USL | Standard |
|-----------|-----|-----|----------|
| Furnace / Melt Temp | 1085 °C | 1130 °C | IS 613, ASTM B49 |
| Casting Speed | 5.0 m/min | 12.0 m/min | OEM process spec |
| Rod Exit Temperature | 400 °C | 700 °C | Conform/Properzi |
| Rod Diameter (8 mm) | 7.8 mm | 8.2 mm | IS 613 |
| Oxygen Content | 0 ppm | 5 ppm | IEC 60028 / IS 613 |
| Conductivity | 100 %IACS | 101.5 %IACS | IEC 60028 (ETP) |
| Tensile Strength | 195 MPa | 250 MPa | IS 613 / ASTM B49 |
| Elongation | 25 % | 40 % | IS 613 / ASTM B49 |

---

## 🛠️ Dependencies

- `pandas` ≥ 2.0
- `numpy` ≥ 1.24
- `scipy` ≥ 1.11
- `openpyxl` ≥ 3.1
- `tabulate` ≥ 0.9
- `xlrd` ≥ 2.0

---

## 📋 Supported Input Formats

- **CSV**: Auto-detects delimiter (`,`, `;`, `\t`)
- **XLSX / XLS**: All sheets (first sheet used per file)
- **Multiple files**: Auto-merged and aligned on timestamp
- **Timestamp columns**: Auto-detected by name (`time`, `date`, `timestamp`, `ts`) or dtype

---

## 🔄 Triggering Manually via GitHub UI

1. Go to **Actions** tab → `Copper Wire Rod Quality Analysis`
2. Click **Run workflow** → provide optional description → **Run**

---

## 📝 Notes

- Agent logs all steps to `reports/agent_log.txt` — safe to inspect after CI run.
- Agent **never crashes silently** — exceptions are caught, logged, and exit codes propagate correctly to GitHub Actions.
- Cpk < 1.0 → 🔴 Critical | Cpk 1.0–1.33 → 🟡 Warning | Cpk ≥ 1.33 → 🟢 World-class
