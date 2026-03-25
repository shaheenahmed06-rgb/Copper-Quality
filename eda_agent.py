"""
eda_agent.py  — AI EDA Agent powered by Claude (Anthropic)
------------------------------------------------------------
Uses Claude claude-sonnet-4-20250514 to write narrative EDA reports.

How it works:
  1. All statistics are computed deterministically with Pandas / NumPy.
  2. The full stats JSON is sent to Claude which writes a structured report.
  3. Plotly generates interactive charts automatically.

Free tier: use your Anthropic API key from https://console.anthropic.com
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import anthropic
from typing import Optional


class EDAAgent:
    """
    Autonomous EDA Agent powered by Claude (Anthropic).

    Parameters
    ----------
    api_key : str   Your Anthropic API key from https://console.anthropic.com
    model   : str   Claude model name (default: claude-sonnet-4-20250514)
    """

    SYSTEM_PROMPT = (
        "You are a senior data scientist specialising in exploratory data analysis. "
        "When given dataset statistics, you write clear, structured, insightful reports "
        "in Markdown. Always be specific with numbers and percentages. "
        "Format your report with proper Markdown headers (##), bullet points, and bold text. "
        "Be concise but comprehensive — every section should add actionable value."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client     = anthropic.Anthropic(api_key=api_key)
        self.model_name = model

    # ------------------------------------------------------------------
    # Step 1 — Compute all stats with Pandas (deterministic, no AI)
    # ------------------------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        stats: dict = {}

        # ── Basic overview ──────────────────────────────────────────────
        stats["shape"]        = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        stats["column_names"] = df.columns.tolist()
        stats["dtypes"]       = df.dtypes.astype(str).to_dict()
        stats["sample_rows"]  = df.head(3).to_dict(orient="records")

        # ── Data quality ────────────────────────────────────────────────
        missing = df.isnull().sum()
        stats["missing_values"] = {
            col: {
                "count":      int(missing[col]),
                "percentage": round(float(missing[col] / len(df) * 100), 2),
            }
            for col in df.columns
        }
        stats["duplicate_rows"] = {
            "count":      int(df.duplicated().sum()),
            "percentage": round(float(df.duplicated().sum() / len(df) * 100), 2),
        }

        # ── Numeric analysis ────────────────────────────────────────────
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            desc = numeric_df.describe().round(4)
            stats["numeric_summary"] = desc.to_dict()
            stats["skewness"]        = numeric_df.skew().round(4).to_dict()
            stats["kurtosis"]        = numeric_df.kurtosis().round(4).to_dict()

            # Correlations
            if len(numeric_df.columns) > 1:
                stats["correlations"] = numeric_df.corr().round(3).to_dict()

            # Outliers — IQR method
            outliers: dict = {}
            for col in numeric_df.columns:
                s   = numeric_df[col].dropna()
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                iqr = q3 - q1
                n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                outliers[col] = {
                    "count":       n_out,
                    "percentage":  round(n_out / len(df) * 100, 2),
                    "lower_fence": round(q1 - 1.5 * iqr, 4),
                    "upper_fence": round(q3 + 1.5 * iqr, 4),
                }
            stats["outliers"] = outliers

        # ── Categorical analysis ────────────────────────────────────────
        cat_df = df.select_dtypes(include=["object", "category"])
        if not cat_df.empty:
            cat_stats: dict = {}
            for col in cat_df.columns:
                vc = df[col].value_counts().head(10)
                cat_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_10_values": {str(k): int(v) for k, v in vc.items()},
                }
            stats["categorical"] = cat_stats

        return stats

    # ------------------------------------------------------------------
    # Step 2 — Ask Claude to write the narrative report
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        stats: dict,
        filename: str,
        user_question: str,
    ) -> str:
        prompt = f"""
I have a dataset called **{filename}** with {stats['shape']['rows']:,} rows and {stats['shape']['columns']} columns.

Here are the fully computed statistics:

```json
{json.dumps(stats, default=str, indent=2)}
```

Please write a **comprehensive EDA report** in Markdown with the following sections:

## 1. Executive Summary
A 3-4 sentence summary of the most important findings.

## 2. Dataset Overview
Shape, column types, what the data appears to represent.

## 3. Data Quality Report
Missing values, duplicate rows — which columns need attention and why.

## 4. Statistical Summary
Key statistics for numeric columns. Highlight unusual values.

## 5. Distribution Analysis
Comment on skewness and kurtosis. Which columns are skewed or have heavy tails?

## 6. Outlier Analysis
Which columns have significant outliers? How many and what percentage?

## 7. Correlation Insights
Which variables are strongly correlated (positive or negative)?

## 8. Categorical Insights
Top values and patterns in categorical columns.

## 9. Key Findings & Recommendations
Bullet points of the 5 most important findings.
Concrete recommendations for data cleaning or further analysis.

{f'**Specific question to address:** {user_question}' if user_question.strip() else ''}

Be specific — use the exact numbers and percentages from the statistics provided.
"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    # ------------------------------------------------------------------
    # Step 3 — Generate Plotly charts (same as EDAEngine)
    # ------------------------------------------------------------------

    def _generate_charts(self, df: pd.DataFrame) -> list:
        charts      = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Histograms (max 6 numeric columns)
        for col in numeric_cols[:6]:
            fig = px.histogram(
                df, x=col,
                title=f"Distribution: {col}",
                nbins=40,
                template="plotly_white",
                marginal="box",
                color_discrete_sequence=["#667eea"],
            )
            charts.append({"title": f"Distribution: {col}", "fig": fig})

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig  = px.imshow(
                corr,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                template="plotly_white",
                aspect="auto",
                text_auto=".2f",
            )
            charts.append({"title": "Correlation Heatmap", "fig": fig})

        # Box plots (outlier overview)
        if numeric_cols:
            fig = px.box(
                df[numeric_cols[:5]],
                title="Box Plots — Outlier Overview",
                template="plotly_white",
                color_discrete_sequence=["#764ba2"],
            )
            charts.append({"title": "Box Plots", "fig": fig})

        # Bar charts for categorical columns (max 3)
        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(15).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(
                vc, x=col, y="count",
                title=f"Value Counts: {col}",
                template="plotly_white",
                color_discrete_sequence=["#48bb78"],
            )
            charts.append({"title": f"Value Counts: {col}", "fig": fig})

        # Scatter matrix (2–6 numeric cols)
        if 2 <= len(numeric_cols) <= 6:
            fig = px.scatter_matrix(
                df[numeric_cols],
                title="Scatter Matrix",
                template="plotly_white",
                opacity=0.5,
            )
            charts.append({"title": "Scatter Matrix", "fig": fig})

        return charts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        user_question: str = "",
        filename: str = "dataset",
        progress_callback=None,
    ) -> dict:
        """
        Run the full AI-powered EDA pipeline.

        Returns dict with:
          - report : str   (Markdown narrative written by Claude)
          - charts : list  (Plotly figure dicts)
        """
        if progress_callback:
            progress_callback("🔢 Computing statistics with Pandas...")

        stats = self._compute_stats(df)

        if progress_callback:
            progress_callback("🤖 Sending stats to Claude for AI analysis...")

        report = self._generate_report(stats, filename, user_question)

        if progress_callback:
            progress_callback("📊 Generating visualisations...")

        charts = self._generate_charts(df)

        return {"report": report, "charts": charts}
