"""
eda_engine.py
-------------
100% free EDA engine — no AI API needed, no keys, no limits.
Uses Pandas + NumPy to compute stats and generate smart insights.
"""

import numpy as np
import pandas as pd
import plotly.express as px
from typing import Optional


class EDAEngine:
    """Pure Python EDA engine. No AI API. No cost. No limits."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _overview(self, df: pd.DataFrame) -> dict:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_cells": int(df.shape[0] * df.shape[1]),
            "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
            "categorical_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime_cols": df.select_dtypes(include="datetime").columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

    def _missing(self, df: pd.DataFrame) -> dict:
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        return {
            col: {"count": int(missing[col]), "pct": float(pct[col])}
            for col in df.columns
            if missing[col] > 0
        }

    def _duplicates(self, df: pd.DataFrame) -> dict:
        n = int(df.duplicated().sum())
        return {"count": n, "pct": round(n / len(df) * 100, 2)}

    def _numeric_stats(self, df: pd.DataFrame) -> dict:
        num = df.select_dtypes(include="number")
        if num.empty:
            return {}
        stats = {}
        for col in num.columns:
            s = num[col].dropna()
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            stats[col] = {
                "mean":     round(float(s.mean()), 4),
                "median":   round(float(s.median()), 4),
                "std":      round(float(s.std()), 4),
                "min":      round(float(s.min()), 4),
                "max":      round(float(s.max()), 4),
                "q25":      round(q1, 4),
                "q75":      round(q3, 4),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
                "outliers": n_out,
                "outlier_pct": round(n_out / len(df) * 100, 2),
                "missing":  int(num[col].isnull().sum()),
            }
        return stats

    def _correlations(self, df: pd.DataFrame) -> dict:
        num = df.select_dtypes(include="number")
        if len(num.columns) < 2:
            return {}
        corr = num.corr()
        # Find strong pairs
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = round(float(corr.iloc[i, j]), 3)
                if abs(r) >= 0.5:
                    pairs.append({"col_a": cols[i], "col_b": cols[j], "r": r})
        pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        return {"matrix": corr.round(3).to_dict(), "strong_pairs": pairs}

    def _categorical_stats(self, df: pd.DataFrame) -> dict:
        cat = df.select_dtypes(include=["object", "category"])
        if cat.empty:
            return {}
        stats = {}
        for col in cat.columns:
            vc = df[col].value_counts()
            stats[col] = {
                "unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "missing": int(df[col].isnull().sum()),
            }
        return stats

    # ------------------------------------------------------------------
    # Smart rule-based insights (replaces AI)
    # ------------------------------------------------------------------

    def _generate_insights(
        self,
        overview: dict,
        missing: dict,
        duplicates: dict,
        numeric: dict,
        correlations: dict,
        categorical: dict,
        user_question: str,
    ) -> str:
        lines = []
        a = lines.append

        a("# 📊 EDA Report\n")

        # ── Executive Summary ─────────────────────────────────────────
        a("## 🔍 Executive Summary\n")
        a(f"- Dataset has **{overview['rows']:,} rows** and **{overview['columns']} columns**")
        a(f"- **{len(overview['numeric_cols'])} numeric** and **{len(overview['categorical_cols'])} categorical** columns")
        total_missing = sum(v["count"] for v in missing.values())
        total_cells = overview["total_cells"]
        a(f"- **{total_missing:,} missing values** ({round(total_missing/total_cells*100,1)}% of all cells)")
        if duplicates["count"] > 0:
            a(f"- ⚠️ **{duplicates['count']:,} duplicate rows** ({duplicates['pct']}%) detected")
        else:
            a("- ✅ No duplicate rows found")
        a("")

        # ── Data Quality ──────────────────────────────────────────────
        a("## 🧹 Data Quality\n")
        if not missing:
            a("✅ **No missing values** — dataset is complete!\n")
        else:
            a("### Missing Values")
            a("| Column | Missing Count | % Missing | Severity |")
            a("|--------|-------------|-----------|----------|")
            for col, info in sorted(missing.items(), key=lambda x: -x[1]["pct"]):
                pct = info["pct"]
                sev = "🔴 High" if pct > 30 else ("🟡 Medium" if pct > 5 else "🟢 Low")
                a(f"| {col} | {info['count']:,} | {pct}% | {sev} |")
            a("")
            # Recommendations
            high = [c for c, v in missing.items() if v["pct"] > 50]
            med  = [c for c, v in missing.items() if 5 < v["pct"] <= 50]
            low  = [c for c, v in missing.items() if v["pct"] <= 5]
            if high:
                a(f"⚠️ **Consider dropping:** {', '.join(high)} (>50% missing)")
            if med:
                a(f"🔧 **Consider imputing:** {', '.join(med)} (5–50% missing)")
            if low:
                a(f"✅ **Minor gaps:** {', '.join(low)} (<5% missing) — safe to impute")
            a("")

        if duplicates["count"] > 0:
            a(f"### Duplicate Rows\n⚠️ Found **{duplicates['count']:,} duplicate rows** ({duplicates['pct']}%). Consider removing with `df.drop_duplicates()`.\n")

        # ── Numeric Analysis ──────────────────────────────────────────
        if numeric:
            a("## 📈 Numeric Columns Analysis\n")
            a("| Column | Mean | Median | Std | Skewness | Outliers |")
            a("|--------|------|--------|-----|----------|----------|")
            for col, s in numeric.items():
                skew_flag = "⚠️ " if abs(s["skewness"]) > 1 else ""
                out_flag  = "⚠️ " if s["outlier_pct"] > 5 else ""
                a(f"| {col} | {s['mean']} | {s['median']} | {s['std']} | {skew_flag}{s['skewness']} | {out_flag}{s['outliers']} ({s['outlier_pct']}%) |")
            a("")

            # Skewness insights
            skewed = [(c, s["skewness"]) for c, s in numeric.items() if abs(s["skewness"]) > 1]
            if skewed:
                a("### Distribution Insights")
                for col, skew in skewed:
                    direction = "right (positive)" if skew > 0 else "left (negative)"
                    a(f"- **{col}** is highly skewed {direction} (skewness: {skew:.2f}) — consider log or square-root transformation")
                a("")

            # Outlier insights
            outlier_cols = [(c, s["outlier_pct"]) for c, s in numeric.items() if s["outlier_pct"] > 5]
            if outlier_cols:
                a("### Outlier Insights")
                for col, pct in sorted(outlier_cols, key=lambda x: -x[1]):
                    a(f"- **{col}** has {pct}% outliers — investigate whether these are data errors or genuine extremes")
                a("")

        # ── Correlations ──────────────────────────────────────────────
        if correlations.get("strong_pairs"):
            a("## 🔗 Correlation Insights\n")
            pairs = correlations["strong_pairs"]
            very_strong = [p for p in pairs if abs(p["r"]) >= 0.9]
            strong      = [p for p in pairs if 0.7 <= abs(p["r"]) < 0.9]
            moderate    = [p for p in pairs if 0.5 <= abs(p["r"]) < 0.7]
            if very_strong:
                a("**Very strong correlations (|r| ≥ 0.9):**")
                for p in very_strong:
                    direction = "positive" if p["r"] > 0 else "negative"
                    a(f"- `{p['col_a']}` ↔ `{p['col_b']}`: r = **{p['r']}** ({direction}) — possible redundancy or multicollinearity")
            if strong:
                a("\n**Strong correlations (0.7 ≤ |r| < 0.9):**")
                for p in strong:
                    a(f"- `{p['col_a']}` ↔ `{p['col_b']}`: r = {p['r']}")
            if moderate:
                a("\n**Moderate correlations (0.5 ≤ |r| < 0.7):**")
                for p in moderate:
                    a(f"- `{p['col_a']}` ↔ `{p['col_b']}`: r = {p['r']}")
            a("")

        # ── Categorical Analysis ──────────────────────────────────────
        if categorical:
            a("## 🏷️ Categorical Columns\n")
            for col, info in categorical.items():
                a(f"### {col}")
                a(f"- **{info['unique']}** unique values")
                top = list(info["top_values"].items())[:5]
                top_str = ", ".join([f"`{k}` ({v:,})" for k, v in top])
                a(f"- Top values: {top_str}")
                if info["unique"] > 50:
                    a(f"- ⚠️ High cardinality ({info['unique']} unique values) — consider grouping rare values")
                if info["missing"] > 0:
                    a(f"- 🟡 {info['missing']:,} missing values")
                a("")

        # ── User Question ─────────────────────────────────────────────
        if user_question.strip():
            a("## ❓ Your Question\n")
            a(f"> {user_question}\n")
            a("Based on the analysis above, here are relevant findings:\n")
            q = user_question.lower()
            if any(w in q for w in ["missing", "null", "empty"]):
                if missing:
                    top_missing = sorted(missing.items(), key=lambda x: -x[1]["pct"])[:3]
                    for col, info in top_missing:
                        a(f"- **{col}** has the most missing data: {info['pct']}% ({info['count']:,} values)")
                else:
                    a("- No missing values found in this dataset.")
            elif any(w in q for w in ["correlat", "relat", "connect"]):
                if correlations.get("strong_pairs"):
                    for p in correlations["strong_pairs"][:3]:
                        a(f"- `{p['col_a']}` and `{p['col_b']}` have a correlation of {p['r']}")
                else:
                    a("- No strong correlations (|r| ≥ 0.5) found between numeric columns.")
            elif any(w in q for w in ["outlier", "anomal", "extreme"]):
                outlier_cols = [(c, s["outlier_pct"]) for c, s in numeric.items() if s["outlier_pct"] > 0]
                for col, pct in sorted(outlier_cols, key=lambda x: -x[1])[:5]:
                    a(f"- **{col}**: {pct}% outliers")
            else:
                a("- Review the sections above for detailed findings relevant to your question.")
            a("")

        # ── Recommendations ───────────────────────────────────────────
        a("## 💡 Recommendations\n")
        recs = []
        if duplicates["count"] > 0:
            recs.append(f"Remove {duplicates['count']:,} duplicate rows using `df.drop_duplicates()`")
        high_missing = [c for c, v in missing.items() if v["pct"] > 50]
        if high_missing:
            recs.append(f"Consider dropping columns with >50% missing: {', '.join(high_missing)}")
        med_missing = [c for c, v in missing.items() if 5 < v["pct"] <= 50]
        if med_missing:
            recs.append(f"Impute columns with moderate missing values: {', '.join(med_missing)}")
        skewed = [c for c, s in numeric.items() if abs(s["skewness"]) > 1]
        if skewed:
            recs.append(f"Apply log/sqrt transformation to skewed columns: {', '.join(skewed)}")
        high_outliers = [c for c, s in numeric.items() if s["outlier_pct"] > 10]
        if high_outliers:
            recs.append(f"Investigate outliers in: {', '.join(high_outliers)}")
        very_strong_corr = [p for p in correlations.get("strong_pairs", []) if abs(p["r"]) >= 0.9]
        if very_strong_corr:
            recs.append(f"Check for multicollinearity — very strong correlations found between some columns")
        high_card = [c for c, info in categorical.items() if info["unique"] > 50]
        if high_card:
            recs.append(f"Group rare categories in high-cardinality columns: {', '.join(high_card)}")
        if not recs:
            recs.append("Dataset looks clean! Ready for analysis or modelling.")
        for i, rec in enumerate(recs, 1):
            a(f"{i}. {rec}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _generate_charts(self, df: pd.DataFrame) -> list:
        charts = []
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in numeric_cols[:6]:
            fig = px.histogram(df, x=col, title=f"Distribution: {col}",
                               nbins=40, template="plotly_white", marginal="box",
                               color_discrete_sequence=["#667eea"])
            charts.append({"title": f"Distribution: {col}", "fig": fig})

        if len(numeric_cols) >= 2:
            fig = px.imshow(df[numeric_cols].corr(), title="Correlation Heatmap",
                            color_continuous_scale="RdBu", template="plotly_white",
                            aspect="auto", text_auto=".2f")
            charts.append({"title": "Correlation Heatmap", "fig": fig})

        if numeric_cols:
            fig = px.box(df[numeric_cols[:5]], title="Box Plots — Outlier Overview",
                         template="plotly_white", color_discrete_sequence=["#764ba2"])
            charts.append({"title": "Box Plots", "fig": fig})

        for col in cat_cols[:3]:
            vc = df[col].value_counts().head(15).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=f"Value Counts: {col}",
                         template="plotly_white", color_discrete_sequence=["#48bb78"])
            charts.append({"title": f"Value Counts: {col}", "fig": fig})

        if 2 <= len(numeric_cols) <= 6:
            fig = px.scatter_matrix(df[numeric_cols], title="Scatter Matrix",
                                    template="plotly_white", opacity=0.5)
            charts.append({"title": "Scatter Matrix", "fig": fig})

        return charts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, user_question: str = "",
                filename: str = "dataset", progress_callback=None) -> dict:
        self.df = df

        if progress_callback: progress_callback("📋 Analyzing dataset structure...")
        overview = self._overview(df)

        if progress_callback: progress_callback("🧹 Checking data quality...")
        missing    = self._missing(df)
        duplicates = self._duplicates(df)

        if progress_callback: progress_callback("📊 Computing statistics...")
        numeric      = self._numeric_stats(df)
        correlations = self._correlations(df)
        categorical  = self._categorical_stats(df)

        if progress_callback: progress_callback("💡 Generating insights...")
        report = self._generate_insights(
            overview, missing, duplicates,
            numeric, correlations, categorical, user_question
        )

        if progress_callback: progress_callback("📈 Building visualisations...")
        charts = self._generate_charts(df)

        return {"report": report, "charts": charts}
