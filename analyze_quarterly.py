"""
Quarterly Sales & Margin Analysis
-----------------------------------
Loads up to 4 quarterly Excel files (same structure as the single-quarter
script), then produces:

  1. Console summary — per-quarter totals and QoQ deltas
  2. Per-company grouped bar chart  — Sales across all quarters
  3. Per-company margin % trend chart — line + markers per company
  4. Margin delta heatmap            — QoQ change in margin % per company
  5. Outlier detection               — flags extreme margin % each quarter
  6. Saves all charts as PNG files

Usage examples
--------------
  # Filenames default to Q1.xlsx, Q2.xlsx, Q3.xlsx, Q4.xlsx
  python analyze_quarterly.py

  # Or supply any 1-4 files explicitly
  python analyze_quarterly.py Q1_Sales.xlsx Q2_Sales.xlsx Q3_Sales.xlsx Q4_Sales.xlsx

  # Works with just 2 quarters too
  python analyze_quarterly.py jan_mar.xlsx apr_jun.xlsx
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_FILES   = ["Q1.xlsx", "Q2.xlsx", "Q3.xlsx", "Q4.xlsx"]
QUARTER_LABELS  = ["Q1", "Q2", "Q3", "Q4"]   # label applied in order supplied
OUTLIER_STDEV   = 1.5    # σ threshold for outlier flagging
MARGIN_HIGH     = 0.30   # ≥ 30 % → High tier
MARGIN_LOW      = 0.20   # < 20 % → Low tier

# Chart palette
COLORS = ["#2E86AB", "#F18F01", "#C73E1D", "#3B1F2B"]   # one per quarter
POS_COLOR = "#2ecc71"   # green  — margin improved
NEG_COLOR = "#e74c3c"   # red    — margin declined
NEUTRAL   = "#95a5a6"
# ───────────────────────────────────────────────────────────────────────────────


# ── Data loading ───────────────────────────────────────────────────────────────

def load_quarter(path: str, label: str) -> pd.DataFrame:
    """Load one quarterly Excel file, clean it, and tag it with a quarter label."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df = df[df["BillTo"].notna()].copy()
    df["BillTo"]      = df["BillTo"].astype(int)
    df["Description"] = df["Description"].str.strip()
    df["Margin %"]    = df["Margin %"].astype(float)
    df["Quarter"]     = label
    return df


def load_all(files: list[str], labels: list[str]) -> pd.DataFrame:
    """Load all supplied files and return a single long-format DataFrame."""
    frames = []
    for path, label in zip(files, labels):
        p = Path(path)
        if not p.exists():
            print(f"  ⚠  File not found, skipping: {path}")
            continue
        frames.append(load_quarter(path, label))
        print(f"  ✓  Loaded {label}: {path}")
    if not frames:
        raise FileNotFoundError("No valid Excel files found. Check file paths.")
    return pd.concat(frames, ignore_index=True)


# ── Console output ─────────────────────────────────────────────────────────────

def print_quarterly_summary(df: pd.DataFrame, quarters: list[str]) -> None:
    """Print per-quarter totals and QoQ deltas."""
    print("\n" + "=" * 70)
    print("QUARTERLY SUMMARY")
    print("=" * 70)
    fmt_sales  = lambda v: f"${v:>13,.0f}"
    fmt_margin = lambda v: f"{v * 100:>7.2f}%"

    prev = None
    for q in quarters:
        qdf = df[df["Quarter"] == q]
        if qdf.empty:
            continue
        total_sales  = qdf["Sales"].sum()
        total_margin = qdf["Margin"].sum()
        avg_margin_p = qdf["Margin %"].mean()

        print(f"\n  {q}")
        print(f"    Companies     : {len(qdf)}")
        print(f"    Total Sales   : {fmt_sales(total_sales)}")
        print(f"    Total Margin  : {fmt_sales(total_margin)}")
        print(f"    Avg Margin %  : {fmt_margin(avg_margin_p)}")

        if prev is not None:
            delta_sales  = total_sales  - prev["sales"]
            delta_margin = avg_margin_p - prev["margin_p"]
            arrow_s = "▲" if delta_sales  >= 0 else "▼"
            arrow_m = "▲" if delta_margin >= 0 else "▼"
            print(f"    QoQ Sales Δ   : {arrow_s} ${abs(delta_sales):>12,.0f}")
            print(f"    QoQ Margin Δ  : {arrow_m} {abs(delta_margin) * 100:>6.2f}pp")

        prev = {"sales": total_sales, "margin_p": avg_margin_p}
    print()


def print_outliers(df: pd.DataFrame, quarters: list[str]) -> None:
    """Flag companies whose margin % is > OUTLIER_STDEV σ from the quarter mean."""
    print("=" * 70)
    print(f"OUTLIER DETECTION  (threshold: ±{OUTLIER_STDEV}σ per quarter)")
    print("=" * 70)
    found_any = False
    for q in quarters:
        qdf = df[df["Quarter"] == q]
        if qdf.empty:
            continue
        mean  = qdf["Margin %"].mean()
        stdev = qdf["Margin %"].std()
        outliers = qdf[np.abs(qdf["Margin %"] - mean) > OUTLIER_STDEV * stdev]
        if not outliers.empty:
            found_any = True
            print(f"\n  {q}  (mean={mean*100:.1f}%,  σ={stdev*100:.1f}%)")
            for _, row in outliers.iterrows():
                tag = "HIGH" if row["Margin %"] > mean else "LOW "
                print(f"    [{tag}]  {row['Description']:<22} "
                      f"Margin: {row['Margin %']*100:>5.1f}%")
    if not found_any:
        print("  No outliers detected.\n")
    print()


# ── Charts ─────────────────────────────────────────────────────────────────────

def chart_sales_by_company(pivot_sales: pd.DataFrame, quarters: list[str]) -> str:
    """Grouped bar chart: Sales per company, one bar group per company."""
    companies = pivot_sales.index.tolist()
    n_cos     = len(companies)
    n_q       = len(quarters)
    x         = np.arange(n_cos)
    width     = 0.8 / n_q

    fig, ax = plt.subplots(figsize=(max(14, n_cos * 0.7), 7))
    for i, (q, color) in enumerate(zip(quarters, COLORS)):
        if q not in pivot_sales.columns:
            continue
        vals   = pivot_sales[q].fillna(0).values
        offset = (i - n_q / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=q, color=color, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=40, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Sales by Company — All Quarters", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Sales ($)")
    ax.legend(title="Quarter")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = "chart_sales_by_company.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_margin_trends(pivot_margin: pd.DataFrame, quarters: list[str]) -> str:
    """Line chart: Margin % trend per company across quarters."""
    companies = pivot_margin.index.tolist()
    n_cos     = len(companies)

    # Layout: up to 4 per row
    ncols = min(4, n_cos)
    nrows = int(np.ceil(n_cos / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.2),
                             squeeze=False)

    q_present = [q for q in quarters if q in pivot_margin.columns]

    for idx, company in enumerate(companies):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        vals = [pivot_margin.loc[company, q] if q in pivot_margin.columns else np.nan
                for q in q_present]

        ax.plot(q_present, [v * 100 for v in vals],
                marker="o", linewidth=2, color="#2E86AB", markersize=6)

        # Colour-code each point by delta direction
        for j in range(1, len(vals)):
            if np.isnan(vals[j]) or np.isnan(vals[j-1]):
                continue
            color = POS_COLOR if vals[j] >= vals[j-1] else NEG_COLOR
            ax.plot(q_present[j], vals[j] * 100,
                    marker="o", markersize=8, color=color, zorder=5)

        ax.set_title(company, fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(60, max((v*100 for v in vals if not np.isnan(v)), default=50) + 5))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for idx in range(n_cos, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Margin % Trend per Company  (green dot = improved, red = declined)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = "chart_margin_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_margin_delta_heatmap(pivot_margin: pd.DataFrame, quarters: list[str]) -> str | None:
    """Heatmap of QoQ margin % point change per company."""
    q_present = [q for q in quarters if q in pivot_margin.columns]
    if len(q_present) < 2:
        print("  (Skipping delta heatmap — need at least 2 quarters)")
        return None

    # Build delta DataFrame  (QoQ change in percentage points)
    delta_cols = []
    for i in range(1, len(q_present)):
        col_label = f"{q_present[i-1]}→{q_present[i]}"
        delta_cols.append(col_label)
        pivot_margin[col_label] = (pivot_margin[q_present[i]] - pivot_margin[q_present[i-1]]) * 100

    delta_df = pivot_margin[delta_cols].copy()

    abs_max = delta_df.abs().max().max()
    if abs_max == 0:
        abs_max = 1
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig, ax = plt.subplots(figsize=(max(6, len(delta_cols) * 2.5),
                                    max(6, len(delta_df) * 0.45 + 1.5)))
    im = ax.imshow(delta_df.values, cmap="RdYlGn", norm=norm, aspect="auto")

    # Annotate cells
    for r in range(len(delta_df)):
        for c in range(len(delta_cols)):
            val = delta_df.iloc[r, c]
            if np.isnan(val):
                continue
            text_color = "black" if abs(val) < abs_max * 0.6 else "white"
            sign = "+" if val > 0 else ""
            ax.text(c, r, f"{sign}{val:.1f}pp",
                    ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(len(delta_cols)))
    ax.set_xticklabels(delta_cols, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(delta_df.index, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Margin Δ (percentage points)", fontsize=9)
    ax.set_title("Quarter-over-Quarter Margin % Change per Company\n"
                 "(green = improved, red = declined)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = "chart_margin_delta_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_overall_qoq(df: pd.DataFrame, quarters: list[str]) -> str:
    """Side-by-side bars: total Sales and avg Margin % per quarter."""
    q_present = [q for q in quarters if q in df["Quarter"].values]
    totals = df.groupby("Quarter").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Margin=("Margin %", "mean")
    ).reindex(q_present)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sales
    bars = ax1.bar(q_present, totals["Total_Sales"],
                   color=[COLORS[i % len(COLORS)] for i in range(len(q_present))],
                   alpha=0.88, width=0.5)
    for bar, val in zip(bars, totals["Total_Sales"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + totals["Total_Sales"].max() * 0.01,
                 f"${val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_title("Total Sales by Quarter", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Sales ($)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    # Avg Margin %
    bars2 = ax2.bar(q_present, totals["Avg_Margin"] * 100,
                    color=[COLORS[i % len(COLORS)] for i in range(len(q_present))],
                    alpha=0.88, width=0.5)
    for bar, val in zip(bars2, totals["Avg_Margin"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.3,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_title("Avg Margin % by Quarter", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Avg Margin %")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Overall Quarter-over-Quarter Performance", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = "chart_overall_qoq.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Resolve file list
    if len(sys.argv) > 1:
        files  = sys.argv[1:]
        labels = [QUARTER_LABELS[i] if i < len(QUARTER_LABELS) else f"Q{i+1}"
                  for i in range(len(files))]
    else:
        files  = DEFAULT_FILES
        labels = QUARTER_LABELS

    print("\nLoading files...")
    df = load_all(files, labels)

    quarters_present = [q for q in labels if q in df["Quarter"].values]

    # ── Pivot tables for charting ──────────────────────────────────────────────
    pivot_sales  = df.pivot_table(index="Description", columns="Quarter",
                                  values="Sales",    aggfunc="sum")
    pivot_margin = df.pivot_table(index="Description", columns="Quarter",
                                  values="Margin %", aggfunc="mean")

    # Align company order by total sales descending
    company_order = (pivot_sales.fillna(0).sum(axis=1)
                     .sort_values(ascending=False).index.tolist())
    pivot_sales  = pivot_sales.reindex(company_order)
    pivot_margin = pivot_margin.reindex(company_order)

    # ── Console outputs ────────────────────────────────────────────────────────
    print_quarterly_summary(df, quarters_present)
    print_outliers(df, quarters_present)

    # ── Charts ─────────────────────────────────────────────────────────────────
    print("Generating charts...")
    outputs = []

    outputs.append(chart_overall_qoq(df, quarters_present))
    print("  ✓  Overall QoQ summary chart")

    outputs.append(chart_sales_by_company(pivot_sales, quarters_present))
    print("  ✓  Sales by company (grouped bars)")

    outputs.append(chart_margin_trends(pivot_margin, quarters_present))
    print("  ✓  Margin % trend per company")

    heatmap_out = chart_margin_delta_heatmap(pivot_margin, quarters_present)
    if heatmap_out:
        outputs.append(heatmap_out)
        print("  ✓  Margin delta heatmap")

    print("\nSaved charts:")
    for f in outputs:
        print(f"  →  {f}")
    print()


if __name__ == "__main__":
    main()
