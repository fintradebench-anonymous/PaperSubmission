#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinTradeBench Results Aggregator & Plotter.

This script parses the final evaluated RAG vs No-RAG results, calculates
statistical significance (paired t-tests), aggregates metrics by reasoning
category (Fundamental, Trading Signals, Hybrid), and generates the LaTeX
table snippets and PDF plots used in the paper.

Usage (terminal):
    python generate_paper_tables_and_plots.py \
        --input-dir ./output/rag_evaluation_results \
        --output-dir ./output/figures
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use 'Agg' for headless environments (no GUI required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats

# =========================
# CONFIGURATION
# =========================
MODEL_NAME_MAP = {
    'deepseek-r1': 'DeepSeek-R1',
    'gemini-2.5-flash': 'Gemini 2.5 Flash',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
    'llama3.3-70b': 'Llama 3.3 70B (API)',
    'openai-gpt-5-mini': 'GPT-5-mini',

    # Snowflake RAG names
    'rag_results_snowflake_150Q_deepseek-r1': 'DeepSeek-R1',
    'rag_results_snowflake_150Q_gemini-2.5-flash': 'Gemini 2.5 Flash',
    'rag_results_snowflake_150Q_gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
    'rag_results_snowflake_150Q_llama3.3-70b': 'Llama 3.3 70B (API)',
    'rag_results_snowflake_150Q_openai-gpt-5-mini': 'GPT-5-mini',

    # Snowflake Non-RAG names
    'rag_results_snowflake_non_rag_response_150q_deepseek-r1': 'DeepSeek-R1',
    'rag_results_snowflake_non_rag_response_150q_gemini-2.5-flash': 'Gemini 2.5 Flash',
    'rag_results_snowflake_non_rag_response_150q_gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
    'rag_results_snowflake_non_rag_response_150q_llama3.3-70b': 'Llama 3.3 70B (API)',
    'rag_results_snowflake_non_rag_response_150q_openai-gpt-5-mini': 'GPT-5-mini',

    # High Performance Open Weights
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 'R1-Distill-Llama (70B)',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'R1-Distill-Qwen (32B)',
    'meta-llama/Llama-3.3-70B-Instruct': 'Llama 3.3 Instruct (70B)',
    'Qwen/Qwen2.5-32B-Instruct': 'Qwen 2.5 Instruct (32B)',
    'google/gemma-3-27b-it': 'Gemma 3 Instruct (27B)',
    'moonshotai/Kimi-Linear-48B-A3B-Base': 'Kimi-Linear A3B (48B)',
    'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4': 'Nemotron-3 Nano (30B)',

    # Efficient / Edge
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama 3.1 Instruct (8B)',
    'microsoft/phi-4': 'Phi-4 (14B)',
    'mistralai/Mistral-7B-Instruct-v0.2': 'Mistral v0.2 (7B)',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B': 'R1-Distill-Qwen (14B)',
    'google/gemma-3-12b-it': 'Gemma 3 Instruct (12B)',
    'moonshotai/Moonlight-16B-A3B-Instruct': 'Moonlight MoE (16B)',
    'LiquidAI/LFM2.5-1.2B-Thinking': 'LFM 2.5 (1.2B)'
}

CATEGORY_MAP = {
    'Large LLMs': [
        'DeepSeek-R1', 'Gemini 2.5 Flash', 'Gemini 2.5 Flash-Lite', 'GPT-5-mini'
    ],
    'Mid LLMs': [
        'R1-Distill-Llama (70B)', 'R1-Distill-Qwen (32B)',
        'Llama 3.3 Instruct (70B)', 'Llama 3.3 70B (API)',
        'Qwen 2.5 Instruct (32B)'
    ],
    'Small LLMs': [
        'Llama 3.1 Instruct (8B)', 'Phi-4 (14B)', 'Mistral v0.2 (7B)',
        'R1-Distill-Qwen (14B)', 'LFM 2.5 (1.2B)'
    ]
}


def get_category(qid):
    qid = str(qid).upper()
    if qid.startswith('FV'): return 'Hybrid (FT)'
    if qid.startswith('F'):  return 'Fundamental (F)'
    if qid.startswith('V'):  return 'Trading Signals (T)'
    return 'Other'


def get_significance_stars(p_value):
    if pd.isna(p_value): return ""
    if p_value < 0.01: return "^{**}"
    if p_value < 0.05: return "^*"
    return ""


def relative_delta(no_rag, rag):
    """
    Relative delta = (RAG - No-RAG) / No-RAG * 100
    Returns the signed percentage change relative to the No-RAG baseline.
    Returns NaN if No-RAG is zero or missing.
    """
    if pd.isna(no_rag) or pd.isna(rag) or no_rag == 0:
        return np.nan
    return (rag - no_rag) / no_rag * 100


def format_delta(delta, stars=""):
    """Format a relative delta value with colour, sign, and optional significance stars."""
    if pd.isna(delta):
        return "-"
    color = "codegreen" if delta > 0 else "red" if delta < 0 else "black"
    sign = "+" if delta > 0 else ""
    if stars:
        return f"\\textcolor{{{color}}}{{{sign}{delta:.1f}\\%${stars}$}}"
    return f"\\textcolor{{{color}}}{{{sign}{delta:.1f}\\%}}"


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. READ ALL CSV FILES
    # ------------------------------------------------------------------
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{input_dir}'.")
        return

    print(f"Found {len(csv_files)} file(s). Loading...")
    df_list = []
    for file in csv_files:
        try:
            df_list.append(pd.read_csv(file))
        except Exception as e:
            print(f"[WARN] Could not read {file.name}. Error: {e}")

    if not df_list:
        print("[ERROR] No valid data loaded.")
        return

    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset: {len(df)} rows.")

    # ------------------------------------------------------------------
    # 2. DATA PREP
    # ------------------------------------------------------------------
    df['question_id'] = (df['question_id'].astype(str)
                         .str.extract(r'([a-zA-Z]+\d+)', expand=False)
                         .str.upper())
    df['clean_model'] = df['model'].map(MODEL_NAME_MAP).fillna(df['model'])
    df['q_cat'] = df['question_id'].apply(get_category)
    df['mode'] = df['mode'].replace({'Non_RAG': 'No RAG', 'Non-RAG': 'No RAG'})
    df['acc_pct'] = (df['correctness'] / 5) * 100

    # ------------------------------------------------------------------
    # 3. STATISTICAL SIGNIFICANCE (paired t-test per model, overall)
    # ------------------------------------------------------------------
    p_values = {}
    for model in df['clean_model'].unique():
        m_df = df[df['clean_model'] == model]
        aligned = (m_df.pivot_table(index='question_id', columns='mode',
                                    values='correctness', aggfunc='mean')
                   .dropna())
        if 'RAG' in aligned.columns and 'No RAG' in aligned.columns and len(aligned) > 1:
            _, p = stats.ttest_rel(aligned['RAG'], aligned['No RAG'])
            p_values[model] = p
        else:
            p_values[model] = np.nan

    # ------------------------------------------------------------------
    # 4. ACCURACY AGGREGATION
    # ------------------------------------------------------------------
    overall_acc = (df.pivot_table(index='clean_model', columns='mode',
                                  values='acc_pct')
                   .reset_index())
    cat_acc = (df.pivot_table(index=['clean_model', 'q_cat'], columns='mode',
                              values='acc_pct')
               .reset_index())

    # ------------------------------------------------------------------
    # 5. TABLE 1 — MAIN ACCURACY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TABLE 1: MAIN ACCURACY & SIGNIFICANCE TABLE")
    print("=" * 60)

    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\scriptsize")
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\setlength{\\tabcolsep}{3.5pt}")
    print(
        "\\caption{\\textbf{Overall and Category-Specific Accuracy (\\%).} "
        "$\\Delta$ denotes the relative improvement of RAG over No-RAG. "
        "Significance assessed via paired t-test "
        "($^*p<0.05$, $^{**}p<0.01$).}"
    )
    print("\\label{tab:main_accuracy}")
    print("\\begin{tabular}{@{}c l ccc ccc ccc ccc@{}}")
    print("\\toprule")
    print(
        "& & \\multicolumn{3}{c}{\\textbf{Fundamental (F)}} "
        "& \\multicolumn{3}{c}{\\textbf{Trading Signals (T)}} "
        "& \\multicolumn{3}{c}{\\textbf{Hybrid (FT)}} "
        "& \\multicolumn{3}{c}{\\textbf{Overall (All)}} \\\\"
    )
    print("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14}")
    print(
        "\\textbf{Category} & \\textbf{Model} "
        "& \\textbf{No-RAG} & \\textbf{RAG} & \\textbf{$\\Delta$} "
        "& \\textbf{No-RAG} & \\textbf{RAG} & \\textbf{$\\Delta$} "
        "& \\textbf{No-RAG} & \\textbf{RAG} & \\textbf{$\\Delta$} "
        "& \\textbf{No-RAG} & \\textbf{RAG} & \\textbf{$\\Delta$} \\\\"
    )
    print("\\midrule")

    def get_cat_stats(model, c_name):
        c_row = cat_acc[(cat_acc['clean_model'] == model) & (cat_acc['q_cat'] == c_name)]
        if c_row.empty:
            return "-", "-", "-"
        no = c_row['No RAG'].values[0]
        rag = c_row['RAG'].values[0]
        if pd.isna(no):
            return "-", f"{rag:.1f}" if pd.notna(rag) else "-", "-"
        d = relative_delta(no, rag)
        return f"{no:.1f}", f"{rag:.1f}", format_delta(d)

    for cat_name, models in CATEGORY_MAP.items():
        valid_models = [m for m in models if m in overall_acc['clean_model'].values]
        if not valid_models:
            continue

        print(
            f"\\multirow{{{len(valid_models)}}}{{*}}"
            f"{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{cat_name}}}}}}}"
        )

        for model in valid_models:
            o_row = overall_acc[overall_acc['clean_model'] == model].iloc[0]
            o_no = o_row.get('No RAG', np.nan)
            o_rag = o_row.get('RAG', np.nan)
            stars = get_significance_stars(p_values.get(model, np.nan))
            o_delta = relative_delta(o_no, o_rag)

            o_no_str = f"{o_no:.1f}" if pd.notna(o_no) else "-"
            o_rag_str = f"{o_rag:.1f}" if pd.notna(o_rag) else "-"
            o_delta_str = format_delta(o_delta, stars)

            f_no, f_rag, f_del = get_cat_stats(model, 'Fundamental (F)')
            v_no, v_rag, v_del = get_cat_stats(model, 'Trading Signals (T)')
            fv_no, fv_rag, fv_del = get_cat_stats(model, 'Hybrid (FT)')

            print(
                f"& {model} "
                f"& {f_no}  & {f_rag}  & {f_del}  "
                f"& {v_no}  & {v_rag}  & {v_del}  "
                f"& {fv_no} & {fv_rag} & {fv_del} "
                f"& {o_no_str} & {o_rag_str} & {o_delta_str} \\\\"
            )

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    # ------------------------------------------------------------------
    # 6. TABLE 2 — GLOBAL QUALITY METRICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TABLE 2: GLOBAL QUALITY METRICS (RAG vs No-RAG)")
    print("=" * 60)

    quality = df.groupby('mode').agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'fundamental_score': 'mean',
        'trading_signals_score': 'mean',
        'reasoning': 'mean'
    }).T

    METRIC_LABELS = {
        'precision': 'Golden Indicator Precision',
        'recall': 'Golden Indicator Recall',
        'f1_score': 'Golden Indicator F1',
        'fundamental_score': 'Fundamental Integration (1–5)',
        'trading_signals_score': 'Trading Signals Integration (1–5)',
        'reasoning': 'Reasoning Depth (1–5)',
    }

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\small")
    print(
        "\\caption{\\textbf{Global Quality Metrics.} "
        "Averages across all evaluated models. "
        "$\\Delta$ is the relative improvement of RAG over No-RAG.}"
    )
    print("\\label{tab:quality_metrics}")
    print("\\begin{tabular}{@{}l ccc@{}}")
    print("\\toprule")
    print("\\textbf{Metric} & \\textbf{No-RAG} & \\textbf{RAG} & \\textbf{$\\Delta$ (\\%)} \\\\")
    print("\\midrule")

    # Collect data for the plot as well
    plot_metrics, plot_no_rag, plot_rag, plot_deltas = [], [], [], []

    for idx, row in quality.iterrows():
        no = row.get('No RAG', np.nan)
        rag = row.get('RAG', np.nan)
        if pd.isna(no):
            continue
        d = relative_delta(no, rag)
        diff_str = format_delta(d)
        print(f"{METRIC_LABELS.get(idx, idx)} & {no:.2f} & {rag:.2f} & {diff_str} \\\\")

        plot_metrics.append(METRIC_LABELS.get(idx, idx))
        plot_no_rag.append(no)
        plot_rag.append(rag)
        plot_deltas.append(d if not np.isnan(d) else 0)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # ------------------------------------------------------------------
    # 7. FIGURE — Grouped bar chart + Relative Δ overlay
    # ------------------------------------------------------------------
    _generate_quality_figure(plot_metrics, plot_no_rag, plot_rag, plot_deltas, output_dir)


# -------------------------------------------------------------------------
def _generate_quality_figure(metrics, no_rag_vals, rag_vals, deltas, output_dir: Path):
    """
    Two separate figures:
      Figure A — Grouped bar chart: No-RAG vs RAG per metric
      Figure B — Horizontal bar chart: Relative Δ (%) per metric
    """
    color_no_rag = '#4C72B0'
    color_rag = '#55A868'
    color_neg = '#C44E52'

    n = len(metrics)
    x = np.arange(n)
    width = 0.35

    short_labels = [
        m.replace(' (1–5)', '\n(1–5)').replace('Golden Indicator ', 'GI ')
        for m in metrics
    ]

    # ── Figure A: Grouped bar chart ───────────────────────────────────
    fig_a, ax1 = plt.subplots(figsize=(10, 5))

    bars1 = ax1.bar(x - width / 2, no_rag_vals, width, label='No-RAG',
                    color=color_no_rag, edgecolor='white', linewidth=0.6)
    bars2 = ax1.bar(x + width / 2, rag_vals, width, label='RAG',
                    color=color_rag, edgecolor='white', linewidth=0.6)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('Global Quality Metrics: No-RAG vs. RAG', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(max(no_rag_vals), max(rag_vals)) * 1.2)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax1.set_axisbelow(True)

    fig_a.tight_layout()
    path_a = output_dir / 'quality_metrics_grouped_bars.pdf'
    fig_a.savefig(path_a, bbox_inches='tight', dpi=300)
    print(f"[OK] Figure A saved to: {path_a}")
    plt.close(fig_a)

    # ── Figure B: Relative Δ horizontal bar chart ─────────────────────
    short_labels_rev = short_labels[::-1]
    deltas_rev = deltas[::-1]
    colors_rev = [color_rag if d >= 0 else color_neg for d in deltas_rev]

    max_abs = max(abs(d) for d in deltas_rev) if deltas_rev else 1
    x_pad = max_abs * 0.18

    fig_b, ax2 = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(deltas_rev))
    ax2.barh(y_pos, deltas_rev, color=colors_rev,
             edgecolor='white', linewidth=0.6, height=0.55)

    for i, v in enumerate(deltas_rev):
        sign = '+' if v >= 0 else ''
        offset = x_pad * 0.25
        ha = 'left' if v >= 0 else 'right'
        xpos = v + offset if v >= 0 else v - offset
        ax2.text(xpos, i, f'{sign}{v:.1f}%',
                 va='center', ha=ha, fontsize=8.5)

    ax2.axvline(0, color='black', linewidth=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_labels_rev, fontsize=9)

    ax2.set_xlim(-max_abs - x_pad * 2, max_abs + x_pad * 2)

    ax2.set_xlabel('Relative $\\Delta$ (%)', fontsize=10)
    ax2.set_title('Relative RAG Improvement per Metric', fontsize=12, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)

    pos_patch = mpatches.Patch(color=color_rag, label='Positive $\\Delta$')
    neg_patch = mpatches.Patch(color=color_neg, label='Negative $\\Delta$')
    ax2.legend(handles=[pos_patch, neg_patch], fontsize=8, loc='lower right')

    fig_b.tight_layout()
    path_b = output_dir / 'quality_metrics_relative_delta.pdf'
    fig_b.savefig(path_b, bbox_inches='tight', dpi=300)
    print(f"[OK] Figure B saved to: {path_b}")
    plt.close(fig_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaTeX tables and plots for FinTradeBench.")
    parser.add_argument("--input-dir", type=str, default="./output/rag_evaluation_results",
                        help="Directory containing the final evaluated CSV files.")
    parser.add_argument("--output-dir", type=str, default="./output/figures",
                        help="Directory to save the generated PDF plots.")

    args = parser.parse_args()
    main(args)