#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Golden Responses for the FinTradeBench Dataset.

This script aggregates all human annotations and LLM-as-a-Judge evaluations,
merges the scores, and selects the absolute best generated response for each
question. Human scores take precedence over LLM scores. Ties are broken by
preferring a specified baseline model.

Usage (terminal):
    python extract_golden_responses.py \
        --human-dir ./data/human_annotations \
        --llm-dir ./output/llm_judgments \
        --out-dir ./output/golden_responses \
        --preferred-model "gemini" due to stronger human-llm judge agreement. See the paper for details.
"""

import argparse
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# Helpers
# =========================
def parse_maybe_json(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


# =========================
# Data Loaders
# =========================
def load_human_excel(path: Path, sheet_name=None):
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        if sheet_name is None:
            xls = pd.ExcelFile(path)
            sheet_name = xls.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet_name)

    # Build human dimension columns to calculate overall human score
    df["human_acc"] = pd.to_numeric(df.get("H_Accuracy (1-5)"), errors="coerce")
    df["human_comp"] = pd.to_numeric(df.get("H_Completeness (1-5)"), errors="coerce")
    df["human_rel"] = pd.to_numeric(df.get("H_Relevance (1-5)"), errors="coerce")
    df["human_cla"] = pd.to_numeric(df.get("H_Clarity (1-5)"), errors="coerce")
    df["human_overall"] = df[["human_acc", "human_comp", "human_rel", "human_cla"]].mean(axis=1)

    # Ensure necessary metadata columns exist
    for col in ['question_id', 'question', 'golden_indicators', 'response']:
        if col not in df.columns:
            df[col] = np.nan

    return df


def load_llm_judge(path: Path):
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'cp1252', 'latin1']
        df = None
        for enc in encodings_to_try:
            try:
                temp_df = pd.read_csv(path, encoding=enc)
                if len(temp_df.columns) > 1:
                    df = temp_df
                    break
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                continue
        if df is None:
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')

    df.columns = df.columns.str.strip()
    if "qualitative_scores" not in df.columns:
        return pd.DataFrame()  # Skip corrupted files safely

    dims = {
        "llm_acc": ("factual_numerical_accuracy",),
        "llm_comp": ("completeness_context",),
        "llm_rel": ("relevance_utility",),
        "llm_cla": ("clarity_rationale",),
    }

    def extract_dim(js, dim_key):
        obj = parse_maybe_json(js)
        if not isinstance(obj, dict):
            return np.nan
        node = obj.get(dim_key, None)
        if isinstance(node, dict):
            return pd.to_numeric(node.get("score", np.nan), errors="coerce")
        return np.nan

    for outcol, (k,) in dims.items():
        df[outcol] = df["qualitative_scores"].apply(lambda s: extract_dim(s, k))

    df["llm_overall"] = df[["llm_acc", "llm_comp", "llm_rel", "llm_cla"]].mean(axis=1)

    # Ensure raw_output (response text) exists
    if 'raw_output' not in df.columns:
        df['raw_output'] = np.nan

    return df


# =========================
# Main Logic
# =========================
def main(args):
    human_dir = Path(args.human_dir)
    llm_dir = Path(args.llm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Extracting Golden Responses for FinTradeBench ---")

    # 1. Load Data
    human_dfs, llm_dfs = [], []
    for f in human_dir.rglob("*.*"):
        if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            try:
                human_dfs.append(load_human_excel(f))
            except Exception as e:
                print(f"[WARN] Failed to load human file {f.name}: {e}")

    for f in llm_dir.rglob("*.csv"):
        if f.is_file():
            try:
                llm_dfs.append(load_llm_judge(f))
            except Exception as e:
                print(f"[WARN] Failed to load LLM file {f.name}: {e}")

    if not human_dfs or not llm_dfs:
        raise ValueError("[ERROR] Missing Human or LLM files. Please check the provided directories.")

    df_h = pd.concat(human_dfs, ignore_index=True)
    df_l = pd.concat(llm_dfs, ignore_index=True)

    # 2. Clean Join Keys
    merge_keys = ["question_id", "model", "prompt_id"]
    for col in merge_keys:
        if col in df_h.columns: df_h[col] = df_h[col].astype(str).str.strip()
        if col in df_l.columns: df_l[col] = df_l[col].astype(str).str.strip()

    # 3. Smart Prompt Mapping
    prompt_mapping = {
        'L1_Baseline': 'Prompt_L1_Baseline',
        'TELER_L2_Strict_Focus': 'Prompt_L2_Strict_Focus',
        'TELER_L3_Step_By_Step': 'Prompt_L3_Step_By_Step',
        'TELER_L4_Auditor_Evidence': 'Prompt_L4_Auditor_Evidence',
        'TELER_L5_Deconstruction': 'Prompt_L5_Deconstruction',
        'TELER_L6_Maximalist': 'Prompt_L6_Maximalist'
    }

    for model in df_l['model'].dropna().unique():
        h_prompts = df_h[df_h['model'] == model]['prompt_id'].unique() if 'model' in df_h.columns else []
        if any(str(p).startswith('Prompt_') for p in h_prompts):
            mask = df_l['model'] == model
            df_l.loc[mask, 'prompt_id'] = df_l.loc[mask, 'prompt_id'].replace(prompt_mapping)

    # 4. FULL OUTER JOIN (Keep everything!)
    merged = df_h.merge(df_l, on=merge_keys, how="outer", suffixes=("_human", "_llm"))

    # 5. Combine Logic
    # Final score prefers human_overall. If human_overall is NaN, it uses llm_overall.
    merged['final_score'] = merged['human_overall'].combine_first(merged['llm_overall'])

    # Final response prefers the 'response' column from human files, falls back to 'raw_output' from LLM
    merged['response_text'] = merged['response'].combine_first(merged['raw_output'])

    # Broadcast 'question' and 'golden_indicators' across all matching question_ids
    # This fixes issues where a question was only evaluated by the LLM and was missing metadata
    merged['question'] = merged.groupby('question_id')['question'].transform(lambda x: x.ffill().bfill())
    merged['golden_indicators'] = merged.groupby('question_id')['golden_indicators'].transform(
        lambda x: x.ffill().bfill())

    # 6. Tie-Breaker Logic
    # Create a boolean flag for the preferred baseline model (e.g., Gemini)
    merged['is_preferred_model'] = merged['model'].astype(str).str.contains(args.preferred_model, case=False, na=False)

    # Drop any rows completely missing a response or score
    merged = merged.dropna(subset=['question_id', 'response_text', 'final_score'])

    # Sort the dataframe:
    # 1. By Question ID (Ascending)
    # 2. By Final Score (Descending - Highest score first)
    # 3. By is_preferred_model (Descending - True comes before False)
    merged = merged.sort_values(by=['question_id', 'final_score', 'is_preferred_model'], ascending=[True, False, False])

    # 7. Select the best response for each question
    best_responses = merged.drop_duplicates(subset=['question_id'], keep='first')

    # Rename to requested output format
    best_responses = best_responses.rename(columns={'response_text': 'golden_response'})

    # Select final columns
    final_output = best_responses[
        ['question_id', 'question', 'golden_indicators', 'golden_response', 'model', 'final_score']
    ]

    # 8. Export
    out_file = out_dir / "best_golden_responses.csv"
    final_output.to_csv(out_file, index=False)

    print(f"\n[SUCCESS] Extracted the best responses for {len(final_output)} unique questions!")
    print(f"[OK] Saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the best generated responses for FinTradeBench.")
    parser.add_argument("--human-dir", type=str, default="./data/human_annotations",
                        help="Directory containing human annotation Excel/CSV files.")
    parser.add_argument("--llm-dir", type=str, default="./output/llm_judgments",
                        help="Directory containing LLM judge output CSV files.")
    parser.add_argument("--out-dir", type=str, default="./output/golden_responses",
                        help="Directory to save the final golden responses CSV.")
    parser.add_argument("--preferred-model", type=str, default="gemini",
                        help="Substring of the model name to prefer in the event of a score tie (e.g., 'gemini').")

    args = parser.parse_args()
    main(args)