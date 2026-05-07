#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human-LLM Judge Alignment and Agreement Metrics.

This script calculates statistical correlation (Spearman, Pearson, Kendall)
and inter-rater reliability (Cohen's Kappa, Krippendorff's Alpha) between
human expert annotations and the LLM-as-a-Judge outputs.

Usage (terminal):
    python calculate_human_alignment.py \
        --human-dir ./data/human_annotations \
        --llm-dir ./output/llm_judgments \
        --out-dir ./output/alignment_results
"""

import argparse
from pathlib import Path
import re
import json
import ast
import numpy as np
import pandas as pd
import warnings

from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import cohen_kappa_score
import krippendorff

# =========================
# Helpers
# =========================
# Regex matches Fundamental (F), Trading Signals/Volatility (V), or Hybrid (FV)
_QTYPE_RE = re.compile(r"^(FV|F|V)\s*\d+", flags=re.IGNORECASE)


def derive_question_type(qid):
    if not isinstance(qid, str):
        return "UNK"
    qid = qid.strip().upper()
    m = _QTYPE_RE.match(qid)
    return m.group(1) if m else "UNK"


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


def corr_metrics(x_human, y_llm):
    x = pd.to_numeric(x_human, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_llm, errors="coerce").to_numpy(dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return {"n": n, "spearman": np.nan, "pearson": np.nan, "kendall": np.nan,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan}

    if np.std(x) == 0 or np.std(y) == 0:
        sp, pr, kt = np.nan, np.nan, np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = spearmanr(x, y).correlation
            pr = pearsonr(x, y)[0]
            kt = kendalltau(x, y).correlation

    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))

    return {"n": n, "spearman": sp, "pearson": pr, "kendall": kt,
            "mae": mae, "rmse": rmse, "bias": bias}


def pretty(name, m):
    return {
        "group": name,
        "n": m["n"],
        "spearman_rho": m["spearman"],
        "pearson_r": m["pearson"],
        "kendall_tau": m["kendall"],
        "MAE": m["mae"],
        "RMSE": m["rmse"],
        "Bias_LLM_minus_Human": m["bias"],
    }


# =========================
# Agreement Metrics
# =========================
def _to_int_array(series: pd.Series) -> np.ndarray:
    return np.round(
        pd.to_numeric(series, errors="coerce").clip(1, 5)
    ).astype("Int64")


def _aligned_int_pair(human: pd.Series, llm: pd.Series):
    h = _to_int_array(human)
    l = _to_int_array(llm)
    mask = ~(h.isna() | l.isna())
    return h[mask].astype(int).to_numpy(), l[mask].astype(int).to_numpy()


def weighted_kappa(human: pd.Series, llm: pd.Series) -> dict:
    h, l = _aligned_int_pair(human, llm)
    n = len(h)
    if n < 2 or len(np.unique(h)) < 2:
        return {"n": n, "kappa": np.nan, "interpretation": "undefined (zero variance)"}

    kappa = cohen_kappa_score(h, l, weights="quadratic")

    if kappa < 0:
        interp = "poor (worse than chance)"
    elif kappa < 0.20:
        interp = "slight"
    elif kappa < 0.40:
        interp = "fair"
    elif kappa < 0.60:
        interp = "moderate"
    elif kappa < 0.80:
        interp = "substantial"
    else:
        interp = "almost perfect"

    return {"n": n, "kappa": round(kappa, 4), "interpretation": interp}


def krippendorff_alpha(human: pd.Series, llm: pd.Series, level: str = "ordinal") -> dict:
    h = pd.to_numeric(human, errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(llm, errors="coerce").to_numpy(dtype=float)

    reliability_matrix = np.array([h, l])
    n_valid = int(np.sum(~np.isnan(h) & ~np.isnan(l)))

    if n_valid < 2:
        return {"n": n_valid, "alpha": np.nan, "level": level, "interpretation": "undefined (insufficient data)"}

    h_valid = h[~np.isnan(h)]
    l_valid = l[~np.isnan(l)]
    if len(np.unique(h_valid)) < 2 and len(np.unique(l_valid)) < 2:
        return {"n": n_valid, "alpha": np.nan, "level": level, "interpretation": "undefined (zero variance)"}

    try:
        alpha = krippendorff.alpha(reliability_data=reliability_matrix, level_of_measurement=level)
    except ValueError as e:
        return {"n": n_valid, "alpha": np.nan, "level": level, "interpretation": f"undefined ({e})"}

    if alpha >= 0.80:
        interp = "reliable (>=0.80)"
    elif alpha >= 0.67:
        interp = "tentative (0.67–0.80)"
    else:
        interp = "unreliable (<0.67)"

    return {"n": n_valid, "alpha": round(alpha, 4), "level": level, "interpretation": interp}


def agreement_row(name: str, human: pd.Series, llm: pd.Series) -> dict:
    kappa_res = weighted_kappa(human, llm)
    alpha_ord = krippendorff_alpha(human, llm, level="ordinal")
    alpha_int = krippendorff_alpha(human, llm, level="interval")

    return {
        "group": name,
        "n": kappa_res["n"],
        "cohen_kappa_quadratic": kappa_res["kappa"],
        "kappa_interpretation": kappa_res["interpretation"],
        "krippendorff_alpha_ordinal": alpha_ord["alpha"],
        "krippendorff_alpha_interval": alpha_int["alpha"],
        "alpha_interpretation": alpha_ord["interpretation"],
    }


def compute_agreement_metrics(merged: pd.DataFrame, existing_results: list, out_dir: Path) -> pd.DataFrame:
    agreement_results = []
    dim_pairs = [
        ("overall", "human_overall", "llm_overall"),
        ("dim_accuracy", "human_acc", "llm_acc"),
        ("dim_completeness", "human_comp", "llm_comp"),
        ("dim_relevance", "human_rel", "llm_rel"),
        ("dim_clarity", "human_cla", "llm_cla"),
    ]
    for label, h_col, l_col in dim_pairs:
        agreement_results.append(agreement_row(label, merged[h_col], merged[l_col]))

    qt_col = "question_type_human" if "question_type_human" in merged.columns else "question_type"
    for qt, g in merged.groupby(qt_col, dropna=False):
        agreement_results.append(agreement_row(f"question_type={qt}", g["human_overall"], g["llm_overall"]))

    model_col = "model" if "model" in merged.columns else "model_human"
    for m, g in merged.groupby(model_col, dropna=False):
        agreement_results.append(agreement_row(f"model={m}", g["human_overall"], g["llm_overall"]))

    agreement_df = pd.DataFrame(agreement_results)
    agreement_path = out_dir / "agreement_metrics.csv"
    agreement_df.to_csv(agreement_path, index=False)
    print(f"[OK] Wrote agreement metrics: {agreement_path}")

    corr_df = pd.DataFrame(existing_results)
    combined = corr_df.merge(agreement_df, on=["group", "n"], how="outer")
    combined_path = out_dir / "full_alignment_metrics.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[OK] Wrote full alignment metrics: {combined_path}")

    print("\n" + "=" * 65)
    print(" AGREEMENT METRICS SUMMARY")
    print("=" * 65)
    cols = ["group", "n", "cohen_kappa_quadratic", "kappa_interpretation", "krippendorff_alpha_ordinal",
            "alpha_interpretation"]
    print(agreement_df[cols].to_string(index=False))

    return agreement_df


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

    df["question_type"] = df["question_id"].apply(derive_question_type)
    df["human_acc"] = pd.to_numeric(df["H_Accuracy (1-5)"], errors="coerce")
    df["human_comp"] = pd.to_numeric(df["H_Completeness (1-5)"], errors="coerce")
    df["human_rel"] = pd.to_numeric(df["H_Relevance (1-5)"], errors="coerce")
    df["human_cla"] = pd.to_numeric(df["H_Clarity (1-5)"], errors="coerce")
    df["human_overall"] = df[["human_acc", "human_comp", "human_rel", "human_cla"]].mean(axis=1)

    if "H_Audit_Agreement (0/1)" in df.columns:
        df["human_audit_agreement"] = pd.to_numeric(df["H_Audit_Agreement (0/1)"], errors="coerce")
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
        raise KeyError(f"Failed to find 'qualitative_scores' in {path.name}. Usually indicates corrupted CSV.")

    df["question_type"] = df["question_id"].apply(derive_question_type)
    dims = {
        "llm_acc": ("factual_numerical_accuracy",),
        "llm_comp": ("completeness_context",),
        "llm_rel": ("relevance_utility",),
        "llm_cla": ("clarity_rationale",),
    }

    def extract_dim(js, dim_key):
        obj = parse_maybe_json(js)
        if not isinstance(obj, dict): return np.nan
        node = obj.get(dim_key, None)
        if isinstance(node, dict): return pd.to_numeric(node.get("score", np.nan), errors="coerce")
        return np.nan

    for outcol, (k,) in dims.items():
        df[outcol] = df["qualitative_scores"].apply(lambda s: extract_dim(s, k))

    df["llm_overall"] = df[["llm_acc", "llm_comp", "llm_rel", "llm_cla"]].mean(axis=1)
    return df


# =========================
# Main Execution
# =========================
def main(args):
    human_dir = Path(args.human_dir)
    llm_dir = Path(args.llm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all human files into a single DataFrame
    human_dfs = []
    for f in human_dir.rglob("*.*"):
        if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls', '.csv']:
            try:
                human_dfs.append(load_human_excel(f))
                print(f"[OK] Loaded Human file: {f.name}")
            except Exception as e:
                print(f"[WARNING] Skipping {f.name}: {e}")

    if not human_dfs:
        raise ValueError(f"No valid human annotation files found in {human_dir}")
    df_h = pd.concat(human_dfs, ignore_index=True)

    # 2. Load all LLM files into a single DataFrame
    llm_dfs = []
    for f in llm_dir.rglob("*.csv"):
        if f.is_file():
            try:
                llm_dfs.append(load_llm_judge(f))
                print(f"[OK] Loaded LLM file: {f.name}")
            except Exception as e:
                print(f"[WARNING] Skipping {f.name}: {e}")

    if not llm_dfs:
        raise ValueError(f"No valid LLM judge files found in {llm_dir}")
    df_l = pd.concat(llm_dfs, ignore_index=True)

    # 3. Force merge keys to be strings and strip whitespace
    merge_keys = ["question_id", "model", "prompt_id"]
    for col in merge_keys:
        if col in df_h.columns:
            df_h[col] = df_h[col].astype(str).str.strip()
        if col in df_l.columns:
            df_l[col] = df_l[col].astype(str).str.strip()

    # 4. CONDITIONAL MAPPING
    prompt_mapping = {
        'L1_Baseline': 'Prompt_L1_Baseline',
        'TELER_L2_Strict_Focus': 'Prompt_L2_Strict_Focus',
        'TELER_L3_Step_By_Step': 'Prompt_L3_Step_By_Step',
        'TELER_L4_Auditor_Evidence': 'Prompt_L4_Auditor_Evidence',
        'TELER_L5_Deconstruction': 'Prompt_L5_Deconstruction',
        'TELER_L6_Maximalist': 'Prompt_L6_Maximalist',
        'No RAG': 'No RAG'
    }

    for model in df_l['model'].unique():
        h_prompts = df_h[df_h['model'] == model]['prompt_id'].unique()
        if any(str(p).startswith('Prompt_') for p in h_prompts):
            print(f"[INFO] Applied mapping to align prompts for model: {model}")
            mask = df_l['model'] == model
            df_l.loc[mask, 'prompt_id'] = df_l.loc[mask, 'prompt_id'].replace(prompt_mapping)

    # 5. Merge on shared keys
    merged = df_h.merge(df_l, on=merge_keys, how="inner", suffixes=("_human", "_llm"))

    print(f"\n[INFO] Successfully merged {len(merged)} rows across all files.")
    if len(merged) == 0:
        raise ValueError("Merge produced 0 rows. Check your folder paths and ensure the models overlap.")

    # 6. Generate Metrics
    results = []
    results.append(pretty("overall", corr_metrics(merged["human_overall"], merged["llm_overall"])))
    results.append(pretty("dim_accuracy", corr_metrics(merged["human_acc"], merged["llm_acc"])))
    results.append(pretty("dim_completeness", corr_metrics(merged["human_comp"], merged["llm_comp"])))
    results.append(pretty("dim_relevance", corr_metrics(merged["human_rel"], merged["llm_rel"])))
    results.append(pretty("dim_clarity", corr_metrics(merged["human_cla"], merged["llm_cla"])))

    for qt, g in merged.groupby("question_type_human", dropna=False):
        results.append(pretty(f"question_type={qt}", corr_metrics(g["human_overall"], g["llm_overall"])))

    for m, g in merged.groupby("model", dropna=False):
        results.append(pretty(f"model={m}", corr_metrics(g["human_overall"], g["llm_overall"])))

    out = pd.DataFrame(results)
    out.to_csv(out_dir / "human_llm_alignment_metrics.csv", index=False)

    compute_agreement_metrics(merged, results, out_dir)
    merged.to_csv(out_dir / "human_llm_merged_rows.csv", index=False)

    if "human_audit_agreement" in merged.columns and "is_numerically_accurate" in merged.columns:
        ha = pd.to_numeric(merged["human_audit_agreement"], errors="coerce")
        na = pd.to_numeric(merged["is_numerically_accurate"], errors="coerce")
        mask = ~ha.isna() & ~na.isna()
        if mask.sum() > 0:
            agreement = (ha[mask] == na[mask]).mean()
            print(f"[INFO] Human audit agreement vs numeric audit match-rate: {agreement:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Human-LLM alignment metrics.")
    parser.add_argument("--human-dir", type=str, default="./data/human_annotations",
                        help="Directory containing human annotation Excel/CSV files.")
    parser.add_argument("--llm-dir", type=str, default="./output/llm_judgments",
                        help="Directory containing LLM judge output CSV files.")
    parser.add_argument("--out-dir", type=str, default="./output/alignment_results",
                        help="Directory to save the computed alignment metrics.")

    args = parser.parse_args()
    main(args)