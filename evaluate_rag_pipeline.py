#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinTradeBench RAG Evaluator (LLM-as-a-Judge)

Evaluates batches of RAG vs. Non-RAG model responses against a Gold Standard.
Incorporates human-aligned rubrics for Factual Accuracy and Clarity while
specifically testing for Trading Signals vs. Fundamentals integration bias.

Usage (terminal):
    python evaluate_rag_pipeline.py \
        --rag-dir ./data/rag_outputs \
        --non-rag-dir ./data/non_rag_outputs \
        --ground-truth ./data/golden_responses.csv \
        --project-id YOUR_GCP_PROJECT_ID
"""

import os
import csv
import json
import time
import glob
import math
import argparse
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import AnthropicVertex
from tqdm import tqdm

# =========================
# CONFIGURATION DEFAULTS
# =========================
REGION = "global" # e.g., us-east5 or us-central1
JUDGE_MODEL = "claude-sonnet-4-5@20250929"
BATCH_SIZE = 5  # Models to judge in a single API call

# =========================
# PROMPT TEMPLATES
# =========================

SYSTEM_PROMPT = """You are an expert Financial Analyst and AI Evaluator. 
Your task is to benchmark multiple AI models against a Gold Standard.
You will assess their reasoning quality, factual accuracy, and specifically their usage of key financial indicators."""

BATCH_JUDGE_PROMPT = """
You are evaluating {num_models} different AI models answering the SAME question.
Compare each Candidate Answer against the **Gold Answer**, **Source Context**, and **Golden Indicators**.

--- SHARED DATA (Applies to all models) ---
[Question]: {question}

[Golden Indicators (Reference Metrics)]: 
{golden_indicators}

[Gold Answer (Ground Truth)]:
{gold_answer}

[Context - Trading Signals]:
{trading_context}

[Context - Fundamentals]:
{fun_context}

--- CANDIDATE ANSWERS TO EVALUATE ---
{candidates_text}

--- EVALUATION RUBRIC (1-5 SCALE) ---

1. Factual and Numerical Accuracy (Maps to `correctness_score`)
- Score 5: All numerical claims are supported by the Source Context and Gold Answer.
- Score 3: Minor calculation errors or hallucinations that do not change the overall thesis.
- Score 1: Severe hallucinations or math errors that invalidate the conclusion.

2. Clarity and Rationale (Maps to `reasoning_score`)
- REWARD responses that are highly structured. Step-by-step breakdowns are highly desirable if they make logic easy to follow.
- Score 5: Crisp, highly readable, actionable, and gets straight to the point without filler.
- Score 3: Understandable, but overly wordy, verbose, or relies on clunky formatting.
- Score 1: Confusing, disjointed, or buried in impenetrable financial jargon.

3. Integration Bias (Maps to `trading_signals_score` and `fundamental_score`)
- `trading_signals_score` (1-5): 1 = Ignored trading signals context entirely, 5 = Perfectly integrated trading signals.
- `fundamental_score` (1-5): 1 = Ignored SEC fundamental context entirely, 5 = Perfectly integrated fundamentals.
- **[CRITICAL HUMAN ALIGNMENT RULE]**: Do NOT penalize these scores heavily just because an answer omits *some* Reference Metrics. If the response successfully answers the prompt using a smaller, highly relevant subset of metrics, it MUST score a 4 or 5.

--- OUTPUT FORMAT ---
Return a JSON object containing a list under the key "evaluations".
IMPORTANT JSON RULES:
- Ensure the output is valid JSON.
- Do NOT use unescaped double quotes inside string values.
- Do NOT include trailing commas.

Example:
{{
  "evaluations": [
    {{
      "model_name": "Model Name Here",
      "mode": "RAG",
      "indicator_analysis": {{
         "found_indicators": ["Metric A", "Metric B"],
         "total_golden_indicators": 3,
         "indicators_found_count": 2
      }},
      "bias_analysis": {{
         "trading_signals_score": 4,
         "fundamental_score": 5
      }},
      "overall_quality": {{
         "correctness_score": 4,
         "reasoning_score": 5
      }}
    }}
  ]
}}
"""


# =========================
# HELPER FUNCTIONS
# =========================

def get_claude_response(client, prompt):
    retries = 3
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL, max_tokens=8192, temperature=0,
                system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}]
            )
            text = msg.content[0].text.strip()

            # Extract JSON block
            s, e = text.find('{'), text.rfind('}')
            if s != -1 and e != -1:
                json_str = text[s:e + 1]
                return json.loads(json_str)

            raise ValueError("No JSON block found in response")

        except json.JSONDecodeError as e:
            print(f"[WARN] JSON Parsing Error on attempt {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"[WARN] API Error on attempt {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))

    return {"error": "Failed to generate valid JSON after retries"}

def calculate_f1(precision, recall):
    if precision + recall == 0: return 0
    return 2 * (precision * recall) / (precision + recall)

def normalize_col_name(col):
    return col.strip().lower().replace(" ", "_")

def load_ground_truth(path):
    print(f"Loading Ground Truth: {path}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(path, encoding='cp1252')

    df.columns = [normalize_col_name(c) for c in df.columns]

    if 'is_self_selected' in df.columns:
        df['is_self_selected'] = df['is_self_selected'].astype(str).str.lower().map(
            {'true': True, '1': True, 'yes': True}).fillna(False)
        df = df[df['is_self_selected'] == True].copy()

    gt_map = {}
    for _, row in df.iterrows():
        qid = str(row.get('question_id', row.get('id')))
        gt_map[qid] = {
            'gold_answer': row.get('response', row.get('golden_response', '')),
            'golden_indicators': row.get('golden_indicators', ''),
            'trading_context': str(row.get('trading_context', row.get('volatility_context', '')))[:10000],
            'fun_context': str(row.get('fundamental_context', ''))[:10000]
        }
    return gt_map

def load_candidate_files(directory, mode):
    print(f"Scanning {mode} directory: {directory}")
    all_responses = []
    files = glob.glob(os.path.join(directory, "*.csv"))

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [normalize_col_name(c) for c in df.columns]

            if 'id' in df.columns: df.rename(columns={'id': 'question_id'}, inplace=True)
            df['mode'] = mode
            df['source_file'] = os.path.basename(f)

            if 'model' not in df.columns:
                df['model'] = os.path.splitext(os.path.basename(f))[0]

            all_responses.append(df)
        except Exception as e:
            print(f"[WARN] Skipping bad file {f}: {e}")

    if not all_responses:
        return pd.DataFrame()
    return pd.concat(all_responses, ignore_index=True)


# =========================
# EVALUATION LOGIC
# =========================

def process_batch(question_id, gt_data, batch_rows, client):
    candidates_text = ""
    for i, row in enumerate(batch_rows):
        ans = str(row.get('final_answer', ''))
        if len(ans) > 4000: ans = ans[:4000] + "...[TRUNCATED]"

        candidates_text += f"""
--- CANDIDATE {i + 1} ---
Model Name: {row.get('model', 'Unknown')}
Mode: {row.get('mode', 'Unknown')}
Answer: 
{ans}
"""

    prompt = BATCH_JUDGE_PROMPT.format(
        num_models=len(batch_rows),
        question=str(batch_rows[0].get('query', batch_rows[0].get('question', ''))),
        golden_indicators=gt_data['golden_indicators'],
        gold_answer=gt_data['gold_answer'],
        trading_context=gt_data['trading_context'],
        fun_context=gt_data['fun_context'],
        candidates_text=candidates_text
    )

    parsed = get_claude_response(client, prompt)
    results = []

    try:
        if "error" in parsed:
            raise ValueError(parsed["error"])

        evals = parsed.get('evaluations', [])

        for i, row in enumerate(batch_rows):
            target_model = row.get('model', 'Unknown')
            target_mode = row.get('mode', 'Unknown')

            if i < len(evals):
                eval_data = evals[i]
            else:
                eval_data = {}

            ind_data = eval_data.get('indicator_analysis', {})
            found_count = ind_data.get('indicators_found_count', 0)
            total_ind = ind_data.get('total_golden_indicators', 1)
            if total_ind == 0: total_ind = 1

            recall = found_count / total_ind
            precision = 1.0 if found_count > 0 else 0.0
            f1 = calculate_f1(precision, recall)

            res_row = {
                "question_id": question_id,
                "model": target_model,
                "mode": target_mode,
                "overall_score": (eval_data.get('overall_quality', {}).get('correctness_score', 0) +
                                  eval_data.get('overall_quality', {}).get('reasoning_score', 0)) / 2,
                "correctness": eval_data.get('overall_quality', {}).get('correctness_score', 0),
                "reasoning": eval_data.get('overall_quality', {}).get('reasoning_score', 0),
                "trading_signals_score": eval_data.get('bias_analysis', {}).get('trading_signals_score', 0),
                "fundamental_score": eval_data.get('bias_analysis', {}).get('fundamental_score', 0),
                "indicators_found": ", ".join(ind_data.get('found_indicators', [])),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1_score": round(f1, 2),
                "time_total": row.get('time_total_sec', 0),
                "time_gen": row.get('time_generation_sec', 0),
                "final_answer": row.get('final_answer', '')[:5000]
            }
            results.append(res_row)

    except Exception as e:
        print(f"[ERROR] Parsing batch for QID {question_id}: {e}")
        for row in batch_rows:
            results.append({
                "question_id": question_id,
                "model": row.get('model'),
                "mode": row.get('mode'),
                "overall_score": -1,
                "error": str(e)
            })

    return results


# =========================
# MAIN
# =========================
def main(args):
    load_dotenv()

    try:
        client = AnthropicVertex(region=REGION, project_id=args.project_id)
    except Exception as e:
        print(f"[ERROR] Failed to init Vertex AI: {e}")
        return

    gt_map = load_ground_truth(args.ground_truth)
    gold_ids = set(gt_map.keys())
    print(f"\n--- [GT PHASE] ---")
    print(f"Verified Gold Standard Questions: {len(gold_ids)}")

    df_rag = load_candidate_files(args.rag_dir, "RAG")
    df_non_rag = load_candidate_files(args.non_rag_dir, "Non_RAG")

    if df_rag.empty and df_non_rag.empty:
        print("[ERROR] No candidate responses loaded from directories.")
        return

    df_all = pd.concat([df_rag, df_non_rag], ignore_index=True)
    df_all['question_id'] = df_all['question_id'].astype(str)

    candidate_ids = set(df_all['question_id'].unique())
    print(f"Total Unique Question IDs found in CSV files: {len(candidate_ids)}")

    discarded_ids = candidate_ids - gold_ids
    missing_from_candidates = gold_ids - candidate_ids

    report_path = "missing_and_discarded_report.txt"
    with open(report_path, "w") as f:
        f.write("=== DATA DISCREPANCY REPORT ===\n\n")
        f.write(f"DATE: {datetime.now()}\n")
        f.write(f"GOLD STANDARD TARGET: {len(gold_ids)} IDs\n")
        f.write(f"CANDIDATE POOL SIZE: {len(candidate_ids)} IDs\n\n")

        f.write(f"--- DISCARDED IDs ({len(discarded_ids)}) ---\n")
        f.write("(Found in candidate CSVs but ignored because they are not Gold Standard)\n")
        f.write(", ".join(sorted(list(discarded_ids))) + "\n\n")

        f.write(f"--- MISSING IDs ({len(missing_from_candidates)}) ---\n")
        f.write("(Exist in Gold Standard but were NOT found in any RAG/Non-RAG CSV files)\n")
        f.write(", ".join(sorted(list(missing_from_candidates))) + "\n")

    print(f"⚠️  Discarded {len(discarded_ids)} non-gold IDs.")
    print(f"❌  Missing {len(missing_from_candidates)} gold IDs from candidate files.")
    print(f"📄  Detailed report exported to: {report_path}")

    df_filtered = df_all[df_all['question_id'].isin(gold_ids)].copy()

    initial_rows = len(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=['question_id', 'model', 'mode'], keep='first')
    dupes_removed = initial_rows - len(df_filtered)

    if dupes_removed > 0:
        print(f"🧹 Cleaned {dupes_removed} redundant model responses.")

    grouped = df_filtered.groupby('question_id')
    print(f"\n--- [FINAL EXECUTION] ---")
    print(f"Processing {len(grouped)} unique questions across all models.")

    all_batches = []
    for qid, group in grouped:
        gt_data = gt_map[qid]
        records = group.to_dict('records')

        num_batches = math.ceil(len(records) / BATCH_SIZE)
        for i in range(num_batches):
            batch_rows = records[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            all_batches.append((qid, gt_data, batch_rows))

    print(f"Created {len(all_batches)} evaluation batches for LLM API.")

    headers = [
        "question_id", "model", "mode", "overall_score",
        "correctness", "reasoning", "trading_signals_score", "fundamental_score",
        "precision", "recall", "f1_score", "indicators_found",
        "time_total", "time_gen", "final_answer", "error"
    ]

    with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(process_batch, qid, gt, rows, client) for qid, gt, rows in all_batches]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results = f.result()
            with lock:
                with open(args.output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerows(results)

    print(f"\n✅ Evaluation Complete. Results: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG vs Non-RAG models using Claude as a Judge.")
    parser.add_argument("--rag-dir", type=str, required=True, help="Path to directory containing RAG outputs.")
    parser.add_argument("--non-rag-dir", type=str, required=True, help="Path to directory containing Non-RAG outputs.")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to Golden Responses CSV.")
    parser.add_argument("--project-id", type=str, required=True, help="GCP Project ID for Vertex AI.")
    parser.add_argument("--output-file", type=str, default=f"rag_evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.csv")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of concurrent threads.")

    args = parser.parse_args()
    main(args)