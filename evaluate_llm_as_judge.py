#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge Evaluation Script (using Claude 3.5 Sonnet via Vertex AI).

This script automates the evaluation of the generated benchmark responses.
It prompts an expert LLM to grade the generated answers against human-expert
"Golden Indicators" and the automated numerical audit report. The judge
outputs qualitative scores (1-5) and exact metric extraction calculations (F1).

Usage (terminal):
    python evaluate_llm_as_judge.py \
        --input-file ./output/model_results.csv \
        --output-file ./output/judged_results.csv \
        --project-id YOUR_GCP_PROJECT_ID \
        --region us-east5 \
        --max-workers 8
"""

import os
import csv
import json
import argparse
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import AnthropicVertex
from tqdm import tqdm
import threading
from datetime import datetime

# =========================
# CONFIGURATION DEFAULTS
# =========================
JUDGE_MODEL = "claude-sonnet-4-5@20250929"  # Latest Claude 3.5 Sonnet on Vertex

# =========================
# LLM-as-a-Judge Prompt
# =========================
SYSTEM_PROMPT = "You are an expert financial analyst and a meticulous fact-checker."

LLM_AS_JUDGE_PROMPT_TEMPLATE = """
Your task is to evaluate a generated answer to a specific financial question against a set of golden reference metrics and an automated numerical audit.

INPUT DATA:
1.  [Question]: {question}
2.  [Reference Metrics] ($M_{{ref}}$): {golden_indicators}
3.  [Automated Audit Report]: {numerical_audit_report}
4.  [Generated Answer]: {response}

### EVALUATION RUBRIC (1-5 SCALE) ###

1. Factual and Numerical Accuracy (1-5)
Rely heavily on the Numerical Audit Report. 
- Score 5: All numerical claims are supported by the audit.
- Score 3: Minor calculation errors or hallucinations that do not change the overall thesis.
- Score 1: Severe hallucinations or math errors that invalidate the conclusion.

2. Completeness and Context (1-5) **[CRITICAL HUMAN ALIGNMENT RULE]**
Evaluate if the response sufficiently answers the core financial question.
- Do NOT penalize the response heavily just because it omits some of the Reference Metrics. If the response successfully and comprehensively answers the prompt using a smaller, highly relevant subset of metrics, it MUST score a 4 or 5.
- Score 5: Fully addresses the prompt with sufficient context and strong explanatory power.
- Score 3: Addresses the main points but leaves minor sub-questions unanswered.
- Score 1: Fails to address the core question or misses critical necessary context.

3. Relevance and Utility (1-5)
Evaluate how useful the analysis is to an investor or financial decision-maker.
- Score 5: Highly actionable, directly answers the prompt without digressing.
- Score 3: Generally relevant, but includes some tangential information.
- Score 1: Misses the point entirely or provides useless information.

4. Clarity and Rationale (1-5) **[CRITICAL HUMAN ALIGNMENT RULE]**
Evaluate the readability, directness, and conciseness of the response.
- REWARD responses that are highly structured. Step-by-step breakdowns (e.g., 'Step 1: Goal...') are highly desirable and should receive a 5 for Clarity if they make the financial logic easy to follow.
- Score 5: Crisp, highly readable, actionable, and gets straight to the point without filler.
- Score 3: Understandable, but overly wordy, verbose, or relies on clunky formatting.
- Score 1: Confusing, disjointed, or buried in impenetrable financial jargon.

### FEW-SHOT ANCHOR EXAMPLES ###
- Anchor 1 (Completeness): If a response perfectly answers "has the stock bottomed?" using just 2 metrics with great reasoning, do NOT dock Completeness points for missing a 3rd or 4th reference metric. Give it a 5.
- Anchor 2 (Clarity): If a response is accurate but starts with a long-winded definition of what a stock is, or uses highly repetitive, robotic step-by-step headers that waste space, cap the Clarity score at 3.

### OUTPUT FORMAT ###
You must return ONLY a valid JSON object. Do not include markdown formatting like ```json. 
Crucially, you must output the `qualitative_scores` FIRST, and the `metric_analysis` SECOND.

{{
  "qualitative_scores": {{
    "factual_numerical_accuracy": {{
      "score": 0,
      "justification": "Reference the Automated Audit Report explicitly."
    }},
    "completeness_context": {{
      "score": 0,
      "justification": "Assess if all M_ref concepts were covered."
    }},
    "relevance_utility": {{
      "score": 0,
      "justification": "Assess if the answer directly addresses the prompt without fluff."
    }},
    "clarity_rationale": {{
      "score": 0,
      "justification": "Is the financial reasoning sound and easy to follow?"
    }}
  }},
  "metric_analysis": {{
    "metrics_generated_M_gen": ["list of metrics found in answer"],
    "metrics_intersection_M_ref_and_M_gen": ["list of semantic matches"],
    "metric_precision": 0.0,
    "metric_recall": 0.0,
    "metric_f1_score": 0.0
  }}
}}
"""


def get_claude_response_threadsafe(client: AnthropicVertex, prompt: str) -> str:
    """
    Gets response from Claude (Vertex AI) and extracts the JSON object.
    """
    try:
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            temperature=0,  # Temperature 0 for deterministic evaluation
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Extract text content
        text = message.content[0].text.strip()

        # --- Robust JSON Extraction ---
        json_start = text.find('{')
        json_end = text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text[json_start:json_end + 1]
            return json_str
        else:
            # Fallback if Claude is chatty
            print(f"[WARN] No valid JSON found. Raw: {text[:100]}...")
            return json.dumps({
                "error": "No valid JSON object found in response",
                "raw_output": text
            })

    except Exception as e:
        return json.dumps({"error": f"Vertex API call failed: {str(e)}"})


def write_header(output_file: str, header: list[str]):
    """Creates or overwrites the output CSV with a header."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=header).writeheader()


def append_row(output_file: str, row: dict[str, any], header: list[str], lock: threading.Lock):
    """Thread-safe appender."""
    with lock:
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)


def run_judge_task(row: dict[str, any], client: AnthropicVertex) -> dict[str, any]:
    """Runs one judging task using the passed client."""
    try:
        prompt = LLM_AS_JUDGE_PROMPT_TEMPLATE.format(
            question=row.get('question', ''),
            golden_indicators=row.get('golden_indicators', '[]'),
            numerical_audit_report=row.get('numerical_audit_report', 'N/A'),
            response=row.get('response', '')
        )
    except KeyError as e:
        return {"error": f"Prompt formatting error: {e}"}

    judge_json_str = get_claude_response_threadsafe(client, prompt)

    try:
        judge_data = json.loads(judge_json_str)
    except json.JSONDecodeError:
        judge_data = {"error": "Failed to decode JSON", "raw_output": judge_json_str}

    base_info = {
        "question_id": row.get('question_id'),
        "model": row.get('model'),
        "prompt_id": row.get('prompt_id'),
        "judge_json_output": judge_json_str
    }

    return {**base_info, **judge_data}


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses using Claude 3.5 via Vertex AI.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV containing generated responses.")
    parser.add_argument("--output-file", type=str, default=f"judged_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        help="Path to save the judged results.")
    parser.add_argument("--project-id", type=str, required=True,
                        help="Your Google Cloud Project ID for Vertex AI authentication.")
    parser.add_argument("--region", type=str, default="us-east5",
                        help="GCP Region for Claude (e.g., us-east5 or us-central1).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Number of concurrent threads (Adjust based on your quota).")

    args = parser.parse_args()

    load_dotenv()

    # Initialize Vertex Client
    try:
        client = AnthropicVertex(region=args.region, project_id=args.project_id)
    except Exception as e:
        print(f"[ERROR] Failed initializing Vertex AI client: {e}")
        return

    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input_file}")
        return

    # Normalize boolean column for self-selected answers
    if 'is_self_selected' in df.columns:
        df['is_self_selected'] = df['is_self_selected'].replace(
            {'True': True, 'False': False, 'true': True, 'false': False}
        ).astype(bool)
        df_selected = df[df['is_self_selected'] == True].copy()
    else:
        print("[WARN] 'is_self_selected' column missing. Processing ALL rows in the CSV.")
        df_selected = df.copy()

    if df_selected.empty:
        print("[INFO] No rows selected for judging. Exiting.")
        return

    tasks = df_selected.to_dict('records')

    output_header = [
        "question_id", "model", "prompt_id", "judge_json_output",
        "metric_analysis", "qualitative_scores", "overall_assessment",
        "error", "raw_output"
    ]

    write_header(args.output_file, output_header)
    write_lock = threading.Lock()

    print(f"--- Starting Claude-as-a-Judge Evaluation ---")
    print(f"Input:    {args.input_file}")
    print(f"Output:   {args.output_file}")
    print(f"Tasks:    {len(tasks)} answers to evaluate")
    print(f"Workers:  {args.max_workers}")
    print(f"Region:   {args.region}")
    print("---------------------------------------------")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_judge_task, task, client) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging Answers"):
            try:
                result = fut.result()
                # Ensure all header keys exist in the result dictionary to avoid KeyError
                row_to_write = {k: result.get(k, "") for k in output_header}
                append_row(args.output_file, row_to_write, output_header, write_lock)
            except Exception as e:
                print(f"[ERROR] Task failed: {e}")

    print(f"\n✅ Evaluation complete. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()