#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a utility script for Phase 2 (Human Evaluation) of the FinTradeBench pipeline.

It reads the model-generated responses (e.g., `model_results.csv`), filters
for the self-selected answers, and generates two files:
    1. `human_annotation_input_[DATE].csv`: A clean CSV for your experts to fill out.
    2. `human_annotation_rubric_[DATE].md`: A Markdown file explaining their task.

Usage (terminal):
    python prepare_human_annotation_task.py \
        --input-file ./output/model_results.csv \
        --output-dir ./output/human_tasks
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# =========================
# RUBRIC FOR HUMAN ANNOTATORS
# =========================
HUMAN_RUBRIC_MD = """# Human Annotation Task: FinTradeBench Rubric

## Your Task
You will be given a CSV file (`human_annotation_input.csv`) containing financial reasoning questions and AI-generated answers. Your job is to act as a financial expert and score these answers based on the provided ground truth and your domain knowledge.

For each row, please provide scores (1-5) for the 5 qualitative criteria. You may also provide a 'Golden Answer' if the AI completely failed.

## Evaluation Criteria (Score 1-5)

Please provide a score from 1 (Very Poor) to 5 (Excellent) for each.

1.  **Audit_Validation_Agreement (Score 0 or 1):**
    * Look at the `numerical_audit_report` and its `is_numerically_accurate` flag (1=Accurate, 0=Errors).
    * Do you agree with the automated audit's conclusion?
    * **Score 1:** Yes, I agree.
    * **Score 0:** No, I disagree (the audit missed an error, or flagged a correct answer).

2.  **Factual_Numerical_Accuracy (Score 1-5):**
    * Based on your own review (and the audit), what is the final accuracy score?
    * **5:** 100% correct, zero hallucinations.
    * **1:** Contains significant, misleading errors.

3.  **Completeness_Context (Score 1-5):**
    * Does the answer fully address the question? Does it correctly *use and contextualize* the required metrics (see `golden_indicators`)?
    * **5:** Excellent. Uses required metrics in a deep, accurate analysis.
    * **1:** Superficial. Misses most required metrics or critical context.

4.  **Relevance_Utility (Score 1-5):**
    * Is every piece of information relevant? Does it avoid "fluff" or misleading information?
    * **5:** Excellent. High precision, actionable, no fluff.
    * **1:** Very Poor. Cluttered with irrelevant or tangentially related info.

5.  **Clarity_Rationale (Score 1-5):**
    * Is the answer clear, well-structured, and easy to understand? Does it explain its reasoning?
    * **5:** Exceptionally clear, readable, and well-reasoned.
    * **1:** Confusing, poorly written, overly robotic, or a "black box" without rationale.

## Output Columns (What you need to fill in)

In the assigned CSV, please fill in the following columns (the first 5 are 1-5, #1 is 0/1):

* `H_Audit_Agreement (0/1)`
* `H_Accuracy (1-5)`
* `H_Completeness (1-5)`
* `H_Relevance (1-5)`
* `H_Clarity (1-5)`
* `H_Golden_Answer (Optional)`: If the answer is bad, write your own *ideal* answer here.
* `H_Notes (Optional)`: Any other comments or observations.

Save your completed file as `human_annotation_results_COMPLETED.csv`.
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare human annotation tasks from model outputs.")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV containing generated model responses.")
    parser.add_argument("--output-dir", type=str, default="./output/human_tasks",
                        help="Directory to save the annotation CSV and Rubric MD.")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_csv = out_dir / f"human_annotation_input_{timestamp}.csv"
    output_rubric = out_dir / f"human_annotation_rubric_{timestamp}.md"
    output_completed_example = out_dir / f"human_annotation_results_COMPLETED_EXAMPLE_{timestamp}.csv"

    print(f"--- Starting Human Annotation Prep ---")
    print(f"Loading {input_path}...")

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {input_path}")
        return

    # Handle string boolean values if necessary
    if 'is_self_selected' in df.columns:
        df['is_self_selected'] = df['is_self_selected'].replace(
            {'True': True, 'False': False, 'true': True, 'false': False}
        ).astype(bool)
        df_selected = df[df['is_self_selected'] == True].copy()
    else:
        print("[WARN] 'is_self_selected' column missing. Using all rows.")
        df_selected = df.copy()

    if df_selected.empty:
        print("[ERROR] No self-selected answers found in the input file.")
        return

    print(f"[INFO] Found {len(df_selected)} target answers for human review.")

    # Select and order columns for the human task
    human_task_cols = [
        "question_id",
        "question",
        "golden_indicators",
        "model",
        "prompt_id",
        "response",
        "is_numerically_accurate",
        "numerical_audit_report"
    ]

    # Ensure all columns exist to prevent KeyError
    existing_cols = [col for col in human_task_cols if col in df_selected.columns]
    df_task = df_selected[existing_cols].copy()

    # Add the empty columns for humans to fill
    human_output_cols = [
        "H_Audit_Agreement (0/1)",
        "H_Accuracy (1-5)",
        "H_Completeness (1-5)",
        "H_Relevance (1-5)",
        "H_Clarity (1-5)",
        "H_Golden_Answer (Optional)",
        "H_Notes (Optional)"
    ]

    for col in human_output_cols:
        df_task[col] = ""

    # Save the CSV task for annotators
    df_task.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"[OK] Successfully created annotation task file: {output_csv.name}")

    # Save the rubric
    with open(output_rubric, 'w', encoding='utf-8') as f:
        f.write(HUMAN_RUBRIC_MD)
    print(f"[OK] Successfully created annotation rubric: {output_rubric.name}")

    # Create an empty example of the "completed" file for reference
    (df_task.head(2).to_csv(output_completed_example, index=False, encoding='utf-8-sig'))

    print(f"\n--- NEXT STEPS ---")
    print(f"1. Distribute '{output_csv.name}' and '{output_rubric.name}' to your experts.")
    print(f"2. Ask them to fill in the 'H_' columns and save their work.")
    print(f"3. Run the alignment metrics script on their completed files.")


if __name__ == "__main__":
    main()