# Software Supplementary Material: FinTradeBench

This anonymized repository contains the complete, end-to-end Python codebase required to reproduce **FinTradeBench**, execute the Retrieval-Augmented Generation (RAG) pipeline on heterogeneous signals, and run the LLM-as-a-Judge evaluations detailed in the paper.

In compliance with NeurIPS double-blind review policies, the repository is hosted via an anonymized link, and all local paths, institutional identifiers, and specific API keys have been removed.

In compliance with double-blind review policies, all local paths, institutional identifiers, and API keys have been removed. 

## ⚙️ Setup & Installation

All scripts are written in Python 3.10+. To install the required dependencies, run:

    pip install -r requirements.txt

*Note: Scripts requiring external APIs (Google Vertex AI, Anthropic, Snowflake) use `argparse` to accept credentials at runtime. You must supply your own GCP Project IDs or Snowflake credentials to execute those specific generation/evaluation steps.*

---

## 📂 Repository Structure

    FinTradeBench_Software/
    ├── README.md                              <- This master guide
    ├── requirements.txt                       <- Python dependencies
    │
    ├── generate_question_variants.py          <- (Phase 1)
    ├── download_sec_filings.py                <- (Phase 1)
    ├── compute_financial_signals.py           <- (Phase 1)
    ├── generate_benchmark_responses.py        <- (Phase 2)
    ├── prepare_human_annotation_task.py       <- (Phase 2)
    ├── calculate_human_alignment.py           <- (Phase 2)
    ├── extract_golden_responses.py            <- (Phase 2)
    ├── run_snowflake_rag_pipeline.py          <- (Phase 3)
    ├── evaluate_rag_pipeline.py               <- (Phase 4)
    ├── generate_paper_tables_and_plots.py     <- (Phase 4)
    │
    ├── dummy_data                             <- Sample data to run the codes.
    ├── data/                                  <- (Empty placeholder for inputs)
    └── output/                                <- (Empty placeholder for outputs)

---

## 🚀 Pipeline Execution Guide

The codebase is modularized into four distinct phases. Reviewers can run individual scripts using the `--help` flag (e.g., `python download_sec_filings.py --help`) to view all available arguments and default directory paths.

### Phase 1: Data Collection & Precomputation
These scripts build the raw corpus required for the benchmark.
1. **`download_sec_filings.py`**: Interacts with the SEC EDGAR API to download historical 10-K and 10-Q corporate filings. Includes rate-limiting compliance.
2. **`compute_financial_signals.py`**: Parses raw OHLCV price history and SEC financial workbooks to compute the expert-defined "Golden Indicators" (e.g., RSI, MACD, Debt/Equity).
3. **`generate_question_variants.py`**: Scales the benchmark by programmatically swapping companies, tickers, and dates within the base question templates.
Note: The price data is provided in the zipped folder for dataset but only a sample of the SEC filings are included due to size constraints. The `download_sec_filings.py` script can be used to obtain the full set of filings if desired.

### Phase 2: Ground Truth & Human Alignment
These scripts execute the TELeR prompt taxonomy, prep data for human experts, and calculate inter-rater reliability.
4. **`generate_benchmark_responses.py`**: Uses Vertex AI to generate baseline responses across multiple prompt complexities (TELeR L1-L6) along with automated numerical auditing.
5. **`prepare_human_annotation_task.py`**: Formats the LLM-generated responses into a blinded CSV task and generates a Markdown grading rubric for human experts.
6. **`calculate_human_alignment.py`**: Computes statistical agreement (MAE, Cohen's Kappa, Krippendorff's Alpha, Spearman/Pearson correlation) between Human Experts and the LLM-as-a-Judge.
7. **`extract_golden_responses.py`**: Merges human and LLM evaluations, applying tie-breaker logic to extract the absolute best "Golden Response" for the final ground-truth dataset.

### Phase 3: The RAG Architecture
This script executes the core experiment of the paper.
8. **`run_snowflake_rag_pipeline.py`**: An end-to-end multimodal RAG system. It parses HTML SEC filings, chunks documents, embeds them using `BAAI/bge-large-en-v1.5`, creates a local FAISS vector index, performs dense retrieval + BM25, and routes the multimodal context to Snowflake Cortex LLMs (DeepSeek, Llama, Gemini).

### Phase 4: Evaluation & Visualization
These scripts grade the RAG outputs and generate the paper's quantitative figures.
9. **`evaluate_rag_pipeline.py`**: Uses Claude 4.5 Sonnet as a judge to evaluate the Standard RAG vs. No-RAG responses against the Golden Answers. It outputs scores for Factual Accuracy, Reasoning, and Modality Integration Bias.
10. **`generate_paper_tables_and_plots.py`**: Aggregates the final evaluation CSVs, computes paired t-test significance, and outputs the LaTeX table snippets and PDF plots (Grouped Bar Charts & Relative Deltas) utilized in the Results section of the manuscript.


The full anonymous dataset can be accessed using the link: https://dataverse.harvard.edu/previewurl.xhtml?token=7424f412-d56d-4a3e-9469-907c243444fb
