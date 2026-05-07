# Dataset Supplementary Material: FinTradeBench (Anonymous Submission)

This folder contains a small, anonymized subset of the full FinTradeBench dataset. It is provided strictly to allow reviewers to quickly verify the reproducibility of the code and execute a test run of the Retrieval-Augmented Generation (RAG) pipeline without downloading the full multi-gigabyte corpus.

The complete dataset (1400 questions) and full corpus will be made anonymously available until review under NEURIPS and will be made open-source upon paper acceptance.

## Folder Structure

```text
FinTradeBench_Dataset_Sample/
│
├── README.md                              <- This documentation file
├── golden_subset_cleaned.csv                      <- The seed questions from benchmark (150 Q&A pairs)
│
├── Precomputed_Contexts/                  <- Data for the "Ideal RAG" Case Studies
│   ├── AAPL-daily_with_fundamentals.csv
│   └── ... 
│
└── Raw_RAG_Documents/                     <- Raw corpus for "Standard RAG" Evaluation
    ├── SEC_Filings/                       <- Unstructured fundamental data (10-K/10-Q)
    │   ├── AAPL_2024.htm
    │   └── ...
    └── Stock_Price_History/               <- Structured market data (OHLCV)
        ├── AAPL-history.xlsx
        └── ...
```
## File Descriptions & Data Schema

### 1. ` golden_subset_cleaned.csv`
This file contains the core benchmark evaluation data. It contains the seed 150 queries used to scale the benchmark covering Fundamental (F), Trading (T), and Hybrid (FT) reasoning types.
Category breakdown:
  Fundamentals (F):    50 questions
  Hybrid (FT):         50 questions
  Trading signals (T): 50 questions

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `question_id` | String | Unique identifier prefix denoting the reasoning category (F, T, or FT) followed by a number. | `FT1` |
| `question` | String | The prompt presented to the LLM. | *"As of August 2025, is Apple a good buy given its valuation..."* |
| `golden_indicators` | String | Comma-separated list of expert-defined metrics required to form a complete and correct rationale. | `Earnings/Price, Book/Price, RSI` |
| `golden_response` | String | Response selected for our benchmark after the benchmark generation pipeline is completed|
---

### 2. `Precomputed_Contexts/` (csv Files)
This folder contains the combined, perfectly curated data used for the "Ideal RAG" (Level 7-9) case studies. 
* **Naming Convention:** `[TICKER]-daily_with_fundamentals.csv` (e.g., `AAPL-daily_with_fundamentals.csv`)
* **Structure:** Each file contains the pre-extracted company fundamentals (prefix F_) like Return on Equity, Debt/Assets) and the calculated trading signals (no prefix) like 20-day EMA, RSI, etc. relevant to the specific timeframe of the query. 
* **Purpose:** Used to bypass the retrieval bottleneck to test if the LLM possesses the latent reasoning capacity to synthesize the data when provided perfectly.

Full signal formulae and economic interpretations are provided
in Appendix B, Table 6 of the accompanying paper.
---

### 3. `Raw_RAG_Documents/`
This directory contains the raw, dense multimodal corpus that the RAG pipeline (Level 5-6) indexes and retrieves from.

**Subfolder: `SEC_Filings/`**
* **Structure:** Organized into sub-directories by stock ticker (e.g., `SEC_Filings/AAPL/`).
* **Format:** `.htm` or `.txt`
* **Naming Convention:** `[Filing Type]_[Date]_[Accession Number].htm` (e.g., `10-K_2016-10-26_0001628280-16-020309.htm`)
* **Description:** Raw, unstructured corporate filings downloaded directly from the SEC EDGAR database. These contain massive amounts of textual boilerplate, risk factors, and accounting tables.

**Subfolder: `Stock_Price_History/`**
* **Format:** `.xlsx`
* **Naming Convention:** `[TICKER]-history.xlsx`
* **Schema:** Standardized tabular time-series data.
  * `Date`: Trading day (YYYY-MM-DD).
  * `Open`, `High`, `Low`, `Close`: Daily price action in USD.
  * `Adj Close`: Closing price adjusted for splits and dividends.
  * `Volume`: Total shares traded during the day.

===============================================================

DATA PROVENANCE AND LICENSING

===============================================================

  SEC Filings:   Public domain. Downloaded from SEC EDGAR
                 (https://www.sec.gov/edgar). No redistribution
                 restrictions apply to US regulatory filings.

  Price Data:    Sourced from publicly available market data feeds.
                 Users are responsible for complying with the terms
                 of their data provider when using this data for
                 commercial purposes.

===============================================================

COVERAGE NOTE

===============================================================

The NASDAQ-100 universe covers 101 companies over the 2015–2025
window. Due to file size constraints, SEC filings are provided 
for a representative subset of 1 company only.
The full company set can be reconstructed using the EDGAR
downloader script (download_sec_filings.py) in the software package. 
Signal computation scripts are also provided to reproduce the signals/ files from
raw OHLCV and filing data.

===============================================================

CONTACT

===============================================================

Will be made available after anonymous review process is completed.

*Note: For the full dataset required to reproduce the paper's exact evaluation metrics, please refer to the anonymous dataset link provided in the main README.*
