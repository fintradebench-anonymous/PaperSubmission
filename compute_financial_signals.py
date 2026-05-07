#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute Volatility and Fundamental Signals for the FinTradeBench Dataset.

This script processes raw historical price data (OHLCV) and quarterly
financial SEC data to calculate the expert-defined Golden Indicators
(e.g., RSI, MACD, Debt/Equity) required for the benchmark evaluation.

Usage (terminal):
    python compute_financial_signals.py --history-dir ./data/history \
        --financials-dir ./data/financial_raw --output-dir ./output
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ----------------------------
# Volatility signals
# ----------------------------
def calculate_volatility_signals(history_df: pd.DataFrame) -> pd.DataFrame:
    df = history_df.copy()

    # Ensure required columns exist
    required = {'Adj. Close', 'Volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"History missing required columns: {missing}")

    # MA(20)
    df['MA_20'] = df['Adj. Close'].rolling(window=20).mean()

    # MACD (12,26,9)
    exp1 = df['Adj. Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj. Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI(14) using simple mean on gains/losses
    delta = df['Adj. Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # EMA(20)
    df['EMA_20'] = df['Adj. Close'].ewm(span=20, adjust=False).mean()

    # OBV
    df['OBV'] = (np.sign(df['Adj. Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # One-day return
    df['One_Day_Reversal'] = df['Adj. Close'].pct_change()

    # Max 1-day return over 20 trading days
    df['Max_Return_20D'] = df['One_Day_Reversal'].rolling(window=20).max()

    # Momentum
    df['Momentum_5D'] = df['Adj. Close'] / df['Adj. Close'].shift(5) - 1
    df['Momentum_20D'] = df['Adj. Close'] / df['Adj. Close'].shift(20) - 1

    # Long-term mean reversion vs 60D mean
    df['Mean_Reversal_60D'] = df['Adj. Close'] / df['Adj. Close'].rolling(window=60).mean() - 1

    return df


# ----------------------------
# File discovery & loading
# ----------------------------
def list_history_files(history_path: str) -> list[Path]:
    """Find history files with common patterns."""
    p = Path(history_path)
    patterns = ["*-history.xlsx", "*-history.csv", "*-history.xlsx - Export.csv"]
    files = []
    for pat in patterns:
        files.extend(p.glob(pat))
    # De-duplicate (some files might match more than one pattern)
    files = sorted(list({f.resolve() for f in files}))
    return files


def load_history_file(path: Path) -> pd.DataFrame:
    """Load CSV or XLSX and return sorted by Date."""
    if str(path).lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Normalize columns (some exports use "Adj Close" vs "Adj. Close")
    if 'Adj Close' in df.columns and 'Adj. Close' not in df.columns:
        df = df.rename(columns={'Adj Close': 'Adj. Close'})

    # Date handling
    if 'Date' not in df.columns:
        raise ValueError(f"'Date' column not found in {path.name}")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Basic sanity columns
    for col in ['Adj. Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in {path.name}")
    return df


# ----------------------------
# Fundamentals (Quarterly) from single Excel workbook
# ----------------------------
def _tidy_financial_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a wide sheet into tidy format with:
      Date | Metric1 | Metric2 | ...
    Assumes first column is metric names, subsequent columns are dates or labels (e.g., TTM).
    """
    df = df.copy()
    # Ensure first column is 'Metric'
    df.rename(columns={df.columns[0]: 'Metric'}, inplace=True)

    long_df = df.melt(id_vars=['Metric'], var_name='Date', value_name='Value').dropna(subset=['Value'])
    # Keep only parseable dates (ignore 'TTM' etc.)
    long_df['Date'] = pd.to_datetime(long_df['Date'], errors='coerce')
    long_df = long_df.dropna(subset=['Date'])

    tidy = long_df.pivot_table(index='Date', columns='Metric', values='Value', aggfunc='first').reset_index()

    # Coerce numeric for all metric columns
    for c in tidy.columns:
        if c != 'Date':
            tidy[c] = pd.to_numeric(tidy[c], errors='coerce')

    return tidy.sort_values('Date')


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a NaN series of proper length."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def calculate_quarterly_fundamentals(financials_folder: str, ticker: str) -> pd.DataFrame:
    """
    Reads a single Excel workbook named like '{ticker.lower()}-financials.xlsx'
    and merges the four *Quarterly* sheets, then computes derived ratios.
    """
    xlsx_path = Path(financials_folder) / f"{ticker.lower()}-financials.xlsx"
    if not xlsx_path.exists():
        # Using concise logging for bulk processing
        return pd.DataFrame()

    xls = pd.ExcelFile(xlsx_path)
    required = ["Balance-Sheet-Quarterly", "Cash-Flow-Quarterly", "Income-Quarterly", "Ratios-Quarterly"]
    missing = [s for s in required if s not in xls.sheet_names]
    if missing:
        return pd.DataFrame()

    bal_q = _tidy_financial_sheet(pd.read_excel(xlsx_path, sheet_name="Balance-Sheet-Quarterly"))
    cfs_q = _tidy_financial_sheet(pd.read_excel(xlsx_path, sheet_name="Cash-Flow-Quarterly"))
    inc_q = _tidy_financial_sheet(pd.read_excel(xlsx_path, sheet_name="Income-Quarterly"))
    rat_q = _tidy_financial_sheet(pd.read_excel(xlsx_path, sheet_name="Ratios-Quarterly"))

    fundamentals = (
        bal_q.merge(cfs_q, on="Date", how="left")
        .merge(inc_q, on="Date", how="left")
        .merge(rat_q, on="Date", how="left")
        .sort_values('Date')
        .reset_index(drop=True)
    )

    # Derived fields (safe division + NaN handling)
    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        return a.divide(b).replace([np.inf, -np.inf], np.nan)

    fundamentals['Cash Flow/Assets'] = safe_div(_safe_series(fundamentals, 'Operating Cash Flow'),
                                                _safe_series(fundamentals, 'Total Assets'))

    # Book/Price = 1 / PB Ratio
    pb = _safe_series(fundamentals, 'PB Ratio')
    fundamentals['Book/Price'] = (1.0 / pb).replace([np.inf, -np.inf], np.nan)

    fundamentals['Earnings/Price'] = safe_div(_safe_series(fundamentals, 'Net Income'),
                                              _safe_series(fundamentals, 'Market Cap'))

    fundamentals['Forecast Earnings/Price'] = np.nan  # not available here

    fundamentals['Sales/Assets'] = safe_div(_safe_series(fundamentals, 'Revenue'),
                                            _safe_series(fundamentals, 'Total Assets'))

    fundamentals['Debt/Assets'] = safe_div(_safe_series(fundamentals, 'Total Debt'),
                                           _safe_series(fundamentals, 'Total Assets'))

    fundamentals['Debt/Equity'] = safe_div(_safe_series(fundamentals, 'Total Debt'),
                                           _safe_series(fundamentals, 'Shareholders Equity'))

    # Final column selection
    keep = [
        'Date', 'Cash Flow/Assets', 'Book/Price', 'Earnings/Price', 'Forecast Earnings/Price',
        'Sales/Assets', 'Debt/Assets', 'Debt/Equity'
    ]
    opt = ['Dividend Yield', 'Return on Assets (ROA)', 'Return on Equity (ROE)']
    for c in opt:
        if c in fundamentals.columns:
            keep.append(c)

    out = fundamentals[keep].copy()
    out.rename(columns={
        'Return on Assets (ROA)': 'Return on Assets',
        'Return on Equity (ROE)': 'Return on Equity'
    }, inplace=True)

    return out.sort_values('Date')


# ----------------------------
# Orchestration
# ----------------------------
def process_all_companies(history_folder: str, financials_folder: str, output_folder: str) -> None:
    vol_dir = Path(output_folder) / 'volatility_signals'
    fund_dir = Path(output_folder) / 'company_fundamentals'
    vol_dir.mkdir(parents=True, exist_ok=True)
    fund_dir.mkdir(parents=True, exist_ok=True)

    history_files = list_history_files(history_folder)

    if not history_files:
        print(
            "[ERROR] No history files found. Expected patterns include:\n"
            "  *-history.xlsx\n"
            "  *-history.csv\n"
            "  *-history.xlsx - Export.csv"
        )
        return

    print(f"[INFO] Found {len(history_files)} history files.")

    for file_path in tqdm(history_files, desc="Processing Companies"):
        file = file_path.name
        # Ticker = prefix before first hyphen (e.g., AAPL-history.xlsx -> AAPL)
        ticker = file.split('-')[0].upper()

        # --- Volatility signals ---
        try:
            hist_df = load_history_file(file_path)
            hist_with = calculate_volatility_signals(hist_df)
            out_vol = vol_dir / f"{ticker}-volatility.csv"
            hist_with.to_csv(out_vol, index=False)
        except Exception as e:
            # Silently skip missing files during bulk processing to avoid log spam
            continue

        # --- Fundamentals (Excel workbook with quarterly sheets) ---
        try:
            fund_df = calculate_quarterly_fundamentals(financials_folder, ticker)
            if not fund_df.empty:
                out_fund = fund_dir / f"{ticker}-fundamentals.csv"
                fund_df.to_csv(out_fund, index=False)
        except Exception as e:
            continue

    print(f"\n[OK] Processing Complete. Outputs written to:\n  {vol_dir}\n  {fund_dir}")


# ----------------------------
# Main execution block
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute financial signals for FinTradeBench.")
    parser.add_argument("--history-dir", type=str, default="./data/history",
                        help="Path to folder containing raw OHLCV history files.")
    parser.add_argument("--financials-dir", type=str, default="./data/financial_raw",
                        help="Path to folder containing raw SEC fundamental excel workbooks.")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Path to save the computed signal CSVs.")

    args = parser.parse_args()

    HISTORY_FOLDER = Path(args.history_dir)
    FINANCIALS_FOLDER = Path(args.financials_dir)
    OUTPUT_FOLDER = Path(args.output_dir)

    print("--- Starting Financial Signal Precomputation ---")
    print(f"History Folder:    {HISTORY_FOLDER.resolve()}")
    print(f"Financials Folder: {FINANCIALS_FOLDER.resolve()}")
    print(f"Output Folder:     {OUTPUT_FOLDER.resolve()}")
    print("----------------------------------------------")

    if not HISTORY_FOLDER.exists():
        print(f"[WARNING] History directory does not exist: {HISTORY_FOLDER}")
    if not FINANCIALS_FOLDER.exists():
        print(f"[WARNING] Financials directory does not exist: {FINANCIALS_FOLDER}")

    process_all_companies(
        history_folder=str(HISTORY_FOLDER),
        financials_folder=str(FINANCIALS_FOLDER),
        output_folder=str(OUTPUT_FOLDER)
    )