#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinTradeBench SEC Filings Downloader.

This script fetches raw corporate filings (10-K, 10-Q) from the SEC EDGAR
database for the companies included in the FinTradeBench dataset.
It strictly adheres to SEC rate limits (max 10 requests/second).

Usage (terminal):
    python download_sec_filings.py \
        --output-dir ./data/sec_filings \
        --years-back 10 \
        --user-agent "Reviewer Name reviewer@university.edu"
"""

import os
import re
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# ---------------------------
# DEFAULT CONFIGURATION
# ---------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "COST", "NFLX", "TMUS", "ASML", "CSCO", "AZN", "LIN", "PEP", "PLTR", "ADBE",
    "ISRG", "QCOM", "AMGN", "TXN", "INTU", "PDD", "AMD", "BKNG", "GILD", "CMCSA",
    "HON", "ARM", "AMAT", "VRTX", "ADP", "SBUX", "PANW", "ADI", "MU", "MELI",
    "LRCX", "KLAC", "APP", "INTC", "MDLZ", "ABNB", "CRWD", "CTAS", "FTNT",
    "REGN", "ORLY", "DASH", "MSTR", "MAR", "PYPL", "SNPS", "WDAY", "CDNS",
    "CEG", "ROP", "TEAM", "MRVL", "CHTR", "CSX", "PCAR", "NXPI", "AEP", "ADSK",
    "PAYX", "MNST", "CPRT", "FAST", "KDP", "ROST", "EXC", "BKR", "VRSK",
    "LULU", "CTSH", "FANG", "AXON", "GEHC", "XEL", "KHC", "ODFL", "CCEP",
    "DDOG", "EA", "TTWO", "IDXX", "CSGP", "TTD", "MCHP", "ZS", "DXCM", "ANSS",
    "WBD", "CDW", "GFS", "BIIB", "ON", "MDB"
]

# SEC ENDPOINTS
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Global HTTP Sessions
SESSION_DATA = requests.Session()
SESSION_WWW = requests.Session()

# ---------------------------
# HELPERS
# ---------------------------
def setup_sessions(user_agent: str):
    """Initializes the request sessions with the required SEC User-Agent."""
    SESSION_DATA.headers.update({
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })

    SESSION_WWW.headers.update({
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", name).strip()


def get_json(url: str, retries: int = 5, backoff: float = 1.8) -> dict:
    sess = SESSION_DATA if "data.sec.gov" in url else SESSION_WWW
    for attempt in range(1, retries + 1):
        resp = sess.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff ** attempt + (0.1 * attempt))
            continue
        raise RuntimeError(f"GET {url} failed: {resp.status_code} {resp.text[:200]}")
    raise RuntimeError(f"GET {url} failed after retries")


def download_file(url: str, out_path: Path, retries: int = 5, backoff: float = 1.8) -> None:
    sess = SESSION_DATA if "data.sec.gov" in url else SESSION_WWW
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        resp = sess.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            return
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff ** attempt + (0.1 * attempt))
            continue
        raise RuntimeError(f"DOWNLOAD {url} failed: {resp.status_code} {resp.text[:200]}")
    raise RuntimeError(f"DOWNLOAD {url} failed after retries")


def load_ticker_to_cik_map() -> Dict[str, str]:
    """Loads SEC's ticker->CIK mapping (CIK returned as string without leading zeros)."""
    data = get_json(TICKER_CIK_URL)
    mapping = {}
    for _, row in data.items():
        ticker = row["ticker"].upper()
        cik = str(row["cik_str"])
        mapping[ticker] = cik
    return mapping


def cik_pad(cik: str) -> str:
    return cik.zfill(10)


def accession_nodashes(accession: str) -> str:
    return accession.replace("-", "")


def within_years_back(filing_date_str: str, years_back: int) -> bool:
    """Return True if filing_date >= (today - years_back)."""
    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = cutoff.replace(year=cutoff.year - years_back)
    return filing_date >= cutoff


def build_filing_doc_url(cik: str, accession: str, primary_doc: str) -> str:
    """Constructs the exact download URL for the SEC Archive."""
    acc_no_dash = accession_nodashes(accession)
    return f"{ARCHIVES_BASE}/{int(cik)}/{acc_no_dash}/{primary_doc}"


# ---------------------------
# CORE LOGIC
# ---------------------------
def list_recent_filings(submissions_json: dict) -> List[dict]:
    """Convert submissions_json['filings']['recent'] parallel arrays into list of dicts."""
    recent = submissions_json.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    out = []
    n = len(forms)
    for i in range(n):
        out.append({
            "form": forms[i],
            "accession": accs[i],
            "filingDate": dates[i],
            "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            "reportDate": report_dates[i] if i < len(report_dates) else "",
        })
    return out


def download_filings_for_ticker(ticker: str, cik: str, out_base: Path, years_back: int, forms_to_keep: set, pause_sec: float) -> Tuple[int, int]:
    """Downloads requested forms for the past `years_back` years for one ticker."""
    ticker_dir = out_base / safe_filename(ticker.upper())
    ticker_dir.mkdir(parents=True, exist_ok=True)

    submissions_url = SUBMISSIONS_URL.format(cik_padded=cik_pad(cik))
    subs = get_json(submissions_url)

    filings = list_recent_filings(subs)

    downloaded = 0
    skipped = 0

    for f in filings:
        form = f["form"].upper().strip()
        if form not in forms_to_keep:
            continue

        filing_date = f["filingDate"]
        if not filing_date or not within_years_back(filing_date, years_back):
            continue

        accession = f["accession"]
        primary_doc = f.get("primaryDocument", "")

        # Skip amendments or missing primary documents
        if not primary_doc:
            continue

        doc_url = build_filing_doc_url(cik, accession, primary_doc)

        ext = Path(primary_doc).suffix or ".htm"
        out_name = f"{form}_{filing_date}_{accession}{ext}"
        out_path = ticker_dir / safe_filename(out_name)

        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            download_file(doc_url, out_path)
            downloaded += 1
            print(f"[{ticker}] downloaded {form} {filing_date} -> {out_name}")
        except Exception as e:
            skipped += 1
            print(f"[{ticker}] SKIP {form} {filing_date} ({accession}): {e}")

        time.sleep(pause_sec)

    return downloaded, skipped


def main(args):
    if "@" not in args.user_agent or " " not in args.user_agent.strip():
        print("[WARNING] SEC requires a descriptive User-Agent (e.g., 'Name email@domain.com'). Your requests may be blocked.")

    setup_sessions(args.user_agent)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker_to_cik = load_ticker_to_cik_map()
    forms_set = set(f.strip().upper() for f in args.forms.split(','))

    # Determine tickers to process
    if args.tickers.upper() == "DEFAULT":
        tickers_to_process = DEFAULT_TICKERS
    else:
        tickers_to_process = [t.strip().upper() for t in args.tickers.split(',')]

    total_dl = 0
    total_skip = 0

    print(f"--- Starting SEC EDGAR Download for FinTradeBench ---")
    print(f"Target Directory: {out_dir.resolve()}")
    print(f"Forms: {forms_set}")
    print(f"Lookback: {args.years_back} years")
    print(f"Tickers to process: {len(tickers_to_process)}")
    print("---------------------------------------------------")

    for ticker in tickers_to_process:
        if not ticker: continue

        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"[{ticker}] not found in SEC ticker map -> skipping")
            continue

        print(f"\n=== Processing {ticker} (CIK {cik}) ===")
        dl, sk = download_filings_for_ticker(
            ticker=ticker,
            cik=cik,
            out_base=out_dir,
            years_back=args.years_back,
            forms_to_keep=forms_set,
            pause_sec=args.pause_seconds
        )
        total_dl += dl
        total_skip += sk

    print(f"\n[OK] Done. Downloaded: {total_dl}, Skipped/Existing/Failed: {total_skip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC 10-K and 10-Q filings for FinTradeBench.")
    parser.add_argument("--output-dir", type=str, default="./data/sec_filings",
                        help="Base folder to save the downloaded filings.")
    parser.add_argument("--years-back", type=int, default=10,
                        help="Number of years of historical filings to retrieve.")
    parser.add_argument("--forms", type=str, default="10-K,10-Q",
                        help="Comma-separated list of SEC forms to download.")
    parser.add_argument("--pause-seconds", type=float, default=0.25,
                        help="Pause between SEC requests to avoid rate limits (Max 10 requests/sec allowed).")
    parser.add_argument("--tickers", type=str, default="DEFAULT",
                        help="Comma-separated list of tickers, or 'DEFAULT' to use the internal FinTradeBench list.")
    parser.add_argument("--user-agent", type=str, default="Anonymous Researcher anon.researcher@university.edu",
                        help="Mandatory User-Agent string for SEC EDGAR API (Format: 'Name Email').")

    args = parser.parse_args()
    main(args)