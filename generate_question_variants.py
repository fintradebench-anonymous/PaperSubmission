#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate benchmarking question variants by swapping company names (from a NASDAQ-100 alias map)
and dates/time periods (bounded between 2015 and mid-2025), while preserving original formatting.

CHANGELOG
- Synchronize tickers when a company name gets replaced, e.g., "Apple (AAPL)" -> "Tesla (TSLA)".
-----------------------------------------------------------------------------
Usage (terminal):
    python generate_question_variants.py --input question.csv --output question_variants_generated.csv \
        --variants-per-question 5 --column-name question --seed 42

Optional:
    --alias-json aliases.json       # supply your own mapping at runtime (keys -> tickers)
    --keep-company                  # don't swap company names; only randomize dates
    --keep-dates                    # don't swap dates; only randomize companies
    --no-sync-tickers               # do NOT sync "(TICKER)" after company names
    --start-year 2015 --end-year 2025 --end-date-limit 2025-06-30
"""

import argparse
import json
import random
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# --------------------------- Default Alias Map ---------------------------
DEFAULT_STATIC_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "broadcom": "AVGO",
    "intel": "INTC",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "paypal": "PYPL",
    "pepsico": "PEP",
    "coca cola": "KO",
    "coca-cola": "KO",
    "airbnb": "ABNB",
    "qualcomm": "QCOM",
    "autodesk": "ADSK",
    "costco": "COST",
    "comcast": "CMCSA",
    "charter": "CHTR",
    "booking": "BKNG",
    "starbucks": "SBUX",
    "intuit": "INTU",
    "marvell": "MRVL",
    "cdw": "CDW",
    "crowdstrike": "CRWD",
    "pinduoduo": "PDD",
    "dexcom": "DXCM",
    "idexx": "IDXX",
    "t-mobile": "TMUS",
    "micron": "MU",
    "illumina": "ILMN",
    "biogen": "BIIB",
    "atlanssian": "TEAM",
    "cadence": "CDNS",
    "mondelez": "MDLZ",
    "lam research": "LRCX",
    "workday": "WDAY",
    "zs": "ZS",
    "zoom": "ZM",
    "ebay": "EBAY",
    "american electric power": "AEP",
    "amgen": "AMGN",
    "analog devices": "ADI",
    "asml": "ASML",
    "astrazeneca": "AZN",
    "roper": "ROP",
    "exelon": "EXC",
    "regeneron": "REGN",
    "chipotle": "CMG",
    "copart": "CPRT",
    "fastenal": "FAST",
    "monster beverage": "MNST",
    "vertex": "VRTX",
    "verisk": "VRSK",
    "marriott": "MAR",
    "lululemon": "LULU",
    "match group": "MTCH",
    "old dominion": "ODFL",
    "paychex": "PAYX",
    "ross stores": "ROST",
    "sbac": "SBAC",
    "synopsys": "SNPS",
    "moderna": "MRNA",
    "nasdaq": "NDAQ",
    "nxp": "NXPI",
    "oreilly automotive": "ORLY",
    "sanofi": "SNY",
    "verizon": "VZ",
    "willis towers watson": "WTW",
    "xcel energy": "XEL",
    "texas instruments": "TXN",
    "micron technology": "MU",
    "paccar": "PCAR",
    "kraft heinz": "KHC",
    "fifth third bancorp": "FITB",
    "fiserv": "FISV",
    "gilead": "GILD",
    "klac": "KLAC",
    "biontech": "BNTX",
    "ericsson": "ERIC",
    "sbac communications": "SBAC",
    "baxter international": "BAX",
    "cintas": "CTAS",
    "nasdaq inc": "NDAQ",
    "roper technologies": "ROP",
    "cdw corporation": "CDW",
    "trade desk": "TTD"
}


# --------------------------- Core Logic ---------------------------

def build_company_regex(alias_map: Dict[str, str], sync_tickers: bool) -> str:
    names = sorted(set([k.strip().lower() for k in alias_map.keys() if isinstance(k, str)]))
    names_sorted = sorted(names, key=lambda x: len(x), reverse=True)
    base = r"(" + "|".join([re.escape(n) for n in names_sorted]) + r")"
    if sync_tickers:
        # Match an optional ticker immediately following the company name, e.g., "Apple (AAPL)"
        pattern = rf"\b{base}\b(\s*\(([A-Z.\-]{{1,6}})\))?"
    else:
        pattern = rf"\b{base}\b"
    return pattern


def normalize_col(df: pd.DataFrame, user_col: str = None) -> str:
    if user_col and user_col in df.columns:
        return user_col
    # otherwise pick the first column as questions
    return df.columns[0]


def random_year(start_year: int, end_year: int) -> int:
    return random.randint(start_year, end_year)


def random_year_for_date(month: int, day: int, start_year: int, end_year: int, end_date_limit: date) -> int:
    while True:
        y = random_year(start_year, end_year)
        try:
            d = date(y, month, day)
        except ValueError:
            # handle Feb 29; fallback to Feb 28 on non-leap years
            if month == 2 and day == 29:
                try:
                    d = date(y, 2, 29)
                except ValueError:
                    d = date(y, 2, 28)
            else:
                continue
        if d <= end_date_limit:
            return d.year


def make_date_patterns(start_year: int, end_year: int) -> Dict[str, re.Pattern]:
    months_regex = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    month_abbr_regex = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"

    return {
        "range_dash": re.compile(rf"\b(20(1[5-9]|2[0-9]))\s*[-–]\s*(20(1[5-9]|2[0-9]))\b"),
        "range_words": re.compile(rf"\b(20(1[5-9]|2[0-9]))\s*(to|through|thru|and)\s*(20(1[5-9]|2[0-9]))\b", re.IGNORECASE),
        "quarter": re.compile(rf"\bQ([1-4])\s*(20(1[5-9]|2[0-9]))\b", re.IGNORECASE),
        "ymd": re.compile(r"\b(20(1[5-9]|2[0-9]))-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),
        "month_year_full": re.compile(rf"\b{months_regex}\s+(20(1[5-9]|2[0-9]))\b", re.IGNORECASE),
        "month_year_abbr": re.compile(rf"\b{month_abbr_regex}\s+(20(1[5-9]|2[0-9]))\b", re.IGNORECASE),
        "year": re.compile(r"\b(20(1[5-9]|2[0-9]))\b")
    }


def replace_date_tokens(text: str, start_year: int, end_year: int, end_date_limit: date, patterns: Dict[str, re.Pattern]) -> Tuple[str, List[Tuple[str, str]]]:
    replacements: List[Tuple[str, str]] = []

    # Ranges with dash
    def sub_range_dash(m):
        y1 = int(m.group(1))
        y2 = int(m.group(3))
        if y2 < y1:
            y1, y2 = y2, y1
        ny1 = random.randint(start_year, end_year - 1)
        ny2 = random.randint(ny1 + 1, end_year)
        old = f"{y1}-{y2}"
        new = f"{ny1}-{ny2}"
        replacements.append((old, new))
        return new

    text = patterns["range_dash"].sub(sub_range_dash, text)

    # Ranges with words
    def sub_range_words(m):
        y1 = int(m.group(1))
        y2 = int(m.group(4))
        conj = m.group(3)
        if y2 < y1:
            y1, y2 = y2, y1
        ny1 = random.randint(start_year, end_year - 1)
        ny2 = random.randint(ny1 + 1, end_year)
        old = f"{y1} {conj} {y2}"
        new = f"{ny1} {conj} {ny2}"
        replacements.append((old, new))
        return new

    text = patterns["range_words"].sub(sub_range_words, text)

    # Quarter-year
    def sub_quarter(m):
        q = m.group(1)
        y = int(m.group(2))
        ny = random.randint(start_year, end_year)
        old = f"Q{q} {y}"
        new = f"Q{q} {ny}"
        replacements.append((old, new))
        return new

    text = patterns["quarter"].sub(sub_quarter, text)

    # YYYY-MM-DD
    def sub_ymd(m):
        y = int(m.group(1))
        mo = int(m.group(3))
        d = int(m.group(4))
        ny = random_year_for_date(mo, d, start_year, end_year, end_date_limit)
        old = f"{y:04d}-{mo:02d}-{d:02d}"
        new = f"{ny:04d}-{mo:02d}-{d:02d}"
        replacements.append((old, new))
        return new

    text = patterns["ymd"].sub(sub_ymd, text)

    # Month Year (full)
    def sub_month_year_full(m):
        month_name = m.group(1)
        y = int(m.group(2))
        ny = random.randint(start_year, end_year)
        old = f"{month_name} {y}"
        new = f"{month_name} {ny}"
        replacements.append((old, new))
        return new

    text = patterns["month_year_full"].sub(sub_month_year_full, text)

    # Month Year (abbr)
    def sub_month_year_abbr(m):
        mon = m.group(1)
        y = int(m.group(2))
        ny = random.randint(start_year, end_year)
        old = f"{mon} {y}"
        new = f"{mon} {ny}"
        replacements.append((old, new))
        return new

    text = patterns["month_year_abbr"].sub(sub_month_year_abbr, text)

    # Single Year
    def sub_year(m):
        y = int(m.group(1))
        ny = random.randint(start_year, end_year)
        old = str(y)
        new = str(ny)
        replacements.append((old, new))
        return new

    text = patterns["year"].sub(sub_year, text)

    return text, replacements


def replace_company(text: str, company_regex: str, company_names: List[str], alias_map: Dict[str, str], sync_tickers: bool) -> Tuple[str, List[Tuple[str, str]]]:
    replaced_pairs: List[Tuple[str, str]] = []

    def sub_company(m):
        old_name = m.group(1)  # the matched alias
        old_ticker_group = m.group(3) if sync_tickers and m.lastindex and m.lastindex >= 3 else None  # ticker without parens

        # Choose a different company name than the matched one
        choices = [c for c in company_names if c != old_name.lower()]
        if not choices:
            return m.group(0)
        new_name = random.choice(choices)
        # Preserve capitalization style of the original token
        if old_name.isupper():
            new_name_text = new_name.upper()
        elif old_name.istitle():
            new_name_text = new_name.title()
        else:
            new_name_text = new_name

        # Build replacement text
        if sync_tickers and old_ticker_group is not None:
            # There was a "(TICKER)" right after the company; swap it to the correct ticker
            new_ticker = alias_map.get(new_name, alias_map.get(new_name.lower(), ""))
            # Ensure uppercase ticker
            new_ticker = (new_ticker or "").upper()
            # Compose: <New Name> (NEWTICKER)
            repl = f"{new_name_text} ({new_ticker})" if new_ticker else f"{new_name_text}"
        else:
            repl = new_name_text

        replaced_pairs.append((m.group(0), repl))
        return repl

    return re.sub(company_regex, sub_company, text, flags=re.IGNORECASE), replaced_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate variants by swapping company names and dates.")
    parser.add_argument("--input", required=True, help="Path to input CSV with a questions column.")
    parser.add_argument("--output", required=True, help="Path to write the output CSV.")
    parser.add_argument("--column-name", default=None, help="Name of the column containing questions. Defaults to first column.")
    parser.add_argument("--variants-per-question", type=int, default=5, help="Number of variants per question.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--alias-json", default=None, help="Optional JSON file with alias mapping (keys->tickers).")
    parser.add_argument("--keep-company", action="store_true", help="If set, do NOT change company names (only dates).")
    parser.add_argument("--keep-dates", action="store_true", help="If set, do NOT change dates (only companies).")
    parser.add_argument("--no-sync-tickers", action="store_true", help="If set, do NOT sync '(TICKER)' after company names.")
    parser.add_argument("--start-year", type=int, default=2015, help="Lower bound year for replacements.")
    parser.add_argument("--end-year", type=int, default=2025, help="Upper bound year for replacements.")
    parser.add_argument("--end-date-limit", default="2025-06-30", help="Cap for exact dates (YYYY-MM-DD).")

    args = parser.parse_args()

    random.seed(args.seed)

    # Load questions
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    col = normalize_col(df, args.column_name)
    if col not in df.columns:
        print(f"[ERROR] Column '{col}' not found in the CSV.", file=sys.stderr)
        sys.exit(1)

    # Prepare alias map
    if args.alias_json:
        with open(args.alias_json, "r", encoding="utf-8") as f:
            alias_map = json.load(f)
    else:
        alias_map = DEFAULT_STATIC_ALIASES

    company_names = sorted(set([k.strip().lower() for k in alias_map.keys()]))
    sync_tickers = not args.no_sync_tickers
    company_regex = build_company_regex(alias_map, sync_tickers=sync_tickers)

    # Date bounds
    try:
        end_date_limit = datetime.strptime(args.end_date_limit, "%Y-%m-%d").date()
    except Exception:
        print(f"[ERROR] end-date-limit must be in YYYY-MM-DD format (got {args.end_date_limit})", file=sys.stderr)
        sys.exit(1)

    if args.start_year > args.end_year:
        print("[ERROR] start-year must be <= end-year", file=sys.stderr)
        sys.exit(1)

    patterns = make_date_patterns(args.start_year, args.end_year)

    # Generate variants
    rows = []
    total = 0

    for idx, row in df.iterrows():
        base_q = str(row[col]).strip()
        if not base_q:
            continue
        for v in range(1, args.variants_per_question + 1):
            q = base_q
            company_repls: List[Tuple[str, str]] = []
            date_repls: List[Tuple[str, str]] = []

            if not args.keep_company:
                q, company_repls = replace_company(q, company_regex, company_names, alias_map, sync_tickers)
            if not args.keep_dates:
                q, date_repls = replace_date_tokens(q, args.start_year, args.end_year, end_date_limit, patterns)

            rows.append({
                "original_index": idx,
                "variant_id": v,
                "original_question": base_q,
                "new_question": q,
                "company_replacements": "; ".join([f"{o} -> {n}" for o, n in company_repls]) if company_repls else "",
                "date_replacements": "; ".join([f"{o} -> {n}" for o, n in date_repls]) if date_repls else ""
            })
            total += 1

    out_df = pd.DataFrame(rows, columns=[
        "original_index",
        "variant_id",
        "original_question",
        "new_question",
        "company_replacements",
        "date_replacements"
    ])
    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Wrote {total} variants to: {out_path}")
    print(f"[INFO] Variants per question: {args.variants_per_question}")
    print(f"[INFO] Start year: {args.start_year}, End year: {args.end_year}, Date cap: {args.end_date_limit}")
    if args.keep_company:
        print("[INFO] Company swapping disabled (--keep-company).")
    if args.keep_dates:
        print("[INFO] Date swapping disabled (--keep-dates).")
    if not sync_tickers:
        print("[INFO] Ticker synchronization disabled (--no-sync-tickers).")


if __name__ == "__main__":
    main()