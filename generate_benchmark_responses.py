#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Benchmark Responses using TELeR Prompts and LLM-as-a-Judge Auditing.
(Multi-Model, Resumable Pipeline)

This script processes a set of financial reasoning queries, retrieves the
relevant precomputed Trading and Fundamental data, and evaluates
LLMs using the TELeR prompt taxonomy. It supports Vertex AI Native models 
(Gemini) and MaaS REST endpoints (Qwen, DeepSeek, etc.).

Usage (terminal):
    python generate_benchmark_responses.py \
        --project-id YOUR_GCP_PROJECT_ID \
        --model "gemini-3.1-pro-preview" \
        --combined-dir ./data/combined \
        --questions-file ./data/questions.csv \
        --output-file ./output/my_results.csv
"""

import os
import re
import csv
import threading
import unicodedata
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import random
import requests

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# --- Auth & Vertex AI Imports ---
import google.auth
from google.auth.transport.requests import Request
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

# =========================
# CONFIG & FILE CONSTANTS
# =========================

DEFAULT_OUTPUT_FILE = f"benchmark_results_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv"

# --- Prompts ---
PROMPTS_TO_EVALUATE = {
    "L1_Baseline": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 1)
Use the provided 'Context' from financial documents as your primary source of truth and answer the given question.

Trading Signals Context:
{trading_context_str}

Fundamental Data Context:
{fundamental_context_str}

Question:
{query_str}

Answer:
""",

    "TELER_L2_Strict_Focus": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 2)
You are a senior financial analyst. Break down the question and answer the question methodically. First, explain what the question is asking and the main goal of the question clearly.
Then, identify the embedded sub-questions in the given question. Answer each part using only the provided context. Finally, combine sub-answers into a single, comprehensive final answer and answer in a professional style.
The context for trading signals is provided as Trading Signals Context: {trading_context_str} and context for Fundamental Data is provided as {fundamental_context_str}, and the question is given as {query_str}.
Your analysis should clearly distinguish between the reasoning process for your answers for each embedded sub question and the conclusion in the final answer.
""",

    "TELER_L3_Step_By_Step": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 3)
You are a senior financial analyst. Break down and answer the question methodically:
1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part using *both* the 'Trading Signals Context' and 'Fundamental Data Context'.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').

Trading Signals Context:
{trading_context_str}

Fundamental Data Context:
{fundamental_context_str}

Question:
{query_str}

Explanation & Justification:
[Detailed rationale referencing both signal types]

Final Answer:
[Provide final answer]
""",

    "TELER_L4_Auditor_Evidence": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 4)
You are a senior financial analyst. Break down and answer the question methodically:
1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part using *both* the 'Trading Signals Context' and 'Fundamental Data Context'.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

Trading Signals Context:
{trading_context_str}

Fundamental Data Context:
{fundamental_context_str}

Question:
{query_str}

Explanation & Justification:
[Detailed rationale referencing both signal types]

Final Answer:
[Provide final answer]
""",

    "TELER_L5_Deconstruction": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 5)
You are a senior financial analyst. Break down and answer the question methodically:
1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part using *both* the 'Trading Signals Context' and 'Fundamental Data Context'.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.
5. **Support:** Cite exact supporting text or evidence retrieved from the given contexts.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

Trading Signals Context:
{trading_context_str}

Fundamental Data Context:
{fundamental_context_str}

Question:
{query_str}

Supporting Evidence:
- "[Quote from Trading Context]"
- "[Quote from Fundamental Context]"

Explanation & Justification:
[Detailed rationale referencing both signal types]

Final Answer:
[Provide final answer]
""",

    "TELER_L6_Maximalist": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 6)
You are a senior financial analyst. Break down and answer the question methodically:

1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part using *both* the 'Trading Signals Context' and 'Fundamental Data Context'.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.
5. **Support:** Cite exact supporting text or evidence retrieved from the given contexts.
6. **Justify:** Justify why information from *both* contexts is included or excluded and explain your reasoning.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

Trading Signals Context:
{trading_context_str}

Fundamental Data Context:
{fundamental_context_str}

Question:
{query_str}

Supporting Evidence:
- "[Quote from Trading Context]"
- "[Quote from Fundamental Context]"

Explanation & Justification:
[Detailed rationale referencing both signal types]

Final Answer:
[Comprehensive, well-justified answer]
"""
}

# --- Self-Selection Prompt ---
SELF_SELECTION_PROMPT_TEMPLATE = """You are a financial analyst. You have been given a single Question and several Candidate Answers generated by an AI.
Your task is to select the single *best* answer that is the most accurate, complete, and relevant to the question.

Question:
{question}

--- CANDIDATE ANSWERS ---

{candidate_answers_str}

---
INSTRUCTIONS:
Review all candidate answers. Respond with *only* the Prompt ID of the answer you choose (e.g., "TELER_L3_Step_By_Step").
Do not add any other text or explanation.

Your Choice (Prompt ID only):
"""

# --- Numerical Auditor Prompt ---
AUDITOR_PROMPT_TEMPLATE = """
You are a meticulous financial auditor. Your task is to fact-check an analyst's response against a set of "Ground Truth Data."
You must evaluate every single numerical claim in the "Analyst Response."

Here is the data you must use:

--- GROUND TRUTH DATA ---
[Trading Context]
{trading_context}

[Fundamental Context]
{fundamental_context}
--- END GROUND TRUTH DATA ---


Here is the analyst's response you must audit:

--- ANALYST RESPONSE ---
{response}
--- END ANALYST RESPONSE ---

Perform the following steps:
1.  Read the "Analyst Response" and identify every specific numerical claim.
2.  For each claim, check the "Ground Truth Data" to see if it is supported.
3.  Generate a final "Audit Report" in a clear, bulleted list. For each claim, state "SUPPORTED," "CONTRADICTED," or "NOT_FOUND."
4.  Finally, add a "Summary" line:
    - "Summary: All claims are SUPPORTED." (if all are supported)
    - "Summary: Contains CONTRADICTED or NOT_FOUND claims." (if any errors exist)

Provide ONLY the audit report.

Audit Report:
"""

# --- Output Header ---
OUTPUT_HEADER = [
    "question_id", "question_type", "question", "golden_indicators", "model",
    "prompt_id", "response", "response_time_sec", "type",
    "is_self_selected", "self_selection_choice",
    "is_numerically_accurate", "numerical_audit_report",
    "trading_context", "fundamental_context",
    "tickers_used", "date_start", "date_end", "date_label"
]

# --- Constants ---
DEFAULT_LOOKBACK_DAYS = 180
TRADING_COLS = ['Adj. Close', 'MA_20', 'MACD', 'MACD_Signal', 'RSI', 'EMA_20', 'OBV', 'One_Day_Reversal',
                'Max_Return_20D', 'Momentum_5D', 'Momentum_20D', 'Mean_Reversal_60D', 'Short_Term_Reversal_1month',
                'Medium_Term_Momentum_2month_to_12month', 'Long_Term_Reversal_13month_to_60month']
FUNDAMENTALS_COLS = ['F_Cash Flow/Assets', 'F_Book/Price', 'F_Earnings/Price', 'F_Sales/Assets', 'F_Debt/Assets',
                     'F_Debt/Equity', 'F_Dividend Yield', 'F_Return on Assets', 'F_Return on Equity']

# Global Dataset End Date
DATASET_END_DATE = datetime(2025, 9, 1)

# =========================
# API Controllers & Universal Model Handler
# =========================
def init_vertex_ai(project_id: str, location: str):
    try:
        vertexai.init(project=project_id, location=location)
        print(f"Vertex AI initialized for project: {project_id}")
    except Exception as e:
        raise EnvironmentError(f"Failed to initialize Vertex AI: {e}")

def get_gcloud_auth_token() -> str:
    """Refreshes and returns the current Google Cloud access token."""
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    return credentials.token

class APIRateController:
    def __init__(self, max_rpm: int = 25, max_daily: int = 950):
        self.lock = threading.Lock()
        self.min_interval = 60.0 / max_rpm
        self.last_call_time = 0.0
        self.max_daily = max_daily
        self.daily_counter = 0
        self.stop_event = threading.Event()

    def wait_and_acquire(self):
        if self.stop_event.is_set():
            raise RuntimeError("Daily limit reached. Stopping execution.")
        with self.lock:
            if self.daily_counter >= self.max_daily:
                print(f"\n[LIMIT] Daily limit of {self.max_daily} requests reached! Stopping.")
                self.stop_event.set()
                raise RuntimeError("Daily limit reached.")
            now = time.monotonic()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.monotonic()
            self.daily_counter += 1
            return self.daily_counter

RATE_CONTROLLER = APIRateController(max_rpm=25, max_daily=950)

def get_model_response_universal(model_name: str, prompt: str, project_id: str, temperature: float = 0.0) -> str:
    """
    Universal wrapper that handles both Gemini (Native SDK) and MaaS models (REST API).
    """
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            try:
                RATE_CONTROLLER.wait_and_acquire()
            except RuntimeError:
                return "Error: Daily API limit reached."

            # --- BRANCH A: GEMINI (Native SDK) ---
            if "gemini" in model_name.lower():
                model = GenerativeModel(model_name)
                config = GenerationConfig(temperature=temperature, max_output_tokens=8192)
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
                response = model.generate_content(prompt, generation_config=config, safety_settings=safety_settings)
                try:
                    return response.text.strip()
                except ValueError:
                    return f"Error: Blocked by safety filters."

            # --- BRANCH B: DEEPSEEK / QWEN (MaaS via REST) ---
            else:
                region = "us-south1" if "qwen" in model_name.lower() else "us-central1"
                endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions"
                auth_token = get_gcloud_auth_token()
                headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 4096,
                    "stream": False
                }
                response = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
                if response.status_code != 200:
                    raise Exception(f"API Error {response.status_code}: {response.text}")
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    return f"Error: Unexpected JSON response."

        except google_exceptions.ResourceExhausted:
            wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: Failed to get response from {model_name} after retries. ({type(e).__name__}: {e})"
            time.sleep(base_delay * (2 ** attempt))

    return "Error: Max retries exceeded."

# =========================
# Data Loading & Analysis
# =========================
def load_data_frames(directory: str, suffix: str) -> Dict[str, pd.DataFrame]:
    store: Dict[str, pd.DataFrame] = {}
    for p in sorted(Path(directory).glob(f"*{suffix}")):
        ticker = p.name.replace(suffix, "").upper()
        try:
            df = pd.read_csv(p)
            df['Date'] = pd.to_datetime(df['Date'])
            store[ticker] = df.sort_values('Date').drop_duplicates(subset=['Date'])
        except Exception as e:
            print(f"[WARN] Failed to load {p.name}: {e}")
    if not store:
        raise FileNotFoundError(f"No files found in {directory} with suffix {suffix}")
    print(f"Loaded {len(store)} tickers from {directory}")
    return store

def load_questions_auto(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() != '.csv':
        raise ValueError("Unsupported questions file format. Please use CSV.")

    df = pd.read_csv(path)
    cols_norm = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=cols_norm, inplace=True)

    required = ['question_id', 'query', 'golden_indicators']
    if missing := [c for c in required if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=['query'])
    questions = []
    for i, r in df.iterrows():
        indicators_str = str(r.get('golden_indicators', '')).strip()
        indicators_list = [ind.strip() for ind in re.split(r'[|,]', indicators_str) if ind.strip()] if indicators_str else []
        questions.append({
            "id": str(r.get('question_id', f'q{i + 1}')),
            "type": str(r.get('question_type', '')),
            "question": str(r['query']),
            "golden_indicators": indicators_list
        })
    return questions

def get_ranking_parameters_from_question(q_text: str) -> Tuple[str, bool]:
    q_lower = q_text.lower()
    sort_ascending = any(kw in q_lower for kw in ['worst', 'bottom', 'lost', 'loser', 'losers', 'weakest', 'underperform', 'lowest', 'smallest'])
    metric_map = [
        (['return on equity', 'f_return on equity', 'f_roe', 'roe'], 'F_Return on Equity'),
        (['return on assets', 'f_return on assets', 'f_roa', 'roa'], 'F_Return on Assets'),
        (['dividend yield', 'f_dividend yield', 'yield', 'dividend'], 'F_Dividend Yield'),
        (['debt to equity', 'debt/equity', 'f_debt/equity', 'd/e'], 'F_Debt/Equity'),
        (['debt to assets', 'debt/assets', 'f_debt/assets'], 'F_Debt/Assets'),
        (['price to earnings', 'p/e', 'earnings/price', 'f_earnings/price', 'pe ratio'], 'F_Earnings/Price'),
        (['price to book', 'p/b', 'book/price', 'f_book/price', 'pb ratio'], 'F_Book/Price'),
        (['sales to assets', 'sales/assets', 'f_sales/assets'], 'F_Sales/Assets'),
        (['cash flow to assets', 'cash flow/assets', 'f_cash flow/assets'], 'F_Cash Flow/Assets'),
        (['long term reversal', '13-60 month reversal'], 'Long_Term_Reversal_13month_to_60month'),
        (['medium term momentum', '2-12 month momentum'], 'Medium_Term_Momentum_2month_to_12month'),
        (['short term reversal', '1 month reversal'], 'Short_Term_Reversal_1month'),
        (['mean reversal 60d', 'mean_reversal_60d'], 'Mean_Reversal_60D'),
        (['max return 20d', 'max_return_20d', 'volatility'], 'Max_Return_20D'),
        (['rsi', 'relative strength index'], 'RSI'),
        (['macd signal'], 'MACD_Signal'),
        (['macd'], 'MACD'),
        (['obv', 'on balance volume'], 'OBV'),
        (['20 day momentum', 'momentum_20d'], 'Momentum_20D'),
        (['5 day momentum', 'momentum_5d'], 'Momentum_5D'),
        (['percent change', 'pct_change', 'change', 'gained', 'lost', 'performance', 'return'], 'pct_change'),
        (['momentum'], 'Momentum_20D'),
    ]

    for keywords, metric_name in metric_map:
        if any(kw in q_lower for kw in keywords):
            return metric_name, sort_ascending
    if any(k in q_lower for k in ['lost', 'gained', 'winner', 'loser']):
        return 'pct_change', sort_ascending
    return 'Momentum_20D', sort_ascending

def rank_stocks_by_metric(data_frames: Dict[str, pd.DataFrame], start: datetime, end: datetime, metric: str = 'Momentum_20D', sort_ascending: bool = False) -> List[Tuple[str, float]]:
    p_start = pd.to_datetime(start).tz_localize(None) if pd.to_datetime(start).tz is not None else pd.to_datetime(start)
    p_end = pd.to_datetime(end).tz_localize(None) if pd.to_datetime(end).tz is not None else pd.to_datetime(end)

    results = []
    for ticker, df in data_frames.items():
        window = df[(df['Date'] >= p_start) & (df['Date'] <= p_end)].copy()
        if window.empty: continue

        value = None
        if metric == 'pct_change':
            close_series = window['Adj. Close'].dropna()
            if len(close_series) > 1:
                first_close, last_close = close_series.iloc[0], close_series.iloc[-1]
                if pd.notna(first_close) and pd.notna(last_close) and first_close != 0:
                    value = (last_close - first_close) / first_close * 100.0
        elif metric in window.columns:
            series = window[metric].dropna()
            if not series.empty:
                value = float(series.median()) if metric.startswith('F_') else float(series.iloc[-1])

        if value is not None and pd.notna(value):
            results.append((ticker, float(value)))

    results.sort(key=lambda x: x[1], reverse=not sort_ascending)
    return results

# =========================
# Retrieval & Context Building
# =========================
_TICKER_TOKEN = re.compile(r"\b[A-Z]{1,6}\b")
def extract_tickers_from_text(text: str) -> List[str]:
    return sorted(list(set(tok.upper() for tok in _TICKER_TOKEN.findall(text or ""))))

STATIC_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX", "adobe": "ADBE", "broadcom": "AVGO",
    "intel": "INTC", "advanced micro devices": "AMD", "amd": "AMD", "paypal": "PYPL", "pepsico": "PEP",
    "coca cola": "KO", "coca-cola": "KO", "airbnb": "ABNB", "qualcomm": "QCOM", "autodesk": "ADSK", "costco": "COST",
    "comcast": "CMCSA", "charter": "CHTR", "booking": "BKNG", "starbucks": "SBUX", "intuit": "INTU", "marvell": "MRVL",
    "cdw": "CDW", "crowdstrike": "CRWD", "pinduoduo": "PDD", "dexcom": "DXCM", "idexx": "IDXX", "t-mobile": "TMUS",
    "micron": "MU", "illumina": "ILMN", "biogen": "BIIB", "atlanssian": "TEAM", "cadence": "CDNS", "mondelez": "MDLZ",
    "lam research": "LRCX", "workday": "WDAY", "zs": "ZS", "zoom": "ZM", "ebay": "EBAY"
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[^a-z0-9\s\.-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

STATIC_GROUP_MAP = {
    _norm("fang stocks"): ["META", "AMZN", "NFLX", "GOOGL"],
    _norm("faang stocks"): ["META", "AMZN", "AAPL", "NFLX", "GOOGL"],
    _norm("manga stocks"): ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"],
    _norm("magnificent seven"): ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    _norm("semiconductor stocks"): ["NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI", "NXPI", "MRVL"],
    _norm("cybersecurity stocks"): ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
    _norm("cloud stocks"): ["ADBE", "INTU", "WDAY", "ADSK", "TEAM", "DDOG", "SNPS", "CDNS"],
    _norm("biotech stocks"): ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "AZN"],
}

def build_alias_index(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for ticker, df in data_frames.items():
        aliases[_norm(ticker)] = ticker
        for col in ["Company", "Name", "Issuer", "CompanyName"]:
            if col in df.columns:
                name_series = df[col].dropna().astype(str).map(_norm)
                if not name_series.empty:
                    name = name_series.value_counts().index[0]
                    if name and name not in aliases: aliases[name] = ticker
                break
    for k, v in STATIC_ALIASES.items(): aliases.setdefault(_norm(k), v)
    return aliases

def build_sector_index(data_frames: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    sector_index: Dict[str, List[str]] = {}
    for ticker, df in data_frames.items():
        if 'Sector' in df.columns:
            sector_series = df['Sector'].dropna()
            if not sector_series.empty:
                sector_norm = _norm(str(sector_series.iloc[0]))
                if sector_norm: sector_index.setdefault(sector_norm, []).append(ticker)
    return sector_index

AMBIGUOUS_ALIASES = {"on", "at", "be", "am", "or", "is", "it", "to", "us", "go", "all", "can", "in", "me", "up", "so", "do", "no", "by"}

def detect_tickers(text: str, data_frames: Dict[str, pd.DataFrame], max_candidates: int = 12) -> List[str]:
    if not text: return []
    found_tickers = set(extract_tickers_from_text(text))
    alias_index = build_alias_index(data_frames)
    text_norm = _norm(text)

    for alias, ticker in alias_index.items():
        if alias in AMBIGUOUS_ALIASES: continue
        if alias in text_norm and re.search(r"\b" + re.escape(alias) + r"\b", text_norm):
            found_tickers.add(ticker)

    return sorted([t for t in found_tickers if t in data_frames])[:max_candidates]

def detect_stock_group(q_text: str, sector_index: Dict[str, List[str]], all_data_frames: Dict[str, pd.DataFrame]) -> Tuple[List[str], str]:
    q_norm = _norm(q_text)
    for group_name_norm, tickers in STATIC_GROUP_MAP.items():
        if group_name_norm in q_norm:
            return [t for t in tickers if t in all_data_frames], group_name_norm
    for sector_name_norm, tickers in sector_index.items():
        if sector_name_norm in q_norm or f"{sector_name_norm} stocks" in q_norm:
            return tickers, sector_name_norm
    return [], None

def is_broad_ranking_question(text: str, mentioned_entities: List[str]) -> bool:
    keywords = ['best', 'worst', 'top', 'bottom', 'rank', 'strongest', 'weakest', 'outperform', 'underperform']
    if mentioned_entities: return False
    return any(k in (text or "").lower() for k in keywords)

MONTHS = {m.lower(): i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], start=1)}

def parse_date_range(text: str, default_end: datetime) -> Tuple[datetime, datetime, str]:
    s = (text or "").lower()
    m = re.search(r"\bq([1-4])\s*(20\d{2})\b", s)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        start_month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        start = datetime(year, start_month, 1)
        end = (datetime(year, start_month + 2, 1) + relativedelta(months=1)) - timedelta(days=1)
        return start, end, f"Q{q}_{year}"
    m = re.search(r"\b([a-z]+)\s+(20\d{2})\b", s)
    if m and m.group(1) in MONTHS:
        year, mon = int(m.group(2)), MONTHS[m.group(1)]
        start = datetime(year, mon, 1)
        end = (start + relativedelta(months=1)) - timedelta(days=1)
        return start, end, f"{year}-{mon:02d}"
    m = re.search(r"\b(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)\b", s)
    if m:
        n, unit = int(m.group(2)), m.group(3)
        if unit.startswith("day"): start = default_end - timedelta(days=n)
        elif unit.startswith("week"): start = default_end - timedelta(weeks=n)
        else: start = default_end - relativedelta(months=n)
        return start, default_end, f"last_{n}_{unit}"
    m = re.search(r"\b(20\d{2})\b", s)
    if m:
        year = int(m.group(1))
        start, end = datetime(year, 1, 1), datetime(year, 12, 31)
        if end > default_end: end = default_end
        return start, end, f"Year_{year}"
    return default_end - timedelta(days=DEFAULT_LOOKBACK_DAYS), default_end, f"default_{DEFAULT_LOOKBACK_DAYS}d"

def _pct(a: float, b: float) -> float:
    try: return (b - a) / a * 100.0
    except Exception: return float("nan")

def build_contexts_for_ticker(ticker: str, df: pd.DataFrame, start: datetime, end: datetime) -> Tuple[str, str]:
    p_start = pd.to_datetime(start).tz_localize(None) if pd.to_datetime(start).tz is not None else pd.to_datetime(start)
    p_end = pd.to_datetime(end).tz_localize(None) if pd.to_datetime(end).tz is not None else pd.to_datetime(end)

    win = df[(df['Date'] >= p_start) & (df['Date'] <= p_end)].copy()
    if win.empty:
        no_data_msg = f"[{ticker}] No data available for {start.date()} → {end.date()}.\n"
        return no_data_msg, no_data_msg

    win = win.sort_values('Date')
    first_close = win['Adj. Close'].dropna().iloc[0] if 'Adj. Close' in win.columns and not win['Adj. Close'].dropna().empty else np.nan
    last_close = win['Adj. Close'].dropna().iloc[-1] if 'Adj. Close' in win.columns and not win['Adj. Close'].dropna().empty else np.nan
    pct_change = _pct(first_close, last_close)

    trading_lines = [
        f"[{ticker}] Trading/Price Data {start.date()} → {end.date()}",
        f"- AdjClose: first={first_close:.4f}, last={last_close:.4f}, pct_change={pct_change:.2f}%",
    ]

    vvals = {}
    for col in [c for c in TRADING_COLS if c != 'Adj. Close']:
        if col in win.columns:
            series = win[col].dropna()
            vvals[col] = float(series.iloc[-1]) if not series.empty else np.nan

    if vvals:
        trading_lines.append("- Indicators: " + ", ".join([f"{k}={v:.4f}" if pd.notna(v) else f"{k}=NA" for k, v in vvals.items()]))
    else:
        trading_lines.append("- No additional trading indicators found.")

    fvals = {}
    for col in FUNDAMENTALS_COLS:
        if col in win.columns:
            series = win[col].dropna()
            fvals[col] = float(series.median()) if not series.empty else np.nan

    fun_lines = [f"[{ticker}] Fundamentals (median in window):"]
    if fvals:
        has_data = False
        for k, v in fvals.items():
            if pd.notna(v):
                fun_lines.append(f"  * {k}: {v:.4f}")
                has_data = True
            else:
                fun_lines.append(f"  * {k}: NA")
        if not has_data: fun_lines.append("  * No fundamental data found.")
    else:
        fun_lines.append("  * No fundamental data found in this window.")
    
    return "\n".join(trading_lines) + "\n", "\n".join(fun_lines) + "\n"

# =========================
# Output Helpers
# =========================
def ensure_header(output_file: Path, header: List[str], lock: threading.Lock):
    with lock:
        if not output_file.exists() or output_file.stat().st_size == 0:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=header).writeheader()

def append_rows_to_csv(output_file: Path, rows: List[Dict[str, Any]], header: List[str], lock: threading.Lock):
    if not rows: return
    with lock:
        try:
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                for r in rows: writer.writerow({k: r.get(k, "") for k in header})
                f.flush()
        except PermissionError:
            print(f"Error: Could not write to {output_file}. Is it open in Excel?")
        except Exception as e:
            print(f"Error writing to CSV: {e}")

# =========================
# Task Runner
# =========================
def run_generation_task(model: str, prompt: str, project_id: str) -> Tuple[str, float]:
    start_time = time.monotonic()
    response = get_model_response_universal(model, prompt, project_id, temperature=0.0)
    duration = time.monotonic() - start_time
    return (response or "").strip(), duration

def run_audit_task(model: str, trading_context: str, fundamental_context: str, response: str, project_id: str) -> Tuple[int, str]:
    audit_prompt = AUDITOR_PROMPT_TEMPLATE.format(trading_context=trading_context, fundamental_context=fundamental_context, response=response)
    audit_report = get_model_response_universal(model, audit_prompt, project_id, temperature=0.0)
    
    if "Summary: All claims are SUPPORTED." in audit_report: is_accurate = 1
    elif "Summary: Contains CONTRADICTED or NOT_FOUND claims." in audit_report: is_accurate = 0
    else: is_accurate = -1
    return is_accurate, audit_report.strip()

def run_self_selection_task(model: str, question: str, candidates: List[Dict[str, Any]], project_id: str) -> str:
    candidate_answers_str = "\n\n".join([f"--- Candidate: {c['prompt_id']} ---\n{c['response']}" for c in candidates])
    selection_prompt = SELF_SELECTION_PROMPT_TEMPLATE.format(question=question, candidate_answers_str=candidate_answers_str)
    choice_text = get_model_response_universal(model, selection_prompt, project_id, temperature=0.1)
    prompt_ids = [c['prompt_id'] for c in candidates]
    
    clean_choice = choice_text.replace("*", "").replace("'", "").replace('"', "").strip()
    if clean_choice in prompt_ids: return clean_choice
    
    found_ids = [pid for pid in prompt_ids if pid.lower() in clean_choice.lower()]
    if len(found_ids) == 1: return found_ids[0]
    elif len(found_ids) > 1:
        for pid in found_ids:
            if clean_choice.lower().startswith(pid.lower()): return pid
        best_match, last_index = None, -1
        for pid in found_ids:
            idx = clean_choice.lower().rfind(pid.lower())
            if idx > last_index: last_index, best_match = idx, pid
        return best_match
    
    return prompt_ids[0]

def process_question_model_pair(model: str, qobj: Dict[str, Any], data_frames: Dict[str, pd.DataFrame], sector_index: Dict[str, List[str]], today: datetime, pbar_inner, project_id: str) -> List[Dict[str, Any]]:
    q_text, q_id, q_type = qobj["question"], qobj["id"], qobj.get("type", "")
    golden_indicators = qobj["golden_indicators"]
    base_info = {"question_id": q_id, "question": q_text, "golden_indicators": "|".join(golden_indicators), "model": model}

    mentioned_tickers = detect_tickers(q_text, data_frames)
    group_tickers, group_name = detect_stock_group(q_text, sector_index, data_frames)
    start, end, date_label = parse_date_range(q_text, default_end=today)

    if group_tickers: mentioned_tickers = []
    trading_context_parts, fun_context_parts, tickers_for_context = [], [], []
    target_data_for_ranking, ranking_scope_name = data_frames, "all stocks"

    if mentioned_tickers:
        target_data_for_ranking = {t: data_frames[t] for t in mentioned_tickers if t in data_frames}
        ranking_scope_name = f"mentioned tickers ({len(target_data_for_ranking)})"
    elif group_tickers:
        target_data_for_ranking = {t: data_frames[t] for t in group_tickers if t in data_frames}
        ranking_scope_name = f"'{group_name}' group ({len(target_data_for_ranking)})"

    is_broad_ranking = is_broad_ranking_question(q_text, mentioned_tickers)
    if is_broad_ranking:
        metric_to_use, sort_ascending = get_ranking_parameters_from_question(q_text)
        all_ranked_stocks = rank_stocks_by_metric(target_data_for_ranking, start, end, metric=metric_to_use, sort_ascending=sort_ascending)
        top_stocks, bottom_stocks = all_ranked_stocks[:10], all_ranked_stocks[-10:]
        bottom_stocks.reverse()
        sort_desc = "Ascending" if sort_ascending else "Descending"
        
        trading_context_parts = [
            f"NOTE: Broad ranking requested. Ranking {ranking_scope_name} by '{metric_to_use}' ({sort_desc}) for period {start.date()} to {end.date()}.",
            f"Top {len(top_stocks)} (List Start): " + ", ".join([f"{t} ({v:.4f})" for t, v in top_stocks]),
            f"Bottom {len(bottom_stocks)} (List End): " + ", ".join([f"{t} ({v:.4f})" for t, v in bottom_stocks]),
            "\n--- Data for Ranked Tickers ---"
        ]
        fun_context_parts = ["\n--- Fundamental Data for Ranked Tickers ---"]
        tickers_for_context = sorted(list(set([t for t, _ in top_stocks] + [t for t, _ in bottom_stocks])))
    else:
        if mentioned_tickers: tickers_for_context = mentioned_tickers
        elif group_tickers: tickers_for_context = group_tickers
        else:
            available_tickers = list(data_frames.keys())
            if len(available_tickers) <= 300: tickers_for_context = available_tickers
            else:
                temp_ranked = rank_stocks_by_metric(data_frames, start, end, metric='Momentum_20D', sort_ascending=False)
                tickers_for_context = [t for t, _ in temp_ranked[:300]]

    for t in tickers_for_context:
        df = data_frames.get(t)
        if df is not None:
            vol_ctx, fun_ctx = build_contexts_for_ticker(t, df, start, end)
            trading_context_parts.append(vol_ctx)
            fun_context_parts.append(fun_ctx)
        else:
            trading_context_parts.append(f"[{t}] No data available.\n")
            fun_context_parts.append(f"[{t}] No data available.\n")

    trading_context_str, fundamental_context_str = "\n".join(trading_context_parts).strip(), "\n".join(fun_context_parts).strip()
    rag_base_info = {
        **base_info, "tickers_used": ",".join(tickers_for_context),
        "date_start": str(start.date()), "date_end": str(end.date()), "date_label": date_label,
        "trading_context": trading_context_str, "fundamental_context": fundamental_context_str
    }

    candidate_results = []
    context_str = f"Trading Signals Context:\n{trading_context_str}\n\nFundamental Context:\n{fundamental_context_str}"

    pbar_inner.set_description("Generating candidates")
    for prompt_id, template in PROMPTS_TO_EVALUATE.items():
        rag_prompt = template.format(query_str=q_text, context_str=context_str, trading_context_str=trading_context_str, fundamental_context_str=fundamental_context_str)
        response, duration = run_generation_task(model, rag_prompt, project_id)
        candidate_results.append({"prompt_id": prompt_id, "response": response, "response_time_sec": f"{duration:.4f}", "type": "With RAG"})

    pbar_inner.set_description("Generating No-RAG")
    response, duration = run_generation_task(model, q_text, project_id)
    candidate_results.append({"prompt_id": "No RAG", "response": response, "response_time_sec": f"{duration:.4f}", "type": "No RAG"})

    pbar_inner.set_description("Self-selecting")
    selected_prompt_id = run_self_selection_task(model, q_text, candidate_results, project_id)

    final_rows = []
    pbar_inner.set_description("Auditing candidates")
    for candidate in candidate_results:
        is_accurate, audit_report = (0, "Audit Failed or Error")
        if candidate.get("response"):
            is_accurate, audit_report = run_audit_task(model, trading_context_str, fundamental_context_str, candidate["response"], project_id)
        final_rows.append({
            **rag_base_info, **candidate,
            "is_self_selected": (candidate["prompt_id"] == selected_prompt_id),
            "self_selection_choice": selected_prompt_id,
            "is_numerically_accurate": is_accurate, "numerical_audit_report": audit_report,
        })

    pbar_inner.update(1)
    return final_rows

def run_evaluation(args):
    print("Initializing Vertex AI Context...")
    init_vertex_ai(args.project_id, args.location)

    print("Loading combined data...")
    data_frames_combined = load_data_frames(args.combined_dir, '-daily_with_fundamentals.csv')
    sector_index = build_sector_index(data_frames_combined)
    questions = load_questions_auto(args.questions_file)

    if args.filter_ids:
        ids_to_run = [i.strip() for i in args.filter_ids.split(',')]
        questions = [q for q in questions if q['id'] in ids_to_run]

    output_path = Path(args.output_file) if args.output_file else Path(DEFAULT_OUTPUT_FILE)
    completed_ids = set()
    
    if output_path.exists():
        try:
            print(f"Checking existing progress in: {output_path}")
            existing_df = pd.read_csv(output_path, on_bad_lines='skip')
            if 'response' in existing_df.columns and 'question_id' in existing_df.columns:
                is_error = existing_df['response'].astype(str).str.contains("Daily limit|Daily API limit|Resource Exhausted", case=False, na=False)
                is_generic_error = existing_df['response'].astype(str).str.startswith("Error:")
                valid_df = existing_df[~(is_error | is_generic_error)]
                completed_ids = set(valid_df['question_id'].unique())
                print(f"   - Valid Success rows: {len(completed_ids)}")
        except Exception as e:
            print(f"⚠️ Could not read existing file correctly: {e}")

    questions_to_process = [q for q in questions if str(q['id']) not in completed_ids]
    if not questions_to_process:
        print("🎉 All questions have been VALIDLY processed! Nothing to do.")
        return

    print(f"Remaining to process: {len(questions_to_process)}")
    write_lock = threading.Lock()
    ensure_header(output_path, OUTPUT_HEADER, write_lock)

    global RATE_CONTROLLER
    RATE_CONTROLLER.stop_event.clear()
    pbar_outer = tqdm(total=len(questions_to_process), desc=f"Processing {args.model}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {}
        for qobj in questions_to_process:
            if RATE_CONTROLLER.stop_event.is_set(): break
            future = ex.submit(process_question_model_pair, args.model, qobj, data_frames_combined, sector_index, DATASET_END_DATE, pbar_outer, args.project_id)
            futures[future] = qobj['id']

        for future in as_completed(futures):
            qid = futures[future]
            try:
                result_rows = future.result()
                if result_rows:
                    if any("Daily limit" in r.get('response', '') for r in result_rows):
                        print(f"\n[SKIP] Not writing error result for {qid} to CSV.")
                        RATE_CONTROLLER.stop_event.set()
                    else:
                        append_rows_to_csv(output_path, result_rows, OUTPUT_HEADER, write_lock)
            except RuntimeError as re:
                if "Daily limit" in str(re): RATE_CONTROLLER.stop_event.set()
            except Exception as e:
                if not RATE_CONTROLLER.stop_event.is_set(): pbar_outer.write(f"Task failed for {qid}: {e}")

    pbar_outer.close()
    if RATE_CONTROLLER.stop_event.is_set(): print(f"\n⏸️ Script paused. Progress saved to {output_path}.")
    else: print(f"\n✅ Complete. Results written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark responses using TELeR prompts (Universal Model Pipeline).")
    parser.add_argument("--project-id", type=str, required=True, help="Your Google Cloud Project ID.")
    parser.add_argument("--location", type=str, default="global", help="Vertex AI location.")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Target model name (e.g., gemini-3.1-pro-preview, qwen/qwen3-235b-a22b-instruct-2507-maas).")
    parser.add_argument("--combined-dir", type=str, default="./data/combined", help="Path to combined daily data CSVs.")
    parser.add_argument("--questions-file", type=str, default="./data/questions.csv", help="Input CSV containing benchmark questions.")
    parser.add_argument("--output-file", type=str, default="", help="Specific output CSV to resume from (optional).")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of concurrent threads.")
    parser.add_argument("--filter-ids", type=str, default="", help="Comma-separated list of Question IDs to force run.")

    args = parser.parse_args()
    print("--- Starting RAG Pipeline (Multi-Model, Resumable) ---")
    run_evaluation(args)