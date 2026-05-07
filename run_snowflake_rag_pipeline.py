#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinTradeBench: Snowflake Cortex RAG Pipeline

This script implements the RAG retrieval and generation pipeline using
Snowflake Cortex LLMs. It parses unstructured SEC filings and structured
trading signals, chunks them, and performs dense vector retrieval + BM25
before routing the context to the LLM.

Usage (terminal):
    export SNOWFLAKE_PASSWORD="your_password"
    python run_snowflake_rag_pipeline.py \
        --account YOUR_ACCOUNT \
        --user YOUR_USER \
        --base-dir ./data
"""

import os
import re
import json
import time
import hashlib
import warnings
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict

import snowflake.connector
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML / Vector Search
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# HTML Parsing
from bs4 import BeautifulSoup, Tag

warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION DEFAULTS
# =========================
SNOWFLAKE_MODELS = [
    "gemini-2.5-flash-lite",
    "llama3.3-70b",
    "openai-gpt-5-mini"
]

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# CHUNKING & QUOTAS
MAX_PARENT_TOKENS = 2000
CHILD_TOKEN_SIZE = 300
CHILD_TOKEN_OVERLAP = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PROMPT TEMPLATES (TELeR Strategy)
# =========================

PROMPTS_TO_EVALUATE = {
    "TELER_L1_Baseline": """You are a financial analyst. Provide a detailed response to Question: {query_str}. Please do not provide any investment advice.
Answer:""",

    "TELER_L2_Strict_Focus": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 2)
You are a senior financial analyst. Break down the question and answer the question methodically. First, explain what the question is asking and the main goal of the question clearly. 
Then, identify the embedded sub-questions in the given question. Answer each part. Consider both trading signals and company fundamentals. Finally, combine sub-answers into a single, comprehensive final answer and answer in a professional style.
The question is given as {query_str}. 
Your analysis should clearly distinguish between the reasoning process for your answers for each embedded sub question and the conclusion in the final answer.
""",

    "L3_Step_By_Step": """You are a senior financial analyst. Break down and answer the question methodically:
1. State what the question is asking.
2. Identify embedded sub-questions.
3. Answer each part. Consider both trading signals and company fundamentals.
4. Synthesize into a final answer.

Question: {query_str}
Answer:""",

    "TELER_L4_Auditor_Evidence": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 4)
You are a senior financial analyst. Break down and answer the question methodically:
1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part. Consider both trading signals and company fundamentals.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

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
3. **Answer Sub-Questions:** Answer each part using the given context.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.
5. **Support:** Cite exact supporting text or evidence retrieved from the given context.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

Context:
{context_str}

Question:
{query_str}

Supporting Evidence:
- "[Quote from Context]"

Explanation & Justification:
[Detailed rationale referencing both trading signals and company data.]

Final Answer:
[Provide final answer]
""",

    "TELER_L6_Maximalist": """(TELeR: Single-Turn, Instruction-Style, Role-Specified, Level 6)
You are a senior financial analyst. Break down and answer the question methodically:

1. **Clear goals:** State and explain what the question is asking.
2. **Deconstruct:** Identify embedded sub-questions.
3. **Answer Sub-Questions:** Answer each part using the given context.
4. **Synthesize:** Combine sub-answers into a single, comprehensive final answer.
5. **Support:** Cite exact supporting text or evidence retrieved from the given context.
6. **Justify:** Justify why information from the context is included or excluded and explain your reasoning.

Do not provide direct investment advice (e.g., 'buy', 'sell', 'hold').
Evaluation Method: A good response will focus on four key metrics: answer correctness, context accuracy, answer completeness and answer coherence.

Context:
{context_str}

Question:
{query_str}

Supporting Evidence:
- "[Quote from Context]"

Explanation & Justification:
[Detailed rationale referencing both trading signals and company data.]

Final Answer:
[Comprehensive, well-justified answer]
"""
}

SELF_SELECTION_PROMPT = """You are a Lead Financial Editor. You have been given a Question and several Candidate Answers generated by an AI analyst.
Your task is to select the single *best* answer that is the most accurate, complete, and relevant.

Question:
{question}

--- CANDIDATE ANSWERS ---
{candidate_answers_str}
-------------------------

INSTRUCTIONS:
Review all candidate answers. Respond with *only* the Prompt ID of the answer you choose (e.g., "L3_Step_By_Step").
Do not add any explanation.

Your Choice (Prompt ID only):"""


# =========================
# 1. HELPERS & CHUNKING
# =========================

def estimate_tokens(text: str) -> int:
    if not text: return 0
    return len(text) // 4


def split_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    if estimate_tokens(text) <= max_tokens: return [text]
    chunk_char_size = max_tokens * 4
    overlap_char = 200
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_char_size, len(text))
        if end < len(text):
            last_newline = text.rfind('\n', end - int(chunk_char_size * 0.1), end)
            if last_newline != -1: end = last_newline + 1
        chunks.append(text[start:end])
        start = end - overlap_char if end < len(text) else end
    return chunks


@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any]

    def __hash__(self): return hash(self.id)

    def __eq__(self, other): return self.id == other.id


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def extract_date_from_filename(filename: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else "Unknown Date"


# =========================
# 2. GROUP & SECTOR MAPS
# =========================

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[^a-z0-9\s\.-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


STATIC_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX", "adobe": "ADBE", "broadcom": "AVGO",
    "intel": "INTC", "advanced micro devices": "AMD", "amd": "AMD", "paypal": "PYPL", "pepsico": "PEP",
    "coca cola": "KO", "coca-cola": "KO"
    # ... (Truncated for brevity in this display, standard aliases remain) ...
}

STATIC_GROUP_MAP = {
    _norm("fang stocks"): ["META", "AMZN", "NFLX", "GOOGL"],
    _norm("faang stocks"): ["META", "AMZN", "AAPL", "NFLX", "GOOGL"],
    _norm("manga stocks"): ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"],
    _norm("magnificent seven"): ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    _norm("semiconductor stocks"): ["NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI",
                                    "NXPI", "MRVL"],
    _norm("chip stocks"): ["NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI"],
    _norm("cybersecurity stocks"): ["PANW", "CRWD", "FTNT", "ZS", "OKTA"],
    _norm("cloud stocks"): ["ADBE", "INTU", "WDAY", "ADSK", "TEAM", "DDOG", "SNPS", "CDNS"],
    _norm("software stocks"): ["MSFT", "ADBE", "INTU", "WDAY", "ADSK", "TEAM", "SNPS", "CDNS"],
    _norm("biotech stocks"): ["AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "AZN"],
    _norm("healthcare stocks"): ["AMGN", "GILD", "VRTX", "REGN", "ISRG", "DXCM", "IDXX"],
    _norm("ev stocks"): ["TSLA", "RIVN", "LCID"],
    _norm("consumer stocks"): ["AMZN", "COST", "PEP", "SBUX", "MDLZ", "LULU", "MAR", "BKNG", "ABNB"],
    _norm("travel stocks"): ["BKNG", "ABNB", "MAR", "EXPE"],
    _norm("social media stocks"): ["META", "PINS", "SNAP", "GOOGL", "MTCH"],
}

FALLBACK_LEADERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

# =========================
# 3. TICKER DETECTION & INDEXING
# =========================
AMBIGUOUS_ALIASES = {"on", "at", "be", "am", "or", "is", "it", "to", "us", "go", "all", "can", "in"}


def detect_tickers_and_groups(text: str, valid_tickers: Set[str]) -> List[str]:
    if not text: return []
    found = set()
    text_norm = _norm(text)

    for group_name, group_tickers in STATIC_GROUP_MAP.items():
        if group_name in text_norm:
            valid_group = [t for t in group_tickers if t in valid_tickers]
            found.update(valid_group)

    for alias, ticker in STATIC_ALIASES.items():
        if ticker in valid_tickers and alias not in AMBIGUOUS_ALIASES:
            if re.search(r"\b" + re.escape(alias) + r"\b", text_norm):
                found.add(ticker)

    regex_matches = re.findall(r"\b[A-Z]{1,5}\b", text)
    for m in regex_matches:
        if m in valid_tickers and m.lower() not in AMBIGUOUS_ALIASES:
            found.add(m)

    return sorted(list(found))


def html_table_to_markdown(table_tag: Tag, context_prefix: str) -> List[str]:
    rows = []
    headers = []
    header_row = table_tag.find('tr')
    if header_row: headers = [clean_text(th.get_text()) for th in header_row.find_all(['th', 'td'])]
    tr_tags = table_tag.find_all('tr')
    start_idx = 1 if header_row else 0
    for tr in tr_tags[start_idx:]:
        cells = [clean_text(td.get_text()) for td in tr.find_all(['td', 'th'])]
        if any(c for c in cells if len(c) > 1): rows.append(cells)
    if not rows: return []
    if not headers and rows: headers = rows[0]; rows = rows[1:]

    def make_md(h_row, b_rows):
        md = f"\n\n**Table: {context_prefix}**\n| " + " | ".join(h_row) + " |\n| " + " | ".join(
            ["---"] * len(h_row)) + " |\n"
        for row in b_rows: md += "| " + " | ".join(row + [""] * (len(h_row) - len(row))) + " |\n"
        return md

    chunks = []
    for i in range(0, len(rows), 15): chunks.append(make_md(headers, rows[i:i + 15]))
    return chunks


def parse_sec_html(html_content: str, ticker: str, filename: str) -> List[Dict]:
    soup = BeautifulSoup(html_content, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]): tag.decompose()
    raw_sections = [];
    current_title = "General";
    current_buffer = [];
    filing_date = extract_date_from_filename(filename)

    for el in soup.find_all(['p', 'div', 'table', 'h1', 'h2', 'h3', 'h4']):
        text = clean_text(el.get_text())
        is_header = False
        if el.name in ['h1', 'h2', 'h3'] or (el.name in ['p', 'div'] and len(text) < 150):
            if re.match(r"^(?:item|part)\s+\d+[a-z]?", text, re.IGNORECASE): is_header = True

        if is_header:
            if current_buffer: raw_sections.append({"title": current_title, "content": "\n".join(current_buffer)})
            current_title = text;
            current_buffer = []
            continue
        if el.name == 'table':
            ctx = f"{current_title}"
            current_buffer.extend(html_table_to_markdown(el, ctx))
            continue
        if len(text) > 20: current_buffer.append(text)
    if current_buffer: raw_sections.append({"title": current_title, "content": "\n".join(current_buffer)})

    final = []
    for sec in raw_sections:
        for sub in split_text_by_tokens(sec['content'], MAX_PARENT_TOKENS):
            final.append({"title": sec['title'], "content": sub, "date": filing_date})
    return final


def get_file_hash(filepath: Path) -> str:
    with open(filepath, 'rb') as f: return hashlib.md5(f.read()).hexdigest()


def generate_snowflake(cursor, model_name, prompt):
    safe_prompt = prompt.replace("'", "''")
    query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_name}', '{safe_prompt}')"
    cursor.execute(query)
    result = cursor.fetchone()[0]
    return result


class TickerIndex:
    def __init__(self, ticker: str, embedder, rag_store_path: Path):
        self.ticker = ticker
        self.embedder = embedder
        self.index_dir = rag_store_path / ticker
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        self.parent_map = {}
        self.manifest_path = self.index_dir / "manifest.json"
        self.manifest = self._load_manifest()
        self.faiss_index = None
        self.bm25 = None

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f: return json.load(f)
        return {"files": {}}

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f: json.dump(self.manifest, f, indent=2)

    def is_file_processed(self, filename: str, filepath: Path) -> bool:
        return self.manifest["files"].get(filename) == get_file_hash(filepath)

    def add_sec_file(self, filepath: Path):
        filename = filepath.name
        if self.is_file_processed(filename, filepath): return
        print(f"[{self.ticker}] Processing SEC: {filename}")
        parents = parse_sec_html(filepath.read_text(encoding='utf-8', errors='ignore'), self.ticker, filename)
        new_chunks = []
        for p in parents:
            pid = hashlib.md5((filename + p["title"] + p["content"][:20]).encode()).hexdigest()
            self.parent_map[pid] = f"# {p['title']} ({p['date']})\n\n{p['content']}"
            start = 0
            full = self.parent_map[pid]
            while start < len(full):
                end = min(start + CHILD_TOKEN_SIZE * 4, len(full))
                if end - start > 100:
                    new_chunks.append(Chunk(id=f"{pid}_{start}", parent_id=pid,
                                            text=f"Ticker: {self.ticker} | Date: {p['date']} | Section: {p['title']}\n{full[start:end]}",
                                            metadata={"source": "SEC", "date": p['date'], "filename": filename,
                                                      "type": "sec"}))
                start += CHILD_TOKEN_SIZE * 4 - CHILD_TOKEN_OVERLAP * 4
        self.chunks.extend(new_chunks)
        self.manifest["files"][filename] = get_file_hash(filepath)

    def add_price_file(self, filepath: Path):
        filename = filepath.name
        if self.is_file_processed(filename, filepath): return
        print(f"[{self.ticker}] Processing Trading Signals/Prices: {filename}")
        df = pd.read_excel(filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns: return
        df['date'] = pd.to_datetime(df['date'])
        new_chunks = []
        for period, group in df.groupby(df['date'].dt.to_period("M")):
            period_str = str(period)
            lines = [f"## Trading Data: {self.ticker} - {period_str}", "| Date | Close | Vol |", "|---|---|---|"]
            for _, row in group.iterrows(): lines.append(
                f"| {row['date'].strftime('%Y-%m-%d')} | {row.get('close', '-')} | {row.get('volume', '-')} |")
            full = "\n".join(lines)
            pid = hashlib.md5((filename + period_str).encode()).hexdigest()
            self.parent_map[pid] = full
            new_chunks.append(Chunk(id=f"{pid}_0", parent_id=pid,
                                    text=f"Ticker: {self.ticker} | Type: Trading Signals Data | Period: {period_str}\n{full}",
                                    metadata={"source": "PRICE", "date": period_str, "filename": filename,
                                              "type": "price"}))
        self.chunks.extend(new_chunks)
        self.manifest["files"][filename] = get_file_hash(filepath)

    def build_and_save(self):
        if not self.chunks: return
        print(f"[{self.ticker}] Building Index ({len(self.chunks)} chunks)...")
        texts = [c.text for c in self.chunks]
        emb = self.embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        self.faiss_index = faiss.IndexFlatIP(emb.shape[1])
        faiss.normalize_L2(emb)
        self.faiss_index.add(emb)
        faiss.write_index(self.faiss_index, str(self.index_dir / "vector.index"))
        with open(self.index_dir / "chunks.json", "w") as f: json.dump([asdict(c) for c in self.chunks], f)
        with open(self.index_dir / "parents.json", "w") as f: json.dump(self.parent_map, f)
        self._save_manifest()

    def load(self) -> bool:
        if not (self.index_dir / "vector.index").exists(): return False
        self.faiss_index = faiss.read_index(str(self.index_dir / "vector.index"))
        with open(self.index_dir / "chunks.json", "r") as f: self.chunks = [Chunk(**d) for d in json.load(f)]
        with open(self.index_dir / "parents.json", "r") as f: self.parent_map = json.load(f)
        self.bm25 = BM25Okapi([c.text.lower().split() for c in self.chunks])
        return True


# =========================
# 4. DYNAMIC RETRIEVAL ENGINE
# =========================

class RAGEngine:
    def __init__(self, sec_root: Path, prices_root: Path, rag_store: Path):
        print(f"Loading Retrieval models on {DEVICE}...")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
        self.reranker = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
        self.indices = {}
        self.sec_root = sec_root
        self.prices_root = prices_root
        self.rag_store = rag_store

    def get_index(self, ticker):
        if ticker not in self.indices:
            idx = TickerIndex(ticker, self.embedder, self.rag_store)
            if idx.load():
                self.indices[ticker] = idx
            else:
                return None
        return self.indices.get(ticker)

    def build_all(self):
        print("Building/Verifying Indices...")
        tickers = [d.name for d in self.sec_root.iterdir() if d.is_dir()]

        for t in tqdm(tickers, desc="Indexing Tickers"):
            idx = TickerIndex(t, self.embedder, self.rag_store)

            for f in (self.sec_root / t).glob("*.htm"):
                idx.add_sec_file(f)

            p_file = self.prices_root / f"{t}-history.xlsx"
            if p_file.exists():
                idx.add_price_file(p_file)

            idx.build_and_save()

    def extract_query_year(self, query: str):
        match = re.search(r"\b(20[12]\d)\b", query)
        return match.group(1) if match else None

    def retrieve_multi(self, query: str, tickers: List[str]) -> List[Dict]:
        target_year = self.extract_query_year(query)
        num_tickers = len(tickers)
        if num_tickers == 0: return []

        base_sec_quota = max(1, int(15 / num_tickers))
        base_price_quota = max(1, int(10 / num_tickers))
        MAX_TOTAL_CHUNKS = 25

        all_sec = []
        all_price = []

        for t in tickers:
            idx = self.get_index(t)
            if not idx: continue

            q_vec = self.embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q_vec)
            D, I = idx.faiss_index.search(q_vec, 50)

            p_query = f"Stock price trading market data history {t}" + (f" {target_year}" if target_year else "")
            p_vec = self.embedder.encode([p_query], convert_to_numpy=True)
            faiss.normalize_L2(p_vec)
            Dp, Ip = idx.faiss_index.search(p_vec, 15)

            indices = set(I[0]).union(set(Ip[0]))
            t_sec = [];
            t_price = []

            for i in indices:
                if i < 0 or i >= len(idx.chunks): continue
                c = idx.chunks[i]
                is_match = target_year in str(c.metadata.get('date', '')) if target_year else True

                if c.metadata['type'] == 'price':
                    if is_match: t_price.append(c)
                else:
                    t_sec.append(c)

            if t_sec:
                scores = self.reranker.predict([[query, c.text] for c in t_sec])
                ranked = sorted(zip(t_sec, scores), key=lambda x: x[1], reverse=True)
                seen = set()
                count = 0
                for c, s in ranked:
                    if c.parent_id in seen: continue
                    full = idx.parent_map.get(c.parent_id, c.text)

                    meta_with_ticker = c.metadata.copy()
                    meta_with_ticker['ticker'] = t

                    all_sec.append({
                        "id": c.id,
                        "parent_id": c.parent_id,
                        "text": full,
                        "meta": meta_with_ticker,
                        "score": s
                    })
                    seen.add(c.parent_id)
                    count += 1
                    if count >= base_sec_quota: break

            if t_price:
                t_price.sort(key=lambda x: x.metadata.get('date', ''), reverse=True)
                seen = set()
                count = 0
                for c in t_price:
                    if c.parent_id in seen: continue
                    full = idx.parent_map.get(c.parent_id, c.text)

                    meta_with_ticker = c.metadata.copy()
                    meta_with_ticker['ticker'] = t

                    all_price.append({
                        "id": c.id,
                        "parent_id": c.parent_id,
                        "text": full,
                        "meta": meta_with_ticker,
                        "score": 1.0
                    })
                    seen.add(c.parent_id)
                    count += 1
                    if count >= base_price_quota: break

        if len(all_sec) + len(all_price) > MAX_TOTAL_CHUNKS:
            remaining_slots = MAX_TOTAL_CHUNKS - len(all_price)
            if remaining_slots > 0:
                all_sec.sort(key=lambda x: x['score'], reverse=True)
                all_sec = all_sec[:remaining_slots]
            else:
                all_price = all_price[:MAX_TOTAL_CHUNKS]
                all_sec = []

        return all_price + all_sec


# =========================
# PROCESSING LOOP
# =========================
def process_single_question_snowflake(cursor, model_name, rag, row, valid_tickers) -> Dict:
    start_total = time.time()
    query = str(row.get('Query', row.get('question', '')))

    target_tickers = detect_tickers_and_groups(query, valid_tickers)

    fallback_used = False
    if not target_tickers:
        target_tickers = [t for t in FALLBACK_LEADERS if t in valid_tickers]
        fallback_used = True

    if not target_tickers:
        return {**row.to_dict(), "Error": "No tickers found"}

    start_ret = time.time()
    context_list = rag.retrieve_multi(query, target_tickers) if target_tickers else []
    retrieval_time = time.time() - start_ret

    combined_text = "\n".join([c['text'] for c in context_list])

    if "deepseek" in model_name.lower():
        MAX_DEEPSEEK_CHARS = 65000
        if len(combined_text) > MAX_DEEPSEEK_CHARS:
            combined_text = combined_text[:MAX_DEEPSEEK_CHARS] + "\n...(truncated for DeepSeek safety)..."

    full_context_str = "MARKET DATA:\n" + combined_text

    candidates = []
    for pid, template in PROMPTS_TO_EVALUATE.items():
        user_prompt = template.format(context_str=full_context_str, query_str=query)

        s_gen = time.time()
        resp = generate_snowflake(cursor, model_name, user_prompt)
        dur = time.time() - s_gen

        candidates.append({"prompt_id": pid, "response": resp, "duration": dur})

    s_sel = time.time()
    cand_str = "\n".join([f"--- {c['prompt_id']} ---\n{c['response']}" for c in candidates])
    select_prompt = SELF_SELECTION_PROMPT.format(question=query, candidate_answers_str=cand_str)

    best_id_raw = generate_snowflake(cursor, model_name, select_prompt)
    select_time = time.time() - s_sel

    clean_id = best_id_raw.split('\n')[0].strip().replace("'", "").replace('"', "")
    winner = next((c for c in candidates if c['prompt_id'] in clean_id), candidates[0])

    total_time = time.time() - start_total

    res = row.to_dict()
    res['Detected_Tickers'] = ", ".join(target_tickers)
    res['Fallback_Used'] = fallback_used
    res['Selected_Strategy'] = clean_id
    res['Final_Answer'] = winner['response']

    res['Time_Total_Sec'] = round(total_time, 2)
    res['Time_Retrieval_Sec'] = round(retrieval_time, 2)
    res['Time_Generation_Sec'] = round(winner['duration'], 2)
    res['Time_Selection_Sec'] = round(select_time, 2)

    readable_sources = []
    for item in context_list:
        m = item.get('meta', {})
        source_str = f"{m.get('ticker', 'UNK')}:{m.get('type', 'UNK').upper()}:{m.get('date', 'UNK')}"
        readable_sources.append(source_str)

    res['Retrieved_Sources'] = " | ".join(readable_sources)
    res['Context_IDs'] = " | ".join([str(item.get('parent_id', 'None')) for item in context_list])

    for c in candidates:
        res[f"Cand_{c['prompt_id']}"] = c['response']

    return res


# =========================
# MAIN
# =========================
def main(args):
    print("--- STARTING SNOWFLAKE RAG FOR FINTRADEBENCH ---")

    sf_account = os.getenv("SNOWFLAKE_ACCOUNT", args.account)
    sf_user = os.getenv("SNOWFLAKE_USER", args.user)
    sf_password = os.getenv("SNOWFLAKE_PASSWORD")
    sf_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", args.warehouse)
    sf_database = os.getenv("SNOWFLAKE_DATABASE", args.database)
    sf_schema = os.getenv("SNOWFLAKE_SCHEMA", args.schema)
    sf_role = os.getenv("SNOWFLAKE_ROLE", args.role)

    if not sf_password:
        print("[ERROR] SNOWFLAKE_PASSWORD environment variable is not set.")
        return

    try:
        conn = snowflake.connector.connect(
            user=sf_user,
            password=sf_password,
            account=sf_account,
            warehouse=sf_warehouse,
            database=sf_database,
            schema=sf_schema,
            role=sf_role
        )
        cursor = conn.cursor()
        print("[OK] Connected to Snowflake")

        print("[INFO] Enabling Cross-Region Inference...")
        try:
            cursor.execute("ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';")
            print("[OK] Cross-Region Inference Enabled!")
        except Exception as e:
            print(f"[WARN] Could not set Cross-Region. Manual setup may be required: {e}")

    except Exception as e:
        print(f"[ERROR] Snowflake Connection Failed: {e}")
        return

    base_dir = Path(args.base_dir)
    sec_root = base_dir / args.sec_dir
    prices_root = base_dir / args.prices_dir
    rag_store = base_dir / args.rag_store_dir
    questions_path = base_dir / args.questions_file

    rag = RAGEngine(sec_root=sec_root, prices_root=prices_root, rag_store=rag_store)
    rag.build_all()

    try:
        df = pd.read_csv(questions_path)
    except FileNotFoundError:
        print(f"[ERROR] Questions file not found at: {questions_path}")
        return

    valid_tickers = set([d.name for d in sec_root.iterdir() if d.is_dir()])

    for model in SNOWFLAKE_MODELS:
        print(f"\n🚀 Running Model: {model}")
        output_file = base_dir / args.output_dir / f"rag_results_snowflake_non_rag_response_{model}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        processed_ids = set()
        if os.path.exists(output_file):
            try:
                existing = pd.read_csv(output_file)
                if 'Question ID' in existing.columns:
                    processed_ids = set(existing['Question ID'].astype(str))
                elif 'question_id' in existing.columns:
                    processed_ids = set(existing['question_id'].astype(str))
                print(f"[INFO] Found {len(processed_ids)} processed questions for {model}. Skipping.")
            except Exception:
                pass

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Querying {model}"):
            q_id = str(row.get('Question ID', row.get('question_id', idx)))
            if q_id in processed_ids: continue

            try:
                res = process_single_question_snowflake(cursor, model, rag, row, valid_tickers)

                write_header = not os.path.exists(output_file)
                pd.DataFrame([res]).to_csv(output_file, mode='a', header=write_header, index=False)

            except Exception as e:
                print(f"[ERROR] Failed on {q_id}: {e}")

    conn.close()
    print("\n✅ All Snowflake Cortex jobs done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Snowflake Cortex RAG Pipeline for FinTradeBench")

    # Snowflake Credentials (can also be passed via ENV VARS)
    parser.add_argument("--account", type=str, help="Snowflake Account Identifier")
    parser.add_argument("--user", type=str, help="Snowflake User")
    parser.add_argument("--warehouse", type=str, default="COMPUTE_WH", help="Snowflake Warehouse")
    parser.add_argument("--database", type=str, default="SNOWFLAKE_LEARNING_DB", help="Snowflake Database")
    parser.add_argument("--schema", type=str, default="PUBLIC", help="Snowflake Schema")
    parser.add_argument("--role", type=str, default="ACCOUNTADMIN", help="Snowflake Role")

    # Directory Configuration
    parser.add_argument("--base-dir", type=str, default="./data", help="Base directory for all data")
    parser.add_argument("--sec-dir", type=str, default="rag/sec_filings", help="Relative path to SEC files")
    parser.add_argument("--prices-dir", type=str, default="history", help="Relative path to Price history files")
    parser.add_argument("--rag-store-dir", type=str, default="rag_store", help="Relative path to save vector indices")
    parser.add_argument("--questions-file", type=str, default="Questions/scaled_question_by_company.csv",
                        help="Relative path to input questions CSV")
    parser.add_argument("--output-dir", type=str, default="output", help="Relative path to save results")

    args = parser.parse_args()
    main(args)