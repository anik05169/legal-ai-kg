"""
evaluate_retrieval.py
=====================
Comprehensive Retrieval Evaluation Harness for the Legal AI RAG pipeline.

Measures:
  • Recall@K  (K = 1, 3, 5, 10)
  • Mean Reciprocal Rank (MRR)
  • Normalised Discounted Cumulative Gain (nDCG@10)
  • Index Coverage (does the ground-truth text exist anywhere in the DB?)
  • Per-clause-category breakdown  (exploits CUAD's clause-type metadata)
  • End-to-end latency  (embedding + vector search, per query)
  • Query reformulation A/B comparison  (raw CUAD prompts vs compressed)

Outputs:
  1. Pretty-printed terminal report
  2. Machine-readable JSON report  (saved to reports/retrieval_eval_<timestamp>.json)

Usage:
  python evaluate_retrieval.py                     # default: 100 samples, both modes
  python evaluate_retrieval.py --samples 50        # quick run on 50 QA pairs
  python evaluate_retrieval.py --mode raw           # only raw-query mode
  python evaluate_retrieval.py --top-k 20           # evaluate up to Recall@20
  python evaluate_retrieval.py --verbose             # show every query result
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import certifi
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from core.data_loader import load_cuad_dataset

# =====================================================================
# ⚙️  CONFIGURATION
# =====================================================================
_ENV_PATH = ".env"
if os.path.exists(_ENV_PATH):
    try:
        with open(_ENV_PATH, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))
    except Exception:
        pass

MONGO_URI      = os.getenv("MONGO_URI", "")
DB_NAME        = "legal_rag"
COLLECTION     = "chunks"
INDEX_NAME     = "vector_index"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"

# CUAD clause categories are encoded in the question ID as "<title>__<category>_<n>"
# Some IDs may use different separators; try multiple patterns.
_CATEGORY_RE = re.compile(r"__(.+?)(?:_\d+|$)")
_CATEGORY_FALLBACK_RE = re.compile(r"/(.+?)(?:_\d+)?$")


# =====================================================================
# 🛠️  HELPERS
# =====================================================================

def _load_env_uri() -> str:
    """Return a usable MONGO_URI or raise."""
    uri = MONGO_URI
    if not uri:
        print("❌  MONGO_URI is not set.  Export it or add it to .env")
        sys.exit(1)
    return uri


def normalize_ws(text: str) -> str:
    """Collapse all whitespace runs (spaces, tabs, newlines) into single spaces.

    CRITICAL: The chunking pipeline uses text.split() + " ".join() which
    collapses multi-space runs.  CUAD ground-truth answers preserve the
    original irregular whitespace from the PDF.  Without this normalization
    the substring match will *always* fail for answers with extra spaces."""
    return re.sub(r"\s+", " ", text).strip()


def compress_query(query: str) -> str:
    """Query reformulation — strip the bloated CUAD instruction down to its
    core legal concept so the embedding model gets a cleaner signal."""

    # 1. Try extracting the quoted clause name AND the description after "Details:"
    #    to give the embedding model both the label and the semantic definition.
    quoted = re.findall(r'"([^"]*)"', query)
    details = ""
    if "details:" in query.lower():
        details = query.split(":", 1)[-1].strip() if ":" in query.split("Details")[-1] else ""
        # Get everything after the last "Details:"
        parts = re.split(r"[Dd]etails:\s*", query)
        if len(parts) > 1:
            details = parts[-1].strip()

    if quoted and details:
        return f"{quoted[0]}: {details}"
    if quoted:
        return f"{quoted[0]} clause in a legal contract"
    if details:
        return details
    return query


def extract_clause_category(qa_id: str) -> str:
    """Derive the CUAD clause category from the question ID string.

    CUAD IDs look like:
      'CreditSuisseGoldmanNY__Parties_0'  or
      'some-title/Governing-Law_3'
    """
    m = _CATEGORY_RE.search(qa_id)
    if m:
        return m.group(1).replace("-", " ").replace("_", " ").strip().title()
    m = _CATEGORY_FALLBACK_RE.search(qa_id)
    if m:
        return m.group(1).replace("-", " ").replace("_", " ").strip().title()
    # Last resort: try to extract from the question text itself
    return "Unknown"


def _extract_category_from_question(question: str) -> str:
    """Fallback: extract clause category from the question text when ID parsing fails."""
    quoted = re.findall(r'"([^"]*)"', question)
    if quoted:
        return quoted[0].strip().title()
    return "Unknown"


# =====================================================================
# 📐  METRIC CALCULATORS
# =====================================================================

def recall_at_k(ranks: list[int], k: int) -> float:
    """Fraction of queries where the gold chunk appeared in the top-K."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 1 <= r <= k) / len(ranks) * 100


def mrr(ranks: list[int]) -> float:
    """Mean Reciprocal Rank — average of 1/rank for found queries, 0 otherwise."""
    if not ranks:
        return 0.0
    return sum(1.0 / r if r >= 1 else 0.0 for r in ranks) / len(ranks)


def ndcg_at_k(ranks: list[int], k: int) -> float:
    """
    nDCG@K with binary relevance (1 if gold chunk found, 0 otherwise).
    Ideal DCG = 1 / log2(2) = 1.0  for each query (single relevant doc).
    """
    if not ranks:
        return 0.0
    dcg_sum = 0.0
    for r in ranks:
        if 1 <= r <= k:
            dcg_sum += 1.0 / math.log2(r + 1)
    ideal_dcg = len(ranks) * 1.0  # best case: every query's doc at rank 1
    return dcg_sum / ideal_dcg if ideal_dcg else 0.0


# =====================================================================
# 🔍  CORE EVALUATION LOOP
# =====================================================================

def evaluate(
    test_suite: list[dict],
    collection,
    embedder,
    *,
    top_k: int = 10,
    use_compression: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run vector-search retrieval against every QA pair and return a metrics dict.

    Each element of ``test_suite`` must have keys:
        question, answer, context, id, category
    """
    ranks: list[int]          = []
    latencies: list[float]    = []
    failures: list[dict]      = []
    category_ranks: dict[str, list[int]] = defaultdict(list)
    index_hit_count           = 0

    num_candidates = max(top_k * 5, 50)  # broad ANN fan-out

    for idx, qa in enumerate(test_suite):
        raw_query       = qa["question"]
        # Normalize whitespace so "The  Distributor" matches "The Distributor" in chunks
        expected_answer = normalize_ws(qa["answer"]).lower()
        category        = qa.get("category", "Unknown")

        query_text = compress_query(raw_query) if use_compression else raw_query

        # — Timed retrieval —
        t0 = time.perf_counter()
        query_vec = embedder.encode([query_text]).tolist()[0]

        pipeline = [
            {
                "$vectorSearch": {
                    "index":         INDEX_NAME,
                    "path":          "embedding",
                    "queryVector":   query_vec,
                    "numCandidates": num_candidates,
                    "limit":         top_k,
                }
            },
            {
                "$project": {
                    "text":  1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(collection.aggregate(pipeline))
        except Exception as exc:
            if verbose:
                print(f"  ⚠️  Query {idx+1} failed: {exc}")
            ranks.append(-1)
            latencies.append(time.perf_counter() - t0)
            continue

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # — Find rank of gold answer —
        found_rank = -1
        retrieved_snippets: list[str] = []
        scores: list[float]           = []

        for rank_pos, doc in enumerate(results, 1):
            chunk_text = doc.get("text", "")
            retrieved_snippets.append(chunk_text)
            scores.append(doc.get("score", 0.0))
            # Normalize both sides so multi-space CUAD answers match single-space chunks
            if found_rank == -1 and expected_answer in normalize_ws(chunk_text).lower():
                found_rank = rank_pos

        ranks.append(found_rank)
        category_ranks[category].append(found_rank)

        # — Index-coverage check on misses —
        if found_rank == -1:
            # Use the whitespace-normalized answer for regex matching too.
            # Replace each space with \s+ to handle any whitespace variation.
            normalized_answer = normalize_ws(qa["answer"])
            ws_flex_pattern = re.sub(r"\s+", r"\\s+", re.escape(normalized_answer))
            exists_in_db = bool(
                collection.find_one({"text": {"$regex": ws_flex_pattern, "$options": "i"}})
            )
            if exists_in_db:
                index_hit_count += 1

            failures.append({
                "index":           idx,
                "raw_query":       raw_query[:120],
                "search_query":    query_text[:120],
                "expected_answer": qa["answer"][:120],
                "in_index":        exists_in_db,
                "top_score":       scores[0] if scores else 0.0,
                "top_chunk":       retrieved_snippets[0][:200] if retrieved_snippets else "",
                "category":        category,
            })
        else:
            index_hit_count += 1

        if verbose:
            status = f"✅ rank={found_rank}" if found_rank >= 1 else "❌ miss"
            print(f"  [{idx+1:>4}/{len(test_suite)}]  {status}  ({elapsed*1000:.0f} ms)  {category}")

    # — Aggregate metrics —
    n = len(ranks) or 1
    ks = [1, 3, 5, 10]
    # extend with top_k if it's non-standard
    if top_k not in ks and top_k > 10:
        ks.append(top_k)
    ks = sorted(set(k for k in ks if k <= top_k))

    recall = {f"recall@{k}": round(recall_at_k(ranks, k), 2) for k in ks}
    mrr_val = round(mrr(ranks), 4)
    ndcg_val = round(ndcg_at_k(ranks, top_k), 4)
    coverage = round(index_hit_count / n * 100, 2)

    lat_ms = [t * 1000 for t in latencies]
    lat_stats = {
        "mean_ms":   round(sum(lat_ms) / len(lat_ms), 1) if lat_ms else 0,
        "p50_ms":    round(sorted(lat_ms)[len(lat_ms) // 2], 1) if lat_ms else 0,
        "p95_ms":    round(sorted(lat_ms)[int(len(lat_ms) * 0.95)], 1) if lat_ms else 0,
        "max_ms":    round(max(lat_ms), 1) if lat_ms else 0,
    }

    # Per-category recall@1
    cat_breakdown = {}
    for cat, cat_ranks in sorted(category_ranks.items()):
        cat_n = len(cat_ranks)
        cat_breakdown[cat] = {
            "n":         cat_n,
            "recall@1":  round(recall_at_k(cat_ranks, 1), 2),
            "recall@5":  round(recall_at_k(cat_ranks, 5), 2),
            "recall@10": round(recall_at_k(cat_ranks, 10), 2),
            "mrr":       round(mrr(cat_ranks), 4),
        }

    return {
        "total_queries":     len(ranks),
        "top_k":             top_k,
        **recall,
        "mrr":               mrr_val,
        f"ndcg@{top_k}":     ndcg_val,
        "index_coverage_%":  coverage,
        "latency":           lat_stats,
        "category_breakdown": cat_breakdown,
        "failure_count":     len(failures),
        "sample_failures":   failures[:10],  # keep report compact
    }


# =====================================================================
# 📊  PRETTY PRINTING
# =====================================================================

_BOLD  = "\033[1m"
_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED   = "\033[91m"
_RESET = "\033[0m"


def _color(val: float, thresholds=(70, 40)) -> str:
    """Green ≥ hi, Yellow ≥ lo, Red otherwise."""
    hi, lo = thresholds
    if val >= hi:
        return f"{_GREEN}{val:>7.2f}%{_RESET}"
    if val >= lo:
        return f"{_YELLOW}{val:>7.2f}%{_RESET}"
    return f"{_RED}{val:>7.2f}%{_RESET}"


def print_report(label: str, metrics: dict) -> None:
    """Pretty-print a single evaluation run."""
    w = 72
    print()
    print(f"{_BOLD}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}  📊  {label}{_RESET}")
    print(f"{_BOLD}{'═' * w}{_RESET}")
    print(f"  Queries evaluated        : {metrics['total_queries']}")
    print(f"  Top-K depth              : {metrics['top_k']}")
    print(f"  Index coverage           : {_color(metrics['index_coverage_%'])}")
    print(f"{'─' * w}")

    # Recall table
    for key in sorted(k for k in metrics if k.startswith("recall@")):
        print(f"  {key:<24} : {_color(metrics[key])}")
    print(f"  {'MRR':<24} : {_BOLD}{metrics['mrr']:.4f}{_RESET}")
    ndcg_key = [k for k in metrics if k.startswith("ndcg@")]
    if ndcg_key:
        print(f"  {ndcg_key[0]:<24} : {_BOLD}{metrics[ndcg_key[0]]:.4f}{_RESET}")

    # Latency
    lat = metrics["latency"]
    print(f"{'─' * w}")
    print(f"  ⏱️  Latency (embed + search)")
    print(f"      Mean  : {lat['mean_ms']:>8.1f} ms")
    print(f"      p50   : {lat['p50_ms']:>8.1f} ms")
    print(f"      p95   : {lat['p95_ms']:>8.1f} ms")
    print(f"      Max   : {lat['max_ms']:>8.1f} ms")

    # Category breakdown
    cats = metrics.get("category_breakdown", {})
    if cats:
        print(f"{'─' * w}")
        print(f"  📂  Per-Clause Category Breakdown (by Recall@1)")
        print(f"  {'Category':<35} {'n':>4}  {'R@1':>8}  {'R@5':>8}  {'MRR':>7}")
        for cat, cv in sorted(cats.items(), key=lambda x: -x[1]["recall@1"]):
            r1 = cv["recall@1"]
            r5 = cv["recall@5"]
            tag = "🟢" if r1 >= 70 else ("🟡" if r1 >= 40 else "🔴")
            print(f"  {tag} {cat:<33} {cv['n']:>4}  {r1:>7.1f}%  {r5:>7.1f}%  {cv['mrr']:>7.4f}")

    # Sample failures
    fails = metrics.get("sample_failures", [])
    if fails:
        print(f"{'─' * w}")
        print(f"  🔍  Sample Failures (showing up to 5)")
        for f in fails[:5]:
            in_db = "in DB but missed" if f["in_index"] else "NOT in DB"
            print(f"    ❌  [{f['category']}]  score={f['top_score']:.4f}  ({in_db})")
            print(f"        Q: {f['search_query'][:90]}…")
            print(f"        A: {f['expected_answer'][:90]}…")
            print()

    print(f"{_BOLD}{'═' * w}{_RESET}")


def print_comparison(raw: dict, clean: dict) -> None:
    """Side-by-side summary of raw vs reformulated queries."""
    w = 72
    print()
    print(f"{_BOLD}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}  📊  A/B COMPARISON: Raw vs Reformulated Queries{_RESET}")
    print(f"{_BOLD}{'═' * w}{_RESET}")
    print(f"  {'Metric':<24} {'Raw':>10} {'Clean':>10} {'Δ':>10}")
    print(f"  {'─'*56}")

    for key in sorted(k for k in raw if k.startswith("recall@")):
        delta = clean[key] - raw[key]
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {key:<24} {raw[key]:>9.2f}% {clean[key]:>9.2f}% {arrow}{abs(delta):>8.2f}%")

    delta_mrr = clean["mrr"] - raw["mrr"]
    arrow = "▲" if delta_mrr > 0 else ("▼" if delta_mrr < 0 else "─")
    print(f"  {'MRR':<24} {raw['mrr']:>10.4f} {clean['mrr']:>10.4f} {arrow}{abs(delta_mrr):>9.4f}")

    print(f"  {'─'*56}")
    print(f"  Failures: Raw={raw['failure_count']}  Clean={clean['failure_count']}")
    print(f"{_BOLD}{'═' * w}{_RESET}")


# =====================================================================
# 🚀  MAIN
# =====================================================================

def build_test_suite(dataset, max_samples: int) -> list[dict]:
    """
    Extract QA pairs that have a non-empty ground-truth answer and
    attach clause category metadata derived from the question ID.
    """
    suite: list[dict] = []
    for item in dataset:
        answers = item.get("answers", {})
        texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not texts or not texts[0].strip():
            continue

        qa_id = item.get("id", "")
        category = extract_clause_category(qa_id)
        # Fallback: extract category from the question text if ID parsing failed
        if category == "Unknown":
            category = _extract_category_from_question(item["question"])
        suite.append({
            "question": item["question"],
            "answer":   texts[0].strip(),
            "context":  item["context"],
            "id":       qa_id,
            "category": category,
        })

        if len(suite) >= max_samples:
            break

    return suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality of the Legal AI RAG pipeline."
    )
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of QA pairs to evaluate (default: 100)")
    parser.add_argument("--top-k",   type=int, default=10,
                        help="Maximum retrieval depth (default: 10)")
    parser.add_argument("--mode",    choices=["raw", "clean", "both"], default="both",
                        help="Query mode: raw / clean (reformulated) / both (default: both)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-query results")
    parser.add_argument("--out-dir", default="reports",
                        help="Directory for JSON report output (default: reports/)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────
    uri = _load_env_uri()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"⚡  Device: [{device.upper()}]")

    print("🧠  Loading embedding model…")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    print("📥  Connecting to MongoDB Atlas…")
    client = MongoClient(uri, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    collection = db[COLLECTION]
    total_docs = collection.count_documents({})
    print(f"    Collection contains {total_docs:,} chunks.")
    if total_docs == 0:
        print("❌  Collection is empty — run indexing first.")
        sys.exit(1)

    print("📥  Loading CUAD dataset…")
    dataset = load_cuad_dataset()
    test_suite = build_test_suite(dataset, args.samples)
    print(f"🚀  Evaluating {len(test_suite)} QA pairs  (top-K={args.top_k}, mode={args.mode})\n")

    # ── Run evaluations ──────────────────────────────────────────────
    report: dict = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "device":      device,
        "model":       EMBED_MODEL,
        "db_chunks":   total_docs,
        "num_queries":  len(test_suite),
        "top_k":        args.top_k,
    }

    if args.mode in ("raw", "both"):
        print("🧪  Running evaluation: RAW queries…")
        raw_metrics = evaluate(
            test_suite, collection, embedder,
            top_k=args.top_k, use_compression=False, verbose=args.verbose,
        )
        print_report("Evaluation A — Raw CUAD Queries", raw_metrics)
        report["raw"] = raw_metrics

    if args.mode in ("clean", "both"):
        print("🧪  Running evaluation: REFORMULATED queries…")
        clean_metrics = evaluate(
            test_suite, collection, embedder,
            top_k=args.top_k, use_compression=True, verbose=args.verbose,
        )
        print_report("Evaluation B — Reformulated Queries", clean_metrics)
        report["clean"] = clean_metrics

    if args.mode == "both":
        print_comparison(raw_metrics, clean_metrics)

    # ── Save JSON report ─────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.out_dir, f"retrieval_eval_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Full report saved → {report_path}")


if __name__ == "__main__":
    main()
