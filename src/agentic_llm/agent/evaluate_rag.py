"""
evaluate_rag.py — Đánh giá RAG pipeline trên 3 bộ dataset: law, admission, tour.

Metrics (qua RAGAS):
  - faithfulness
  - answer_correctness
  - answer_relevancy
  - context_precision
  - context_recall

Cách chạy:
    python evaluate_rag.py \
        --law-file   law_questions.json \
        --adm-file   admission_questions.json \
        --tour-file  tour_question.json \
        --output     eval_results.csv \
        --model      gpt-4o-mini \
        --eval-model gpt-4o-mini \
        --max-items  0
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from tqdm.asyncio import tqdm as atqdm

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── RAGAS imports ─────────────────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_V2 = True
except ImportError:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset as HFDataset
    RAGAS_V2 = False
    logger.warning("RAGAS v0.1 detected — using legacy API.")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from agent_routing import create_router_agent


# ── Vietnamese prompt override cho AnswerRelevancy ────────────────────────────
# RAGAS mặc định sinh câu hỏi ngược bằng tiếng Anh → cosine thấp với question
# tiếng Việt. Override prompt để LLM sinh câu hỏi bằng tiếng Việt.
def _make_vietnamese_answer_relevancy(ragas_llm, ragas_emb):
    """
    Trả về AnswerRelevancy với prompt tiếng Việt thay vì mặc định tiếng Anh.
    Tương thích với cả cấu trúc prompt cũ lẫn mới của RAGAS.
    """
    metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)

    VI_INSTRUCTION = (
        "Hãy tạo ra một câu hỏi bằng tiếng Việt phù hợp với câu trả lời được cung cấp. "
        "Đồng thời xác định xem câu trả lời có mang tính né tránh hoặc không rõ ràng không. "
        "Trả về noncommittal = 1 nếu câu trả lời vague hoặc né tránh "
        "(ví dụ: 'Tôi không biết', 'Có thể'); trả về 0 nếu câu trả lời cụ thể và rõ ràng."
    )

    # Thử các attribute name RAGAS dùng qua các phiên bản khác nhau
    for attr in ("question_generation", "question_generation_prompt", "_question_generation"):
        prompt_obj = getattr(metric, attr, None)
        if prompt_obj is not None and hasattr(prompt_obj, "instruction"):
            prompt_obj.instruction = VI_INSTRUCTION
            break
    else:
        # Fallback: ghi thẳng vào __dict__ nếu không tìm thấy
        import logging as _log
        _log.getLogger(__name__).warning(
            "Could not find question_generation prompt attribute — "
            "AnswerRelevancy will use default English prompt."
        )

    return metric



# ── QA logger riêng — ghi ra file ngay khi có kết quả ────────────────────────
def _get_qa_logger(log_path: str) -> logging.Logger:
    qa_log = logging.getLogger("qa_trace")
    qa_log.setLevel(logging.DEBUG)
    qa_log.propagate = False
    if not qa_log.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        qa_log.addHandler(fh)
    return qa_log


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load datasets
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    return items


def extract_gold_context_texts(item: dict) -> list[str]:
    return [c.get("text", "") for c in item.get("gold_contexts", []) if c.get("text")]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Run agent via astream_events
# ═════════════════════════════════════════════════════════════════════════════

async def run_single(
    graph,
    question: str,
    semaphore: asyncio.Semaphore,
    qa_logger: logging.Logger,
    item_index: int,
    domain: str,
    retries: int = 2,
    delay: float = 5.0,
) -> tuple[str, list[str]]:
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                retrieved_contexts: list[str] = []
                final_answer = ""
                t_start = asyncio.get_event_loop().time()

                async for event in graph.astream_events(
                    {
                        "messages": [HumanMessage(content=question)],
                        "hop_count": 0,
                        "thinking_streamed": False,
                    },
                    version="v2",
                ):
                    kind = event.get("event", "")

                    # ── Capture tool output ──────────────────────────────
                    if kind == "on_tool_end":
                        output = event["data"].get("output", "")
                        if output:
                            if hasattr(output, "content"):
                                text = output.content
                            elif isinstance(output, str):
                                text = output
                            else:
                                text = json.dumps(output, ensure_ascii=False, default=str)
                            if text:
                                retrieved_contexts.append(text)

                    # ── Capture final LLM answer ─────────────────────────
                    elif kind == "on_chat_model_end":
                        msg = event["data"].get("output")
                        if msg is None:
                            continue
                        if getattr(msg, "tool_calls", None):
                            continue
                        content = getattr(msg, "content", "")
                        if content:
                            final_answer = content

                # ── Log Q&A ngay sau khi có kết quả ─────────────────────
                elapsed = asyncio.get_event_loop().time() - t_start
                qa_logger.info(
                    "\n%s\n"
                    "[%s] #%d  (%.2fs)\n"
                    "Q: %s\n"
                    "A: %s\n"
                    "CONTEXTS (%d):\n%s\n"
                    "%s",
                    "=" * 70,
                    domain.upper(),
                    item_index,
                    elapsed,
                    question,
                    final_answer if final_answer else "[no answer]",
                    len(retrieved_contexts),
                    "\n---\n".join(retrieved_contexts) if retrieved_contexts else "[none]",
                    "=" * 70,
                )
                logger.info("[%s] #%d done — %.2fs | %d contexts | answer %d chars",
                            domain.upper(), item_index, elapsed,
                            len(retrieved_contexts), len(final_answer))

                return final_answer, retrieved_contexts

            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %r: %s",
                    attempt + 1, retries + 1, question[:60], exc,
                )
                if attempt < retries:
                    await asyncio.sleep(delay * (attempt + 1))
                else:
                    logger.error("All retries exhausted for: %r", question[:60])
                    elapsed = asyncio.get_event_loop().time() - t_start
                    qa_logger.info(
                        "\n%s\n[%s] #%d FAILED (%.2fs)\nQ: %s\nError: %s\n%s",
                        "=" * 70, domain.upper(), item_index, elapsed, question, exc, "=" * 70,
                    )
                    return "", []


async def run_dataset(
    graph,
    items: list[dict],
    domain: str,
    qa_logger: logging.Logger,
    max_items: int = 0,
    concurrency: int = 3,
) -> list[dict]:
    subset = items[:max_items] if max_items > 0 else items
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_single(graph, item["question"], semaphore, qa_logger, i, domain)
        for i, item in enumerate(subset)
    ]
    results = await atqdm.gather(*tasks, desc=f"[{domain}]")

    rows = []
    for item, (answer, contexts) in zip(subset, results):
        rows.append(
            {
                "question":           item["question"],
                "answer":             answer,
                "contexts":           contexts if contexts else ["[no context retrieved]"],
                "ground_truth":       item.get("gold_answer", ""),
                "reference_contexts": extract_gold_context_texts(item),
                "difficulty":         item.get("difficulty", ""),
            }
        )
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# 3. RAGAS evaluation
# ═════════════════════════════════════════════════════════════════════════════

def _build_metrics(ragas_llm, ragas_emb):
    if RAGAS_V2:
        return [
            Faithfulness(llm=ragas_llm),
            AnswerCorrectness(llm=ragas_llm, embeddings=ragas_emb),
            _make_vietnamese_answer_relevancy(ragas_llm, ragas_emb),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]
    return [faithfulness, answer_correctness, answer_relevancy,
            context_precision, context_recall]


def evaluate_with_ragas(rows: list[dict], ragas_llm, ragas_emb) -> pd.DataFrame:
    metrics = _build_metrics(ragas_llm, ragas_emb)

    if RAGAS_V2:
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
                reference_contexts=r["reference_contexts"] or r["contexts"],
            )
            for r in rows
        ]
        result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
        df = result.to_pandas()
    else:
        hf_dataset = HFDataset.from_dict(
            {
                "question":      [r["question"]       for r in rows],
                "answer":        [r["answer"]         for r in rows],
                "contexts":      [r["contexts"]       for r in rows],
                "ground_truths": [[r["ground_truth"]] for r in rows],
            }
        )
        result = evaluate(hf_dataset, metrics=metrics)
        df = result.to_pandas()

    df["question"]           = [r["question"]                 for r in rows]
    df["difficulty"]         = [r["difficulty"]               for r in rows]
    df["answer"]             = [r["answer"]                   for r in rows]
    df["ground_truth"]       = [r["ground_truth"]             for r in rows]
    df["retrieved_contexts"] = ["\n---\n".join(r["contexts"]) for r in rows]

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 4. Save results — CSV per domain + summary CSV
# ═════════════════════════════════════════════════════════════════════════════

METRIC_COLS = [
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def save_results(results: dict[str, pd.DataFrame], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Mỗi domain ra 1 file CSV riêng
    stem   = out.stem   # e.g. "eval_results"
    suffix = out.suffix if out.suffix == ".csv" else ".csv"
    parent = out.parent

    for domain, df in results.items():
        domain_path = parent / f"{stem}_{domain}{suffix}"
        df.to_csv(domain_path, index=False, encoding="utf-8-sig")
        logger.info("Saved [%s] → %s", domain.upper(), domain_path)

    # Summary CSV
    summary_rows = []
    for domain, df in results.items():
        row = {"domain": domain, "n_samples": len(df)}
        for col in METRIC_COLS:
            if col in df.columns:
                row[col] = round(float(df[col].mean()), 4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    summary_path = parent / f"{stem}_summary{suffix}"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("Summary saved → %s", summary_path)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═════════════════════════════════════════════════════════════════════════════

async def main(args):
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL") or None

    agent_graph = create_router_agent(
        model=args.model,
        openai_api_key=api_key,
        openai_base_url=base_url,
    )

    _eval_llm = ChatOpenAI(model=args.eval_model, temperature=0,
                           api_key=api_key, base_url=base_url)
    _eval_emb = OpenAIEmbeddings(model="text-embedding-3-small",
                                 api_key=api_key, base_url=base_url)

    ragas_llm = LangchainLLMWrapper(_eval_llm) if RAGAS_V2 else _eval_llm
    ragas_emb = LangchainEmbeddingsWrapper(_eval_emb) if RAGAS_V2 else _eval_emb

    datasets: dict[str, list[dict]] = {}
    for key, path in [("law", args.law_file), ("admission", args.adm_file), ("tour", args.tour_file)]:
        if path:
            datasets[key] = load_dataset(path)

    if not datasets:
        logger.error("No dataset files provided.")
        sys.exit(1)

    # QA trace log — ghi liên tục vào file trong suốt quá trình chạy
    out      = Path(args.output)
    log_path = out.parent / f"{out.stem}_qa_trace.log"
    qa_logger = _get_qa_logger(str(log_path))
    logger.info("QA trace log → %s", log_path)

    all_results: dict[str, pd.DataFrame] = {}

    for domain, items in datasets.items():
        logger.info("=" * 50)
        logger.info("[%s] Running agent on %d items", domain.upper(), len(items))
        logger.info("=" * 50)

        rows = await run_dataset(
            agent_graph, items,
            domain=domain,
            qa_logger=qa_logger,
            max_items=args.max_items,
            concurrency=args.concurrency,
        )

        # Raw JSON backup
        raw_path = out.parent / f"{out.stem}_{domain}_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("Raw rows saved → %s", raw_path)

        logger.info("[%s] Running RAGAS on %d rows...", domain.upper(), len(rows))
        try:
            df = evaluate_with_ragas(rows, ragas_llm, ragas_emb)
        except Exception as exc:
            logger.error("RAGAS failed for [%s]: %s", domain, exc)
            df = pd.DataFrame(rows)

        all_results[domain] = df

    save_results(all_results, args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS.")

    # Chế độ thông thường
    p.add_argument("--law-file")
    p.add_argument("--adm-file")
    p.add_argument("--tour-file")

    # Chế độ re-eval từ raw JSON
    p.add_argument(
        "--from-raw", nargs="+", metavar="DOMAIN:PATH",
        help="Tính lại metrics từ file raw JSON đã có, không chạy agent. "
             "Ví dụ: --from-raw law:eval_results_law_raw.json tour:eval_results_tour_raw.json",
    )

    p.add_argument("--output",      default="eval_results.csv")
    p.add_argument("--model",       default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    p.add_argument("--eval-model",  default="gpt-4o-mini")
    p.add_argument("--max-items",   type=int, default=0)
    p.add_argument("--concurrency", type=int, default=3)
    return p.parse_args()


async def main_from_raw(args, ragas_llm, ragas_emb):
    all_results = {}
    for entry in args.from_raw:
        if ":" not in entry:
            logger.error("--from-raw format sai, cần DOMAIN:PATH, got: %r", entry)
            continue
        domain, path = entry.split(":", 1)
        if not Path(path).exists():
            logger.error("File không tồn tại: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        if args.max_items > 0:
            rows = rows[:args.max_items]
        logger.info("[%s] Loaded %d rows from %s", domain.upper(), len(rows), path)
        logger.info("[%s] Running RAGAS on %d rows...", domain.upper(), len(rows))
        try:
            df = evaluate_with_ragas(rows, ragas_llm, ragas_emb)
        except Exception as exc:
            logger.error("RAGAS failed for [%s]: %s", domain, exc)
            df = pd.DataFrame(rows)
        all_results[domain] = df
    if all_results:
        save_results(all_results, args.output)
    else:
        logger.error("Không có domain nào được load thành công.")


async def main(args):
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL") or None

    _eval_llm = ChatOpenAI(model=args.eval_model, temperature=0,
                           api_key=api_key, base_url=base_url)
    _eval_emb = OpenAIEmbeddings(model="text-embedding-3-small",
                                 api_key=api_key, base_url=base_url)
    ragas_llm = LangchainLLMWrapper(_eval_llm) if RAGAS_V2 else _eval_llm
    ragas_emb = LangchainEmbeddingsWrapper(_eval_emb) if RAGAS_V2 else _eval_emb

    # Chế độ re-eval — không cần khởi động agent
    if args.from_raw:
        await main_from_raw(args, ragas_llm, ragas_emb)
        return

    # Chế độ thông thường
    agent_graph = create_router_agent(
        model=args.model, openai_api_key=api_key, openai_base_url=base_url,
    )
    datasets = {}
    for key, path in [("law", args.law_file), ("admission", args.adm_file), ("tour", args.tour_file)]:
        if path:
            datasets[key] = load_dataset(path)
    if not datasets:
        logger.error("No dataset files provided.")
        sys.exit(1)

    out       = Path(args.output)
    log_path  = out.parent / f"{out.stem}_qa_trace.log"
    qa_logger = _get_qa_logger(str(log_path))
    logger.info("QA trace log -> %s", log_path)
    all_results = {}

    for domain, items in datasets.items():
        logger.info("=" * 50)
        logger.info("[%s] Running agent on %d items", domain.upper(), len(items))
        logger.info("=" * 50)
        rows = await run_dataset(
            agent_graph, items, domain=domain, qa_logger=qa_logger,
            max_items=args.max_items, concurrency=args.concurrency,
        )
        raw_path = out.parent / f"{out.stem}_{domain}_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("Raw rows saved -> %s", raw_path)
        logger.info("[%s] Running RAGAS on %d rows...", domain.upper(), len(rows))
        try:
            df = evaluate_with_ragas(rows, ragas_llm, ragas_emb)
        except Exception as exc:
            logger.error("RAGAS failed for [%s]: %s", domain, exc)
            df = pd.DataFrame(rows)
        all_results[domain] = df

    save_results(all_results, args.output)

if __name__ == "__main__":
    asyncio.run(main(parse_args()))