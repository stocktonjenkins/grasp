"""Command line: `uv run grasp <command>`. See README.md for the paper's run order."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from neo4j import Driver
from tqdm import tqdm

from grasp.config import RunConfig
from grasp.datasets import load_corpus, load_questions
from grasp.embeddings import Embedder
from grasp.eval.runner import as_completed_or_cancel, evaluate, read_results, write_results
from grasp.graph.server import open_graph
from grasp.indexing.extraction import Extractor
from grasp.indexing.indexer import Indexer, deadlock_retry, embed_passages
from grasp.llm import Backbone
from grasp.methods import GraspMethod, Method
from grasp.tokens import TokenUsage

INDEX_STATS = "index_stats.json"


def backbone(config: RunConfig) -> Backbone:
    return Backbone(model=config.model, provider=config.provider, base_url=config.base_url)


def embedder(config: RunConfig) -> Embedder:
    return Embedder(config.embedding_model, config.embedding_provider)


def build_corpus_graph(config: RunConfig, driver: Driver) -> TokenUsage:
    """Indexes the passages not yet in the graph, `index_workers` batches at a time.

    Construction tokens accumulate in `index_stats.json`; a failed batch is logged and
    picked up by the next run."""
    settings = config.grasp
    passages = load_corpus(config.data)
    batches = [passages[begin : begin + settings.batch_size] for begin in range(0, len(passages), settings.batch_size)]
    stats_path = config.neo4j_dir / INDEX_STATS
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {"construction_token_usage": {}, "seconds": 0.0}
    usage = TokenUsage.from_dict(stats["construction_token_usage"])
    indexer = Indexer(driver, Extractor(settings.extraction, backbone(config), settings.batch_size), embedder(config), config.embedding_dim)
    index_batch = deadlock_retry(indexer.index)
    start, failed = time.time(), 0
    with ThreadPoolExecutor(max_workers=settings.index_workers) as pool, tqdm(total=len(batches), desc="Indexing", unit="batch") as bar:
        for future in as_completed_or_cancel(pool, [pool.submit(index_batch, batch, skip_existing=True) for batch in batches]):
            try:
                usage += future.result()
            except Exception as error:  # noqa: BLE001
                failed += 1
                tqdm.write(f"Batch failed (re-run to retry it): {error}")
            stats = {"construction_token_usage": usage.to_dict(), "seconds": stats["seconds"] + time.time() - start}
            stats_path.write_text(json.dumps(stats, indent=1))
            start = time.time()
            bar.update(1)
    if failed:
        raise SystemExit(f"{failed} of {len(batches)} batches failed; re-run the same command to index them.")
    return usage


def shares_corpus_graph(config: RunConfig) -> bool:
    return config.grasp is not None and not config.grasp.per_question


def index_corpus(args: argparse.Namespace) -> None:
    config = RunConfig.load(args.config)
    if not shares_corpus_graph(config):
        raise SystemExit("`index` builds the shared corpus graph; per-question configs index during `eval`.")
    config.neo4j_dir.mkdir(parents=True, exist_ok=True)
    with open_graph(config.neo4j_dir) as driver:
        usage = build_corpus_graph(config, driver)
    print(f"Indexed the {config.dataset} corpus; construction tokens: {usage.total_tokens:,}")


def run_eval(args: argparse.Namespace) -> None:
    config = RunConfig.load(args.config)
    with open_graph(config.neo4j_dir) if config.grasp else nullcontext() as driver:
        answer_questions(config, driver, args, resume=args.resume)


def run_all(args: argparse.Namespace) -> None:
    config = RunConfig.load(args.config)
    if config.grasp is None:
        raise SystemExit("`run` is for GRASP configs; use `eval` for baselines.")
    config.neo4j_dir.mkdir(parents=True, exist_ok=True)
    with open_graph(config.neo4j_dir) as driver:
        if shares_corpus_graph(config):
            build_corpus_graph(config, driver)
        if config.grasp.graph == "dense_passage":
            embed_passages(driver, embedder(config), config.embedding_dim)
        answer_questions(config, driver, args, resume=True)


def answer_questions(config: RunConfig, driver: Driver | None, args: argparse.Namespace, *, resume: bool) -> None:
    output = args.output or config.output
    questions = load_questions(config.dataset, config.data, config.chunking)[: args.limit]
    method, index_usage = build_method(config, driver)
    results = evaluate(
        method,
        questions,
        output,
        config=config.model_dump(mode="json"),
        parallel=config.parallel,
        resume=resume,
        index_usage=index_usage,
    )
    summary = results["summary"]
    print(f"EM {summary['exact_match']:.3f}  F1 {summary['f1']:.3f}  tokens {summary['tokens']['total']['total_tokens']:,}  -> {output}")


def build_method(config: RunConfig, driver: Driver | None) -> tuple[Method, TokenUsage]:
    if config.baseline is not None:
        from grasp.baselines.dense import DenseIndex
        from grasp.baselines.vanilla import SelfAsk, Vanilla

        settings = config.baseline
        cache: list = []

        def dense_index() -> DenseIndex:
            if not cache:
                cache.append(DenseIndex.load(settings.index_path))
            return cache[0]

        if settings.variant == "self_ask":
            return SelfAsk(config.model, embedder(config), dense_index, settings.top_k, settings.max_hops), TokenUsage()
        return Vanilla(
            settings.variant, config.model, embedder(config), dense_index, settings.top_k, settings.global_retrieval
        ), TokenUsage()

    stats_path = config.neo4j_dir / INDEX_STATS
    index_usage = TokenUsage()
    if shares_corpus_graph(config) and stats_path.exists():
        index_usage = TokenUsage.from_dict(json.loads(stats_path.read_text())["construction_token_usage"])
    return GraspMethod(config.grasp, backbone(config), embedder(config), driver, config.embedding_dim), index_usage


def run_embed_passages(args: argparse.Namespace) -> None:
    config = RunConfig.load(args.config)
    with open_graph(config.neo4j_dir) as driver:
        print(f"Embedded {embed_passages(driver, embedder(config), config.embedding_dim)} passages")


def run_dense_index(args: argparse.Namespace) -> None:
    from grasp.baselines.dense import build_dense_index

    if args.corpus:
        scoped = [(hashlib.sha256(text.encode()).hexdigest()[:16], title, text) for title, text in load_corpus(args.data)]
    else:
        scoped = [(q.id, p.title, p.text) for q in load_questions(args.dataset, args.data) for p in q.passages]
    added = build_dense_index(scoped, Embedder(), args.out)
    print(f"Added {added} chunks to {args.out}")


def run_judge(args: argparse.Namespace) -> None:
    from grasp.eval.judge import Judge

    results = read_results(args.results)
    results["questions"] = Judge(args.model).judge(results["questions"])
    results["summary"]["llm_judge"] = sum(r["llm_judge"] for r in results["questions"]) / len(results["questions"])
    write_results(args.results, results)
    print(f"LLM judge: {results['summary']['llm_judge']:.3f}")


def run_difficulty(args: argparse.Namespace) -> None:
    from grasp.eval.success_economy import estimate_difficulty

    questions = load_questions(args.dataset, args.data)
    difficulty = estimate_difficulty(
        questions, args.model, args.provider, args.base_url, with_context=args.with_context, workers=args.workers
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(difficulty, indent=1))


def run_success_economy(args: argparse.Namespace) -> None:
    from grasp.eval.success_economy import from_results

    score = from_results(read_results(args.results), json.loads(args.difficulty.read_text()))
    print(f"C_w = {score['success_economy'] / 1e3:.1f}k  (95% CI {score['ci_low'] / 1e3:.1f}-{score['ci_high'] / 1e3:.1f}k)")


def run_recall(args: argparse.Namespace) -> None:
    from grasp.eval.recall import question_recall, rewrite_gold_subquestions
    from grasp.graph.retrieval import retrieve_passages

    if args.rewrite:
        items = json.loads((args.data / "qa.json").read_text())
        args.rewritten.write_text(
            json.dumps(rewrite_gold_subquestions(items, args.model, n=-1, workers=args.workers), indent=2, ensure_ascii=False)
        )
    config = RunConfig.load(args.config)
    items = json.loads(args.rewritten.read_text())
    with open_graph(config.neo4j_dir) as driver:
        embed = embedder(config)

        def retrieve(query: str, keywords: list[str]) -> list[dict]:
            rows = retrieve_passages(
                driver, embed, query, keywords, scoring=args.scoring, n_candidates=100, lexical_weight=args.lexical_weight
            )
            return sorted(rows, key=lambda row: row["score"], reverse=True)

        def score(item: dict) -> dict | None:
            return question_recall(item, retrieve, args.mode, args.k)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            scores = [s for s in tqdm(pool.map(score, items), total=len(items)) if s]
    print(
        f"Recall@{args.k} ({args.mode}, {args.scoring}): {sum(s['recall'] for s in scores) / len(scores):.3f} over {len(scores)} questions"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="grasp")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("run", help="Index (if needed) and answer every question, continuing any earlier run.")
    command.add_argument("config", type=Path)
    command.add_argument("--limit", type=int, default=None)
    command.add_argument("--output", type=Path, default=None)
    command.set_defaults(run=run_all)

    command = commands.add_parser("index", help="Build the shared corpus graph (retrieval setting).")
    command.add_argument("config", type=Path)
    command.set_defaults(run=index_corpus)

    command = commands.add_parser("eval", help="Answer every question and write a results file.")
    command.add_argument("config", type=Path)
    command.add_argument("--limit", type=int, default=None)
    command.add_argument("--resume", action="store_true", help="Skip questions already answered without error.")
    command.add_argument("--output", type=Path, default=None)
    command.set_defaults(run=run_eval)

    command = commands.add_parser("embed-passages", help="Embed passages for the dense-passage ablation.")
    command.add_argument("config", type=Path)
    command.set_defaults(run=run_embed_passages)

    command = commands.add_parser("dense-index", help="Build the flat dense index used by the baselines.")
    command.add_argument("--dataset", required=True)
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--corpus", action="store_true", help="Index corpus.json instead of per-question passages.")
    command.set_defaults(run=run_dense_index)

    command = commands.add_parser("judge", help="Add LLM-judge verdicts to a results file.")
    command.add_argument("results", type=Path)
    command.add_argument("--model", default="gemini-3-flash-preview")
    command.set_defaults(run=run_judge)

    command = commands.add_parser("difficulty", help="Estimate r_i by sampling the backbone zero-shot.")
    command.add_argument("--dataset", required=True)
    command.add_argument("--data", type=Path, default=None)
    command.add_argument("--model", default="gemini-3-flash-preview")
    command.add_argument("--provider", default="gemini", choices=["gemini", "vllm"])
    command.add_argument("--base-url", default=None)
    command.add_argument("--with-context", action="store_true")
    command.add_argument("--workers", type=int, default=32)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=run_difficulty)

    command = commands.add_parser("cw", help="Success economy with its bootstrap 95%% CI.")
    command.add_argument("results", type=Path)
    command.add_argument("--difficulty", type=Path, required=True)
    command.set_defaults(run=run_success_economy)

    command = commands.add_parser("recall", help="Table 2 passage recall over a corpus graph.")
    command.add_argument("config", type=Path)
    command.add_argument("--rewritten", type=Path, required=True, help="rewritten_subquestions.json")
    command.add_argument("--rewrite", action="store_true", help="First create --rewritten from --data/qa.json.")
    command.add_argument("--data", type=Path, default=None)
    command.add_argument("--model", default="gemini-3-flash-preview")
    command.add_argument("--workers", type=int, default=8)
    command.add_argument("--mode", choices=["decomposed", "single"], default="decomposed")
    command.add_argument("--scoring", choices=["rr", "count"], default="rr")
    command.add_argument("--k", type=int, default=5)
    command.add_argument("--lexical-weight", type=float, default=0.2, help="Eq. 1 lambda (Table 2: 0.2; QA runs: 0).")
    command.set_defaults(run=run_recall)

    args = parser.parse_args()
    args.run(args)
