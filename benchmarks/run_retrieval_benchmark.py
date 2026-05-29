from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corecoder.context.models import ContextRequest
from corecoder.context.retriever import RepositoryContextRetriever
from corecoder.codebase.indexing.index import RepoIndex

DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "data" / "retrieval_eval.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "retrieval_report.json"


@dataclass
class BenchmarkCase:
    id: str
    query: str
    expected_files: list[str]
    notes: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline retrieval benchmarks against the current repository.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSONL retrieval benchmark dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the JSON report file to write.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Primary cutoff used for Recall@K in the summary report.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cases.append(BenchmarkCase(**payload))
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def extract_ranked_files(fragments) -> list[str]:
    ranked_files: list[str] = []
    for frag in fragments:
        for rf in frag.metadata.get("ranked_files", []):
            path = rf.get("path")
            if path and path not in ranked_files:
                ranked_files.append(path)
    return ranked_files


def reciprocal_rank(actual: list[str], expected: set[str]) -> float:
    for idx, path in enumerate(actual, start=1):
        if path in expected:
            return 1.0 / idx
    return 0.0


def recall_at_k(actual: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(actual[:k]) & expected) / len(expected)


def hit_at_k(actual: list[str], expected: set[str], k: int) -> float:
    return 1.0 if set(actual[:k]) & expected else 0.0


def evaluate_case(
    retriever: RepositoryContextRetriever,
    case: BenchmarkCase,
    top_k: int,
) -> dict:
    request = ContextRequest(
        task_title=case.query,
        task_description=case.query,
        goal=case.query,
    )
    fragments = retriever.retrieve(request)
    meta = retriever.get_last_retrieval_meta()
    ranked_files = extract_ranked_files(fragments)
    expected = set(case.expected_files)

    return {
        "id": case.id,
        "query": case.query,
        "notes": case.notes,
        "expected_files": case.expected_files,
        "retrieved_files": ranked_files,
        "metrics": {
            "hit_at_1": hit_at_k(ranked_files, expected, 1),
            "hit_at_3": hit_at_k(ranked_files, expected, 3),
            "hit_at_5": hit_at_k(ranked_files, expected, 5),
            f"recall_at_{top_k}": recall_at_k(ranked_files, expected, top_k),
            "mrr": reciprocal_rank(ranked_files, expected),
        },
        "retrieval_meta": {
            "intent_family": meta.intent.family if meta else "",
            "intent_type": meta.intent.type if meta else "",
            "total_files_considered": meta.total_files_considered if meta else 0,
            "total_files_ranked": meta.total_files_ranked if meta else 0,
            "retrieval_time_ms": round(meta.retrieval_time_ms, 3) if meta else 0.0,
            "pipeline_stages": meta.pipeline_stages if meta else [],
        },
    }


def summarize(results: list[dict], top_k: int) -> dict:
    metric_key = f"recall_at_{top_k}"
    return {
        "cases": len(results),
        "hit_at_1": round(mean(r["metrics"]["hit_at_1"] for r in results), 4),
        "hit_at_3": round(mean(r["metrics"]["hit_at_3"] for r in results), 4),
        "hit_at_5": round(mean(r["metrics"]["hit_at_5"] for r in results), 4),
        metric_key: round(mean(r["metrics"][metric_key] for r in results), 4),
        "mrr": round(mean(r["metrics"]["mrr"] for r in results), 4),
        "avg_retrieval_time_ms": round(
            mean(r["retrieval_meta"]["retrieval_time_ms"] for r in results), 3
        ),
        "avg_files_considered": round(
            mean(r["retrieval_meta"]["total_files_considered"] for r in results), 2
        ),
        "avg_files_ranked": round(
            mean(r["retrieval_meta"]["total_files_ranked"] for r in results), 2
        ),
    }


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_dataset(dataset_path)

    repo_index = RepoIndex(REPO_ROOT)
    repo_index.build()
    retriever = RepositoryContextRetriever(
        working_dir=str(REPO_ROOT),
        repo_index=repo_index,
    )

    results = [evaluate_case(retriever, case, args.top_k) for case in cases]
    summary = summarize(results, args.top_k)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "dataset": str(dataset_path),
        "top_k": args.top_k,
        "summary": summary,
        "results": results,
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Retrieval benchmark complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
