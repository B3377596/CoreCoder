# Retrieval Benchmarks

This directory contains a lightweight, offline benchmark suite for CoreCoder's
symbolic repository retrieval layer.

## What it measures

- `Hit@1`, `Hit@3`, `Hit@5`: whether any expected file appears in the top-k
- `Recall@5`: how many expected files are recovered in the top-5
- `MRR`: reciprocal rank of the first expected file
- Latency and candidate/ranking counts from `RetrievalMeta`

## Run

```bash
uv run python benchmarks/run_retrieval_benchmark.py
```

Optional arguments:

```bash
uv run python benchmarks/run_retrieval_benchmark.py ^
  --dataset benchmarks/data/retrieval_eval.jsonl ^
  --output benchmarks/retrieval_report.json
```

## Dataset format

Each line in `data/retrieval_eval.jsonl` is a JSON object:

```json
{
  "id": "cli_overview",
  "query": "How does the CLI work?",
  "expected_files": ["corecoder/cli.py", "corecoder/__main__.py"],
  "notes": "CLI entrypoints and user-facing command flow"
}
```

The benchmark is intentionally repository-local: it evaluates whether the
retriever can surface the right files for this codebase's own architecture.
