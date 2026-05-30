"""Evaluation helpers for Retrieval V2."""

from __future__ import annotations

from statistics import mean

from corecoder.retrieval.models import RetrievalMetrics


def summarize_metrics(metrics: list[RetrievalMetrics]) -> dict[str, float]:
    """Aggregate a list of RetrievalMetrics for reporting."""
    if not metrics:
        return {
            "mrr": 0.0,
            "context_size": 0.0,
            "token_cost": 0.0,
        }

    ks = sorted({k for metric in metrics for k in metric.hit_rate_at_k})
    summary: dict[str, float] = {
        "mrr": mean(metric.mrr for metric in metrics),
        "context_size": mean(metric.context_size for metric in metrics),
        "token_cost": mean(metric.token_cost for metric in metrics),
    }
    for k in ks:
        summary[f"hit_rate@{k}"] = mean(metric.hit_rate_at_k.get(k, 0.0) for metric in metrics)
        summary[f"recall@{k}"] = mean(metric.recall_at_k.get(k, 0.0) for metric in metrics)
    return summary
