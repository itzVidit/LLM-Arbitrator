"""
api/analytics.py
----------------
Analytics engine: computes an AnalyticsReport from the raw analytics_rows
stored in SQLite after each arbitration run.

All computation is pure Python — no extra SQL aggregation — so the logic
is easy to test and extend.  The single public function is:

    report = await compute_analytics(db)

Metrics computed
----------------
  Overview
    total arbitrations, approve/revise/reject counts and rates

  Score health
    avg overall score, avg quality score, avg confidence, avg agreement

  Pipeline behaviour
    adjudication rate, fast-path rate, avg latency, total fallbacks

  Per-critic (for each of the 5 dimensions)
    avg/min/max score, pass rate
    avg confidence
    total / avg issues raised, critical / major breakdown
    overrule count, overrule rate, raised vs lowered direction
    avg and p95 latency
    fallback count and rate
    most common issue severity categories

  Top highlights
    most issues raised by, most overruled, highest/lowest confidence,
    most common issue category
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from api.models import AnalyticsReport, CriticStats
from storage.database import Database


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _safe_mean(values: list[float | int]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_p95(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * 0.95) - 1)
    return round(sorted_v[idx], 1)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# ─────────────────────────────────────────────────────────────────────
#  Core computation
# ─────────────────────────────────────────────────────────────────────

async def compute_analytics(db: Database) -> AnalyticsReport:
    """
    Read all analytics_rows and the arbitrations summary table, then
    compute and return a fully populated AnalyticsReport.
    """
    rows = await db.get_analytics_rows()
    rec_counts = await db.get_recommendation_counts()
    total = await db.count_verdicts()

    # ── Overview ──────────────────────────────────────────────────────
    approve_n = rec_counts.get("APPROVE", 0)
    revise_n  = rec_counts.get("REVISE",  0)
    reject_n  = rec_counts.get("REJECT",  0)

    approve_rate = _rate(approve_n, total)
    revise_rate  = _rate(revise_n,  total)
    reject_rate  = _rate(reject_n,  total)

    # ── Aggregate score / confidence / agreement from arbitrations ────
    # We need these from the arbitrations table which isn't in analytics_rows.
    # Re-query the list (lightweight — no verdict_json).
    summaries = await db.list_verdicts(limit=200)
    if len(summaries) < total:
        # Fetch more pages if needed
        all_summaries = list(summaries)
        offset = 200
        while offset < total:
            page = await db.list_verdicts(limit=200, offset=offset)
            all_summaries.extend(page)
            offset += 200
    else:
        all_summaries = summaries

    overall_scores = [r["overall_score"] for r in all_summaries]
    quality_scores  = [r["quality_score"]  for r in all_summaries if r["quality_score"] > 0]
    confidences     = [r["overall_confidence"] for r in all_summaries]
    agreements      = [r["critic_agreement"]   for r in all_summaries]
    latencies       = [r["total_latency_ms"]   for r in all_summaries if r["total_latency_ms"]]
    adj_count       = sum(1 for r in all_summaries if r["adjudication_used"])
    fast_count      = sum(1 for r in all_summaries if r["fast_path"])

    avg_overall    = round(_safe_mean(overall_scores), 2)
    avg_quality    = round(_safe_mean(quality_scores), 2) if quality_scores else 0.0
    avg_confidence = round(_safe_mean(confidences), 4)
    avg_agreement  = round(_safe_mean(agreements), 4)
    avg_latency    = round(_safe_mean(latencies), 1) if latencies else None

    adj_rate       = _rate(adj_count, total)
    fast_rate      = _rate(fast_count, total)

    # ── Group analytics_rows by dimension ────────────────────────────
    by_dim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r)

    critic_stats_list: list[CriticStats] = []
    total_fallbacks = 0

    for dim, dim_rows in by_dim.items():
        n = len(dim_rows)
        scores      = [r["score"]      for r in dim_rows]
        confs       = [r["confidence"] for r in dim_rows]
        pass_count  = sum(1 for r in dim_rows if r["passed"])
        issue_counts  = [r["issue_count"]    for r in dim_rows]
        critical_tots = sum(r["critical_count"] for r in dim_rows)
        major_tots    = sum(r["major_count"]    for r in dim_rows)
        overruled     = [r for r in dim_rows if r["was_overruled"]]
        raised_count  = sum(1 for r in overruled if r["overrule_direction"] == "raised")
        lowered_count = sum(1 for r in overruled if r["overrule_direction"] == "lowered")
        fallback_n    = sum(1 for r in dim_rows if r["used_fallback"])
        latencies_dim = [r["latency_ms"] for r in dim_rows if r["latency_ms"] is not None]
        total_fallbacks += fallback_n

        # Most common issue severity categories
        cat_counter: Counter[str] = Counter()
        for r in dim_rows:
            try:
                cats = json.loads(r["issue_categories"])
                cat_counter.update(cats)
            except (json.JSONDecodeError, TypeError):
                pass
        common_cats = [c for c, _ in cat_counter.most_common(3)]

        # Model used (most common)
        model_counter: Counter[str] = Counter(r["model_used"] for r in dim_rows)
        primary_model = model_counter.most_common(1)[0][0] if model_counter else "unknown"

        critic_stats_list.append(CriticStats(
            dimension=dim,
            model_used=primary_model,
            total_evaluations=n,
            avg_score=round(_safe_mean(scores), 2),
            min_score=min(scores),
            max_score=max(scores),
            pass_rate=_rate(pass_count, n),
            avg_confidence=round(_safe_mean(confs), 4),
            total_issues_raised=sum(issue_counts),
            avg_issues_per_run=round(_safe_mean(issue_counts), 2),
            critical_issues_raised=critical_tots,
            major_issues_raised=major_tots,
            overrule_count=len(overruled),
            overrule_rate=_rate(len(overruled), n),
            raised_overrule_count=raised_count,
            lowered_overrule_count=lowered_count,
            avg_latency_ms=round(_safe_mean(latencies_dim), 1) if latencies_dim else None,
            p95_latency_ms=_safe_p95(latencies_dim),
            fallback_count=fallback_n,
            fallback_rate=_rate(fallback_n, n),
            common_categories=common_cats,
        ))

    # ── Top highlights ────────────────────────────────────────────────
    most_issues_by: str | None = None
    most_overruled: str | None = None
    highest_conf:   str | None = None
    lowest_conf:    str | None = None

    if critic_stats_list:
        most_issues_by = max(
            critic_stats_list, key=lambda s: s.avg_issues_per_run
        ).dimension
        most_overruled = max(
            critic_stats_list, key=lambda s: s.overrule_rate
        ).dimension
        highest_conf = max(
            critic_stats_list, key=lambda s: s.avg_confidence
        ).dimension
        lowest_conf = min(
            critic_stats_list, key=lambda s: s.avg_confidence
        ).dimension

    # Most common issue category across all critics
    global_cat: Counter[str] = Counter()
    for r in rows:
        try:
            global_cat.update(json.loads(r["issue_categories"]))
        except (json.JSONDecodeError, TypeError):
            pass
    most_common_cat = global_cat.most_common(1)[0][0] if global_cat else None

    return AnalyticsReport(
        total_arbitrations=total,
        recommendation_counts=rec_counts,
        approve_rate=approve_rate,
        revise_rate=revise_rate,
        reject_rate=reject_rate,
        avg_overall_score=avg_overall,
        avg_quality_score=avg_quality,
        avg_confidence=avg_confidence,
        avg_critic_agreement=avg_agreement,
        adjudication_rate=adj_rate,
        fast_path_rate=fast_rate,
        avg_latency_ms=avg_latency,
        total_fallbacks_used=total_fallbacks,
        critic_stats=critic_stats_list,
        most_issues_raised_by=most_issues_by,
        most_overruled_critic=most_overruled,
        highest_confidence_critic=highest_conf,
        lowest_confidence_critic=lowest_conf,
        most_common_issue_category=most_common_cat,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
