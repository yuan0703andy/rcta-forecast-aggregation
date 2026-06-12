from pathlib import Path

import numpy as np
import polars as pl

from .aggregation import aggregate_site_balanced_bounded_log_odds


def score_summary(scores: pl.DataFrame, weighting: str) -> pl.DataFrame:
    if weighting == "question_day_weighted":
        return (
            scores.lazy()
            .group_by("method")
            .agg(
                mean_brier=pl.col("brier").mean(),
                standard_error=(pl.col("brier").std(ddof=1) / pl.len().sqrt()),
                n_question_days=pl.len(),
                n_questions=pl.col("discover_question_id").n_unique(),
                mean_n_forecasters=pl.col("n_forecasters").mean(),
                median_n_forecasters=pl.col("n_forecasters").median(),
            )
            .with_columns(scoring_scale=pl.lit("0-2"), weighting=pl.lit(weighting))
            .sort("mean_brier")
            .collect()
        )
    q_means = (
        scores.lazy()
        .group_by(["method", "discover_question_id"])
        .agg(
            question_mean_brier=pl.col("brier").mean(),
            mean_n_forecasters_q=pl.col("n_forecasters").mean(),
        )
    )
    qday_counts = (
        scores.lazy()
        .group_by("method")
        .agg(
            n_question_days=pl.len(),
            n_questions=pl.col("discover_question_id").n_unique(),
            mean_n_forecasters=pl.col("n_forecasters").mean(),
            median_n_forecasters=pl.col("n_forecasters").median(),
        )
    )
    return (
        q_means.group_by("method")
        .agg(
            mean_brier=pl.col("question_mean_brier").mean(),
            standard_error=(pl.col("question_mean_brier").std(ddof=1) / pl.len().sqrt()),
        )
        .join(qday_counts, on="method", how="left")
        .with_columns(scoring_scale=pl.lit("0-2"), weighting=pl.lit(weighting))
        .sort("mean_brier")
        .collect()
    )

def score_forecasts(forecasts: pl.DataFrame) -> pl.DataFrame:
    keys = ["method", "discover_question_id", "scoring_day_et"]
    sorted_forecasts = forecasts.sort(keys + ["answer_order", "answer_option_id"])
    nominal = (
        sorted_forecasts.lazy()
        .filter(~pl.col("is_ordinal") | (pl.col("n_answer_options") <= 2))
        .with_columns(sq_err=(pl.col("aggregate_probability") - pl.col("resolved_probability")) ** 2)
        .group_by(keys)
        .agg(
            brier=pl.col("sq_err").sum(),
            n_forecasters=pl.col("n_forecasters").first(),
            n_answer_options=pl.col("n_answer_options").first(),
            is_binary_single_yes=pl.col("is_binary_single_yes").first(),
            is_ordinal=pl.col("is_ordinal").first(),
            subset=pl.col("subset").first(),
            correctness_known_date=pl.col("correctness_known_date").first(),
            scoring_rule_used=pl.col("scoring_rule_used").first(),
        )
    )
    ordinal_terms = (
        sorted_forecasts.lazy()
        .filter(pl.col("is_ordinal") & (pl.col("n_answer_options") > 2))
        .with_columns(
            cumulative_p=pl.col("aggregate_probability").cum_sum().over(keys),
            cumulative_y=pl.col("resolved_probability").cum_sum().over(keys),
            option_position=pl.int_range(1, pl.len() + 1).over(keys),
        )
        .filter(pl.col("option_position") < pl.col("n_answer_options"))
        .with_columns(jnw_term=(pl.col("cumulative_p") - pl.col("cumulative_y")) ** 2)
        .group_by(keys)
        .agg(
            brier=(2.0 / (pl.col("n_answer_options").first() - 1) * pl.col("jnw_term").sum()),
            n_forecasters=pl.col("n_forecasters").first(),
            n_answer_options=pl.col("n_answer_options").first(),
            is_binary_single_yes=pl.col("is_binary_single_yes").first(),
            is_ordinal=pl.col("is_ordinal").first(),
            subset=pl.col("subset").first(),
            correctness_known_date=pl.col("correctness_known_date").first(),
            scoring_rule_used=pl.col("scoring_rule_used").first(),
        )
    )
    return pl.concat([nominal, ordinal_terms], how="vertical").collect()

def write_cap_sensitivity(vectors: pl.DataFrame, qday_meta: pl.DataFrame, table: Path) -> pl.DataFrame:
    rows = []
    for cap in [0.005, 0.01, 0.02, 0.05]:
        forecasts = aggregate_site_balanced_bounded_log_odds(vectors, qday_meta, cap)
        scores = score_forecasts(forecasts)
        rows.append(
            {
                "cap": cap,
                "max_individual_odds": (1 - cap) / cap,
                "question_balanced_brier": score_summary(scores, "question_balanced")["mean_brier"][0],
                "question_day_weighted_brier": score_summary(scores, "question_day_weighted")["mean_brier"][0],
            }
        )
    out = pl.DataFrame(rows)
    out.write_csv(table / "site_balanced_cap_sensitivity.csv")
    return out

def bootstrap_vs_median(baseline_scores: pl.DataFrame, final_scores: pl.DataFrame, table: Path) -> pl.DataFrame:
    paired = (
        baseline_scores.filter(pl.col("method") == "median")
        .select(["discover_question_id", "scoring_day_et", pl.col("brier").alias("median_brier")])
        .join(
            final_scores.select(["discover_question_id", "scoring_day_et", pl.col("brier").alias("final_brier")]),
            on=["discover_question_id", "scoring_day_et"],
            how="inner",
        )
        .with_columns(improvement=pl.col("median_brier") - pl.col("final_brier"))
        .group_by("discover_question_id")
        .agg(improvement=pl.col("improvement").mean())
        .sort("discover_question_id")
    )
    q_improvements = paired["improvement"].to_numpy()
    rng = np.random.default_rng(20260612)
    boot = np.array([rng.choice(q_improvements, size=len(q_improvements), replace=True).mean() for _ in range(2000)])
    out = pl.DataFrame(
        {
            "method": ["site_balanced_capped_gmo_0.02"],
            "comparison_baseline": ["median"],
            "observed_q_balanced_improvement_vs_median": [float(q_improvements.mean())],
            "ci_low_95": [float(np.quantile(boot, 0.025))],
            "ci_high_95": [float(np.quantile(boot, 0.975))],
            "n_questions": [int(len(q_improvements))],
            "n_bootstrap": [2000],
            "seed": [20260612],
        }
    )
    out.write_csv(table / "final_method_bootstrap_vs_median.csv")
    return out
