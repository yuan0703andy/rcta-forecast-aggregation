from pathlib import Path

import numpy as np
import polars as pl

from .scoring import score_forecasts


def write_murphy_decomposition(forecasts: pl.DataFrame, table: Path) -> pl.DataFrame:
    rows = []
    for method in ["raw_mean", "median", "geometric_mean", "trimmed_mean", "geometric_mean_of_odds"]:
        method_forecasts = forecasts.filter(pl.col("method") == method)
        for subset in ["all questions", "binary only", "multi-option nominal only", "ordinal only"]:
            data = method_forecasts if subset == "all questions" else method_forecasts.filter(pl.col("subset") == subset)
            p = data["aggregate_probability"].to_numpy()
            y = data["resolved_probability"].to_numpy()
            if len(y) == 0:
                continue
            y_bar = float(y.mean())
            uncertainty = y_bar * (1.0 - y_bar)
            bins = np.clip(np.digitize(p, np.linspace(0.0, 1.0, 11)) - 1, 0, 9)
            reliability = 0.0
            resolution = 0.0
            for b in range(10):
                mask = bins == b
                n_b = int(mask.sum())
                if n_b == 0:
                    continue
                f_b = float(p[mask].mean())
                o_b = float(y[mask].mean())
                reliability += n_b * (f_b - o_b) ** 2
                resolution += n_b * (o_b - y_bar) ** 2
            reliability /= len(y)
            resolution /= len(y)
            rows.append(
                {
                    "method": method,
                    "subset": subset,
                    "reliability": reliability,
                    "resolution": resolution,
                    "uncertainty": uncertainty,
                    "brier_decomposition": reliability - resolution + uncertainty,
                    "brier_direct": float(np.mean((p - y) ** 2)),
                    "rel_share": reliability / (reliability + resolution) if (reliability + resolution) > 0 else None,
                    "n_option_rows": len(y),
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(table / "diagnostic_murphy_decomposition.csv")
    return out


def write_brier_stratification(scores: pl.DataFrame, vectors: pl.DataFrame, table: Path) -> pl.DataFrame:
    median_scores = scores.filter(pl.col("method") == "median")
    type_rows = (
        median_scores.lazy()
        .group_by("subset")
        .agg(
            axis=pl.lit("question_type"),
            group=pl.col("subset").first(),
            n_question_days=pl.len(),
            mean_brier=pl.col("brier").mean(),
            median_brier=pl.col("brier").median(),
            sd_brier=pl.col("brier").std(),
        )
        .with_columns(group=pl.col("group").cast(pl.Utf8))
        .select(["axis", "group", "n_question_days", "mean_brier", "median_brier", "sd_brier"])
        .collect()
    )

    n_rows = (
        median_scores.lazy()
        .with_columns(group=pl.col("n_forecasters").qcut(4, labels=["Q1_lowest_n", "Q2", "Q3", "Q4_highest_n"]))
        .group_by("group")
        .agg(
            axis=pl.lit("n_forecasters"),
            n_question_days=pl.len(),
            mean_brier=pl.col("brier").mean(),
            median_brier=pl.col("brier").median(),
            sd_brier=pl.col("brier").std(),
        )
        .with_columns(group=pl.col("group").cast(pl.Utf8))
        .select(["axis", "group", "n_question_days", "mean_brier", "median_brier", "sd_brier"])
        .collect()
    )

    dispersion = (
        vectors.lazy()
        .group_by(["discover_question_id", "scoring_day_et", "answer_option_id"])
        .agg(option_sd=pl.col("forecast_probability").std())
        .group_by(["discover_question_id", "scoring_day_et"])
        .agg(dispersion=pl.col("option_sd").mean())
        .collect()
    )
    dispersion_rows = (
        median_scores.join(dispersion, on=["discover_question_id", "scoring_day_et"], how="left")
        .lazy()
        .with_columns(group=pl.col("dispersion").qcut(3, labels=["low", "medium", "high"]))
        .group_by("group")
        .agg(
            axis=pl.lit("forecast_dispersion"),
            n_question_days=pl.len(),
            mean_brier=pl.col("brier").mean(),
            median_brier=pl.col("brier").median(),
            sd_brier=pl.col("brier").std(),
        )
        .with_columns(group=pl.col("group").cast(pl.Utf8))
        .select(["axis", "group", "n_question_days", "mean_brier", "median_brier", "sd_brier"])
        .collect()
    )

    staleness = (
        vectors.lazy()
        .with_columns(
            scoring_day=pl.col("scoring_day_et").str.to_date(),
            forecast_day=pl.col("forecast_timestamp_et").dt.date(),
        )
        .with_columns(age_days=(pl.col("scoring_day") - pl.col("forecast_day")).dt.total_days())
        .group_by(["discover_question_id", "scoring_day_et"])
        .agg(median_age_days=pl.col("age_days").median())
        .collect()
    )
    staleness_rows = (
        median_scores.join(staleness, on=["discover_question_id", "scoring_day_et"], how="left")
        .lazy()
        .with_columns(
            group=pl.when(pl.col("median_age_days") <= 0)
            .then(pl.lit("same_day"))
            .when(pl.col("median_age_days") <= 1)
            .then(pl.lit("one_day"))
            .otherwise(pl.lit("older"))
        )
        .group_by("group")
        .agg(
            axis=pl.lit("forecast_age"),
            n_question_days=pl.len(),
            mean_brier=pl.col("brier").mean(),
            median_brier=pl.col("brier").median(),
            sd_brier=pl.col("brier").std(),
        )
        .select(["axis", "group", "n_question_days", "mean_brier", "median_brier", "sd_brier"])
        .collect()
    )

    out = pl.concat([type_rows, n_rows, dispersion_rows, staleness_rows], how="vertical")
    out.write_csv(table / "diagnostic_brier_stratification.csv")
    return out


def write_gmo_tail_autopsy(scores: pl.DataFrame, table: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    paired = (
        scores.filter(pl.col("method").is_in(["median", "geometric_mean_of_odds"]))
        .select(["method", "discover_question_id", "scoring_day_et", "brier", "subset"])
        .pivot(index=["discover_question_id", "scoring_day_et", "subset"], on="method", values="brier")
        .with_columns(delta_gmo_minus_median=pl.col("geometric_mean_of_odds") - pl.col("median"))
    )
    q = (
        paired.lazy()
        .group_by(["discover_question_id", "subset"])
        .agg(
            mean_delta=pl.col("delta_gmo_minus_median").mean(),
            question_day_positive_excess_loss=pl.col("delta_gmo_minus_median").clip(0, None).sum(),
            n_question_days=pl.len(),
        )
        .with_columns(question_balanced_positive_excess_loss=pl.col("mean_delta").clip(0, None))
        .sort("question_balanced_positive_excess_loss", descending=True)
        .collect()
    )

    rows = []
    for subset in ["all questions", "binary only", "multi-option nominal only", "ordinal only"]:
        data = paired if subset == "all questions" else paired.filter(pl.col("subset") == subset)
        qdata = (
            paired.lazy()
            .group_by("discover_question_id")
            .agg(
                subset=pl.lit("all questions"),
                mean_delta=pl.col("delta_gmo_minus_median").mean(),
                question_day_positive_excess_loss=pl.col("delta_gmo_minus_median").clip(0, None).sum(),
                n_question_days=pl.len(),
            )
            .with_columns(question_balanced_positive_excess_loss=pl.col("mean_delta").clip(0, None))
            .collect()
            if subset == "all questions"
            else q.filter(pl.col("subset") == subset)
        )
        question_day_excess = qdata.sort("question_day_positive_excess_loss", descending=True)[
            "question_day_positive_excess_loss"
        ].to_numpy()
        question_balanced_excess = qdata.sort("question_balanced_positive_excess_loss", descending=True)[
            "question_balanced_positive_excess_loss"
        ].to_numpy()
        total_question_day_excess = float(question_day_excess.sum())
        total_question_balanced_excess = float(question_balanced_excess.sum())
        rows.append(
            {
                "subset": subset,
                "n_question_days": data.height,
                "share_question_days_gmo_beats_median": float((data["delta_gmo_minus_median"] < 0).mean()) if data.height else None,
                "median_delta_gmo_minus_median": float(data["delta_gmo_minus_median"].median()) if data.height else None,
                "mean_delta_gmo_minus_median": float(data["delta_gmo_minus_median"].mean()) if data.height else None,
                "p90_delta_gmo_minus_median": float(data["delta_gmo_minus_median"].quantile(0.90)) if data.height else None,
                "p95_delta_gmo_minus_median": float(data["delta_gmo_minus_median"].quantile(0.95)) if data.height else None,
                "share_qday_positive_excess_loss_worst_10_questions": float(question_day_excess[:10].sum() / total_question_day_excess)
                if total_question_day_excess > 0
                else None,
                "share_qday_positive_excess_loss_worst_20_questions": float(question_day_excess[:20].sum() / total_question_day_excess)
                if total_question_day_excess > 0
                else None,
                "share_qbalanced_positive_excess_loss_worst_10_questions": float(
                    question_balanced_excess[:10].sum() / total_question_balanced_excess
                )
                if total_question_balanced_excess > 0
                else None,
                "share_qbalanced_positive_excess_loss_worst_20_questions": float(
                    question_balanced_excess[:20].sum() / total_question_balanced_excess
                )
                if total_question_balanced_excess > 0
                else None,
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(table / "diagnostic_gmo_tail_autopsy_summary.csv")
    q.write_csv(table / "diagnostic_gmo_vs_median_question_delta.csv")
    return summary, q


def write_relative_to_individual(vectors: pl.DataFrame, baseline_scores: pl.DataFrame, table: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    qday_counts = vectors.group_by(["discover_question_id", "scoring_day_et"]).agg(
        n_forecasters=pl.col("membership_guid").n_unique()
    )
    individual_forecasts = vectors.join(qday_counts, on=["discover_question_id", "scoring_day_et"], how="left").with_columns(
        method=pl.concat_str([pl.lit("forecaster_"), pl.col("membership_guid")]),
        aggregate_probability=pl.col("forecast_probability"),
    )
    individual_scores = score_forecasts(individual_forecasts)
    agg_scores = baseline_scores.filter(pl.col("method").is_in(["median", "geometric_mean_of_odds"]))
    ranks = (
        agg_scores.join(
            individual_scores.select(
                [
                    "discover_question_id",
                    "scoring_day_et",
                    pl.col("method").alias("individual_id"),
                    pl.col("brier").alias("individual_brier"),
                ]
            ),
            on=["discover_question_id", "scoring_day_et"],
            how="inner",
        )
        .with_columns(aggregate_beats_individual=pl.col("individual_brier") > pl.col("brier"))
        .group_by(["method", "discover_question_id", "scoring_day_et", "subset"])
        .agg(
            n_individuals=pl.len(),
            aggregate_brier=pl.col("brier").first(),
            median_individual_brier=pl.col("individual_brier").median(),
            mean_individual_brier=pl.col("individual_brier").mean(),
            aggregate_percentile_rank=pl.col("aggregate_beats_individual").mean(),
        )
    )
    summary = (
        ranks.lazy()
        .group_by(["method", "subset"])
        .agg(
            n_question_days=pl.len(),
            mean_percentile_rank=pl.col("aggregate_percentile_rank").mean(),
            median_percentile_rank=pl.col("aggregate_percentile_rank").median(),
            mean_aggregate_brier=pl.col("aggregate_brier").mean(),
            mean_median_individual_brier=pl.col("median_individual_brier").mean(),
        )
        .sort(["subset", "method"])
        .collect()
    )
    ranks.write_csv(table / "diagnostic_relative_to_individual_question_day.csv")
    summary.write_csv(table / "diagnostic_relative_to_individual_summary.csv")
    return summary, ranks


def write_rolling_window_validation(baseline_scores: pl.DataFrame, final_scores: pl.DataFrame, table: Path) -> pl.DataFrame:
    daily = (
        baseline_scores.filter(pl.col("method") == "median")
        .select(["discover_question_id", "scoring_day_et", pl.col("brier").alias("median_brier")])
        .join(
            final_scores.select(["discover_question_id", "scoring_day_et", pl.col("brier").alias("final_brier")]),
            on=["discover_question_id", "scoring_day_et"],
            how="inner",
        )
        .with_columns(improvement=pl.col("median_brier") - pl.col("final_brier"))
    )
    day_series = (
        daily.lazy()
        .group_by("scoring_day_et")
        .agg(improvement=pl.col("improvement").mean())
        .sort("scoring_day_et")
        .collect()
    )
    rows = []
    x = day_series["improvement"].to_numpy()
    for window in [7, 14, 30, 45]:
        if len(x) < window:
            continue
        vals = np.array([x[i : i + window].mean() for i in range(len(x) - window + 1)])
        rows.append(
            {
                "window": f"{window}-day daily-weighted",
                "window_days": window,
                "mean_improvement": float(vals.mean()),
                "share_positive_windows": float((vals > 0).mean()),
                "n_windows": int(len(vals)),
            }
        )

    q_daily = (
        daily.lazy()
        .group_by(["scoring_day_et", "discover_question_id"])
        .agg(improvement=pl.col("improvement").mean())
        .group_by("scoring_day_et")
        .agg(improvement=pl.col("improvement").mean())
        .sort("scoring_day_et")
        .collect()
    )
    xq = q_daily["improvement"].to_numpy()
    for window in [14, 30, 45]:
        if len(xq) < window:
            continue
        vals = np.array([xq[i : i + window].mean() for i in range(len(xq) - window + 1)])
        rows.append(
            {
                "window": f"{window}-day question-balanced",
                "window_days": window,
                "mean_improvement": float(vals.mean()),
                "share_positive_windows": float((vals > 0).mean()),
                "n_windows": int(len(vals)),
            }
        )
    out = pl.DataFrame(rows)
    out.write_csv(table / "diagnostic_rolling_window_validation.csv")
    return out


def write_tested_conditional_improvements_summary(table: Path) -> pl.DataFrame:
    out = pl.DataFrame(
        {
            "candidate_direction": [
                "skill_weighting",
                "type_conditioned_selection",
                "thin_crowd_override",
                "dispersion_shrinkage",
                "activity_update_history",
            ],
            "statistical_object": [
                "prior resolved questions per active forecaster",
                "walk-forward best method by question type",
                "switch method when n_qt is small",
                "shrink high-dispersion forecasts toward uniform",
                "accuracy change after forecast updates",
            ],
            "memo_result": [
                "first_half_median_prior_resolved=3.5; second_half=16",
                "oracle improves slightly; walk-forward underperforms median",
                "best threshold mostly collapses to median",
                "Brier worsens monotonically as shrinkage increases",
                "large updates improve 79% of the time",
            ],
            "decision": [
                "not stable enough for season-wide use",
                "hindsight structure is not deployable signal",
                "not a meaningful improvement",
                "dispersion is a marker, not an intervention",
                "future signal, not final rule",
            ],
        }
    )
    out.write_csv(table / "diagnostic_tested_conditional_improvements_summary.csv")
    return out


def write_supporting_diagnostics(
    vectors: pl.DataFrame,
    baseline_forecasts: pl.DataFrame,
    baseline_scores: pl.DataFrame,
    final_scores: pl.DataFrame,
    table: Path,
) -> None:
    write_murphy_decomposition(baseline_forecasts, table)
    write_brier_stratification(baseline_scores, vectors, table)
    write_tested_conditional_improvements_summary(table)
    write_gmo_tail_autopsy(baseline_scores, table)
    write_relative_to_individual(vectors, baseline_scores, table)
    write_rolling_window_validation(baseline_scores, final_scores, table)
