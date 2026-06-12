from pathlib import Path

import polars as pl

from .aggregation import (
    aggregate_baseline_methods,
    aggregate_capped_gmo,
    aggregate_site_balanced_bounded_log_odds,
    aggregate_site_balanced_median,
)
from .data import prepare_vectors
from .scoring import bootstrap_vs_median, score_forecasts, score_summary, write_cap_sensitivity


FINAL_CAP = 0.02


def resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "requirements.txt").exists():
        return here
    if (here.parent / "requirements.txt").exists():
        return here.parent
    return here


def resolve_raw_data_dir(workdir: str, project_root: Path) -> Path:
    configured = Path(workdir)
    if configured.exists():
        return configured
    local_raw = project_root / "data" / "raw"
    if local_raw.exists():
        return local_raw
    return project_root


def run_analysis(workdir: str) -> None:
    root = resolve_project_root()
    raw = resolve_raw_data_dir(workdir, root)
    table = root / "output" / "table"
    table.mkdir(parents=True, exist_ok=True)

    print(f"Using raw data directory: {raw}")
    print(f"Writing tables to: {table}")
    vectors, qday_meta = prepare_vectors(raw)
    vectors.write_parquet(table / "latest_vectors.parquet")

    baseline_forecasts = aggregate_baseline_methods(vectors, qday_meta)
    baseline_scores = score_forecasts(baseline_forecasts)
    baseline_balanced = score_summary(baseline_scores, "question_balanced")
    baseline_weighted = score_summary(baseline_scores, "question_day_weighted")
    baseline_balanced.write_csv(table / "baseline_scores_question_balanced.csv")
    baseline_weighted.write_csv(table / "baseline_scores_question_day_weighted.csv")

    final_forecasts = aggregate_site_balanced_bounded_log_odds(vectors, qday_meta, FINAL_CAP)
    final_scores = score_forecasts(final_forecasts)
    final_balanced = score_summary(final_scores, "question_balanced")
    final_weighted = score_summary(final_scores, "question_day_weighted")
    final_balanced.write_csv(table / "improved_scores_question_balanced.csv")
    final_weighted.write_csv(table / "improved_scores_question_day_weighted.csv")

    capped = aggregate_capped_gmo(vectors, qday_meta, FINAL_CAP)
    site_median = aggregate_site_balanced_median(vectors, qday_meta)
    capped_scores = score_forecasts(capped)
    site_median_scores = score_forecasts(site_median)
    ablation = pl.DataFrame(
        {
            "method": [
                "median_baseline",
                "original_gmo",
                "capped_gmo_0.02",
                "site_balanced_median",
                "site_balanced_capped_gmo_0.02",
            ],
            "question_balanced_brier": [
                baseline_balanced.filter(pl.col("method") == "median")["mean_brier"][0],
                baseline_balanced.filter(pl.col("method") == "geometric_mean_of_odds")["mean_brier"][0],
                score_summary(capped_scores, "question_balanced")["mean_brier"][0],
                score_summary(site_median_scores, "question_balanced")["mean_brier"][0],
                final_balanced["mean_brier"][0],
            ],
            "question_day_weighted_brier": [
                baseline_weighted.filter(pl.col("method") == "median")["mean_brier"][0],
                baseline_weighted.filter(pl.col("method") == "geometric_mean_of_odds")["mean_brier"][0],
                score_summary(capped_scores, "question_day_weighted")["mean_brier"][0],
                score_summary(site_median_scores, "question_day_weighted")["mean_brier"][0],
                final_weighted["mean_brier"][0],
            ],
        }
    )
    ablation.write_csv(table / "final_method_ablation_full.csv")
    write_cap_sensitivity(vectors, qday_meta, table)
    bootstrap_vs_median(baseline_scores, final_scores, table)
    print(f"Done. Polars outputs written to: {table}")
