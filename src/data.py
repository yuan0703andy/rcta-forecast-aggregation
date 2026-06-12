from pathlib import Path
import re

import polars as pl


TOL = 1e-4
CUTOFF_MINUTES = 14 * 60 + 1


def clean_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())).strip("_")

def scan_clean_csv(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, infer_schema_length=2000)
    return lf.rename({name: clean_name(name) for name in lf.collect_schema().names()})

def prepare_vectors(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    qa = scan_clean_csv(raw_dir / "rct-a-questions-answers.csv")
    ps = scan_clean_csv(raw_dir / "rct-a-prediction-sets.csv")

    qmeta = (
        qa.group_by("discover_question_id")
        .agg(
            n_stored_answer_options=pl.len(),
            question_text=pl.col("question_name").first(),
            question_status=pl.col("question_status").first(),
            is_ordinal=pl.col("use_ordinal_scoring").first(),
            resolution_date_or_correctness_known_date=pl.col("question_correctness_known_at").first(),
            question_resolved_at=pl.col("question_resolved_at").first(),
            n_missing_outcomes=pl.col("answer_resolved_probability").is_null().sum(),
            sum_resolved_probability=pl.col("answer_resolved_probability").sum(),
        )
        .with_columns(
            is_binary_single_yes=pl.col("n_stored_answer_options") == 1,
            is_multiclass=pl.col("n_stored_answer_options") > 1,
            has_resolved_outcome=pl.col("n_missing_outcomes") == 0,
        )
        .with_columns(
            correctness_known_utc=pl.coalesce(
                pl.col("resolution_date_or_correctness_known_date").str.to_datetime(strict=False, time_zone="UTC"),
                pl.col("question_resolved_at").str.to_datetime(strict=False, time_zone="UTC"),
            )
        )
        .with_columns(
            correctness_known_date=pl.col("correctness_known_utc")
            .dt.convert_time_zone("America/New_York")
            .dt.date()
            .cast(pl.Utf8)
        )
    )

    ps = ps.join(
        qmeta.select(
            [
                "discover_question_id",
                "n_stored_answer_options",
                "is_binary_single_yes",
                "is_multiclass",
                "is_ordinal",
                "question_text",
                "question_status",
                "correctness_known_date",
                "has_resolved_outcome",
            ]
        ),
        on="discover_question_id",
        how="left",
    )

    clean_ps = (
        ps.filter(pl.col("forecasted_probability").is_not_null())
        .filter(pl.col("answer_resolved_probability").is_not_null())
        .filter(pl.col("forecasted_probability").is_between(0, 1))
        .filter(pl.col("answer_resolved_probability").is_between(0, 1))
        .filter(pl.col("made_after_correctness_known").fill_null(False) != True)
        .filter(pl.col("membership_guid").is_not_null())
        .filter(pl.col("prediction_set_id").is_not_null())
        .filter(pl.col("discover_question_id").is_not_null())
        .filter(pl.col("answer_id").is_not_null())
        .filter(pl.col("prediction_set_created_at").is_not_null())
        .with_columns(
            forecast_timestamp_utc=pl.col("prediction_set_created_at").str.to_datetime(strict=False, time_zone="UTC"),
        )
        .filter(pl.col("forecast_timestamp_utc").is_not_null())
        .with_columns(
            forecast_timestamp_et=pl.col("forecast_timestamp_utc").dt.convert_time_zone("America/New_York"),
        )
        .with_columns(
            forecast_minutes_et=pl.col("forecast_timestamp_et").dt.hour().cast(pl.Int64) * 60
            + pl.col("forecast_timestamp_et").dt.minute().cast(pl.Int64),
        )
        .with_columns(
            shifted_after_cutoff=pl.col("forecast_minutes_et") >= CUTOFF_MINUTES,
        )
        .with_columns(
            scoring_day_et=(
                pl.col("forecast_timestamp_et").dt.date()
                + pl.when(pl.col("shifted_after_cutoff")).then(pl.duration(days=1)).otherwise(pl.duration(days=0))
            ).cast(pl.Utf8)
        )
    )

    submission = clean_ps.group_by("prediction_set_id").agg(
        discover_question_id=pl.col("discover_question_id").first(),
        membership_guid=pl.col("membership_guid").first(),
        forecast_timestamp_et=pl.col("forecast_timestamp_et").max(),
        scoring_day_et=pl.col("scoring_day_et").first(),
        n_answer_rows=pl.len(),
        sum_forecasted_probability=pl.col("forecasted_probability").sum(),
        sum_answer_resolved_probability=pl.col("answer_resolved_probability").sum(),
        is_binary_single_yes=pl.col("is_binary_single_yes").first(),
        is_ordinal=pl.col("is_ordinal").first(),
    )

    valid_submission = submission.filter(
        pl.col("is_binary_single_yes")
        | ((pl.col("sum_forecasted_probability") > 0) & (pl.col("sum_answer_resolved_probability") > 0))
    ).with_columns(
        normalize_forecast=(~pl.col("is_binary_single_yes"))
        & (pl.col("sum_forecasted_probability") > 0)
        & ((pl.col("sum_forecasted_probability") - 1.0).abs() > TOL),
        normalize_outcome=(~pl.col("is_binary_single_yes"))
        & (pl.col("sum_answer_resolved_probability") > 0)
        & ((pl.col("sum_answer_resolved_probability") - 1.0).abs() > TOL),
    )

    valid_ps = (
        clean_ps.join(
            valid_submission.select(
                [
                    "prediction_set_id",
                    "sum_forecasted_probability",
                    "sum_answer_resolved_probability",
                    "normalize_forecast",
                    "normalize_outcome",
                ]
            ),
            on="prediction_set_id",
            how="inner",
        )
        .with_columns(
            forecast_probability=pl.when(pl.col("normalize_forecast"))
            .then(pl.col("forecasted_probability") / pl.col("sum_forecasted_probability"))
            .otherwise(pl.col("forecasted_probability")),
            resolved_probability=pl.when(pl.col("normalize_outcome"))
            .then(pl.col("answer_resolved_probability") / pl.col("sum_answer_resolved_probability"))
            .otherwise(pl.col("answer_resolved_probability")),
        )
    )

    submission_valid = (
        valid_ps.group_by("prediction_set_id")
        .agg(
            discover_question_id=pl.col("discover_question_id").first(),
            membership_guid=pl.col("membership_guid").first(),
            forecast_timestamp_et=pl.col("forecast_timestamp_et").max(),
            scoring_day_et=pl.col("scoring_day_et").first(),
            n_answer_rows=pl.len(),
            is_binary_single_yes=pl.col("is_binary_single_yes").first(),
        )
        .sort(["discover_question_id", "scoring_day_et", "membership_guid", "forecast_timestamp_et", "prediction_set_id"])
    )
    selected_submission = submission_valid.unique(
        subset=["discover_question_id", "scoring_day_et", "membership_guid"],
        keep="last",
        maintain_order=True,
    )
    latest = valid_ps.join(selected_submission.select("prediction_set_id"), on="prediction_set_id", how="inner")

    base_cols = [
        "prediction_set_id",
        "discover_question_id",
        "scoring_day_et",
        "membership_guid",
        "site_name",
        "forecast_timestamp_et",
        "forecast_probability",
        "resolved_probability",
        "is_binary_single_yes",
        "is_ordinal",
        "question_text",
        "correctness_known_date",
        "answer_id",
        "answer_sort_order",
    ]
    binary_yes = (
        latest.filter(pl.col("is_binary_single_yes"))
        .select(base_cols)
        .with_columns(
            answer_option_id=pl.lit("Yes"),
            answer_order=pl.lit(1.0),
            n_answer_options=pl.lit(2).cast(pl.Int64),
        )
    )
    binary_no = binary_yes.with_columns(
        answer_option_id=pl.lit("No_implied"),
        answer_order=pl.lit(2.0),
        forecast_probability=1.0 - pl.col("forecast_probability"),
        resolved_probability=1.0 - pl.col("resolved_probability"),
    )
    multi_rows = (
        latest.filter(~pl.col("is_binary_single_yes"))
        .select(base_cols)
        .with_columns(
            answer_order=pl.coalesce(pl.col("answer_sort_order").cast(pl.Float64), pl.col("answer_id").cast(pl.Float64)),
            n_answer_options=pl.len().over("prediction_set_id").cast(pl.Int64),
        )
        .with_columns(answer_option_id=pl.concat_str([pl.lit("option_"), pl.col("answer_order").cast(pl.Int64).cast(pl.Utf8)]))
    )
    vectors = (
        pl.concat([binary_yes, binary_no, multi_rows], how="diagonal")
        .with_columns(
            subset=pl.when(pl.col("is_binary_single_yes"))
            .then(pl.lit("binary only"))
            .when(pl.col("is_ordinal"))
            .then(pl.lit("ordinal only"))
            .otherwise(pl.lit("multi-option nominal only")),
            scoring_rule_used=pl.when(pl.col("is_binary_single_yes"))
            .then(pl.lit("binary_brier_0_2"))
            .when(pl.col("is_ordinal"))
            .then(pl.lit("jnw_ordered_brier_0_2"))
            .otherwise(pl.lit("multinomial_brier_0_2")),
        )
        .collect()
    )
    qday_meta = (
        vectors.lazy()
        .group_by(["discover_question_id", "scoring_day_et"])
        .agg(
            n_forecasters=pl.col("membership_guid").n_unique(),
            is_binary_single_yes=pl.col("is_binary_single_yes").first(),
            is_ordinal=pl.col("is_ordinal").first(),
            subset=pl.col("subset").first(),
            scoring_rule_used=pl.col("scoring_rule_used").first(),
            correctness_known_date=pl.col("correctness_known_date").first(),
            question_text=pl.col("question_text").first(),
        )
        .collect()
    )
    return vectors, qday_meta
