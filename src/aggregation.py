import polars as pl


EPS = 1e-6


def sigmoid_expr(expr: pl.Expr) -> pl.Expr:
    z = expr.clip(-700, 700)
    return 1.0 / (1.0 + (-z).exp())

def clipped_probability_expr(col: str, eps: float = EPS) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).clip(eps, 1.0 - eps)

def add_trim_columns(lf: pl.LazyFrame, group_cols: list[str], value_col: str = "forecast_probability") -> pl.LazyFrame:
    n = pl.len().over(group_cols)
    h = (n.cast(pl.Float64) * 0.10).floor().cast(pl.Int64)
    rank = pl.col(value_col).rank(method="ordinal").over(group_cols)
    keep = (h == 0) | ((n - 2 * h) < 1) | ((rank > h) & (rank <= (n - h)))
    return lf.with_columns(
        trim_n=n.alias("_trim_n"),
        trim_h=h.alias("_trim_h"),
        trim_rank=rank.alias("_trim_rank"),
        trim_value=pl.when(keep).then(pl.col(value_col)).otherwise(None).alias("_trim_value"),
    )

def aggregate_baseline_methods(vectors: pl.DataFrame, qday_meta: pl.DataFrame) -> pl.DataFrame:
    meta_cols = [
        "discover_question_id",
        "scoring_day_et",
        "n_forecasters",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]

    binary_yes = vectors.lazy().filter(pl.col("is_binary_single_yes") & (pl.col("answer_option_id") == "Yes"))
    binary_group = ["discover_question_id", "scoring_day_et"]
    binary_trimmed = add_trim_columns(binary_yes, binary_group)
    binary_scores = (
        binary_trimmed.group_by(binary_group)
        .agg(
            raw_mean=pl.col("forecast_probability").mean(),
            median=pl.col("forecast_probability").median(),
            geometric_mean=clipped_probability_expr("forecast_probability").log().mean().exp(),
            trimmed_mean=pl.col("trim_value").mean(),
            geometric_mean_of_odds=sigmoid_expr((clipped_probability_expr("forecast_probability") / (1 - clipped_probability_expr("forecast_probability"))).log().mean()),
            resolved_probability=pl.col("resolved_probability").first(),
        )
        .unpivot(
            index=["discover_question_id", "scoring_day_et", "resolved_probability"],
            on=["raw_mean", "median", "geometric_mean", "trimmed_mean", "geometric_mean_of_odds"],
            variable_name="method",
            value_name="p_yes",
        )
        .join(qday_meta.lazy().select(meta_cols), on=binary_group, how="left")
    )
    binary_yes_out = binary_scores.with_columns(
        answer_option_id=pl.lit("Yes"),
        answer_order=pl.lit(1.0),
        aggregate_probability=pl.col("p_yes"),
        n_answer_options=pl.lit(2),
        resolved_probability=pl.col("resolved_probability"),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
    )
    binary_no_out = binary_scores.with_columns(
        answer_option_id=pl.lit("No_implied"),
        answer_order=pl.lit(2.0),
        aggregate_probability=1.0 - pl.col("p_yes"),
        n_answer_options=pl.lit(2),
        resolved_probability=1.0 - pl.col("resolved_probability"),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
    )

    multi = vectors.lazy().filter(~pl.col("is_binary_single_yes"))
    multi_group = ["discover_question_id", "scoring_day_et", "answer_option_id"]
    multi_trimmed = add_trim_columns(multi, multi_group)
    multi_scores = (
        multi_trimmed.group_by(multi_group)
        .agg(
            raw_mean=pl.col("forecast_probability").mean(),
            median=pl.col("forecast_probability").median(),
            geometric_mean=clipped_probability_expr("forecast_probability").log().mean().exp(),
            trimmed_mean=pl.col("trim_value").mean(),
            geometric_mean_of_odds=sigmoid_expr((clipped_probability_expr("forecast_probability") / (1 - clipped_probability_expr("forecast_probability"))).log().mean()),
            answer_order=pl.col("answer_order").first(),
            resolved_probability=pl.col("resolved_probability").first(),
        )
        .unpivot(
            index=["discover_question_id", "scoring_day_et", "answer_option_id", "answer_order", "resolved_probability"],
            on=["raw_mean", "median", "geometric_mean", "trimmed_mean", "geometric_mean_of_odds"],
            variable_name="method",
            value_name="score",
        )
        .with_columns(
            aggregate_probability=pl.col("score")
            / pl.col("score").sum().over(["discover_question_id", "scoring_day_et", "method"])
        )
        .with_columns(n_answer_options=pl.col("answer_option_id").n_unique().over(["discover_question_id", "scoring_day_et"]))
        .join(qday_meta.lazy().select(meta_cols), on=["discover_question_id", "scoring_day_et"], how="left")
    )

    columns = [
        "discover_question_id",
        "scoring_day_et",
        "method",
        "answer_option_id",
        "answer_order",
        "aggregate_probability",
        "resolved_probability",
        "n_forecasters",
        "n_answer_options",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    output_cast = [
        pl.col("n_forecasters").cast(pl.Int64),
        pl.col("n_answer_options").cast(pl.Int64),
        pl.col("answer_order").cast(pl.Float64),
    ]
    return pl.concat(
        [
            binary_yes_out.with_columns(output_cast).select(columns),
            binary_no_out.with_columns(output_cast).select(columns),
            multi_scores.with_columns(output_cast).select(columns),
        ],
        how="vertical",
    ).collect()

def aggregate_site_balanced_bounded_log_odds(vectors: pl.DataFrame, qday_meta: pl.DataFrame, cap: float) -> pl.DataFrame:
    method_name = f"site_balanced_capped_gmo_{cap:g}"
    meta_cols = [
        "discover_question_id",
        "scoring_day_et",
        "n_forecasters",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    bounded_logit = (pl.col("forecast_probability").cast(pl.Float64).clip(cap, 1 - cap) / (1 - pl.col("forecast_probability").cast(pl.Float64).clip(cap, 1 - cap))).log()
    df = vectors.lazy().with_columns(bounded_logit=bounded_logit)

    binary_site = (
        df.filter(pl.col("is_binary_single_yes") & (pl.col("answer_option_id") == "Yes"))
        .group_by(["discover_question_id", "scoring_day_et", "site_name"])
        .agg(site_logit=pl.col("bounded_logit").mean(), resolved_probability=pl.col("resolved_probability").first())
    )
    binary_qday = (
        binary_site.group_by(["discover_question_id", "scoring_day_et"])
        .agg(mean_logit=pl.col("site_logit").mean(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(p_yes=sigmoid_expr(pl.col("mean_logit")))
        .join(qday_meta.lazy().select(meta_cols), on=["discover_question_id", "scoring_day_et"], how="left")
    )
    binary_yes = binary_qday.with_columns(
        method=pl.lit(method_name),
        answer_option_id=pl.lit("Yes"),
        answer_order=pl.lit(1.0),
        aggregate_probability=pl.col("p_yes"),
        n_answer_options=pl.lit(2),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
    )
    binary_no = binary_qday.with_columns(
        method=pl.lit(method_name),
        answer_option_id=pl.lit("No_implied"),
        answer_order=pl.lit(2.0),
        aggregate_probability=1.0 - pl.col("p_yes"),
        resolved_probability=1.0 - pl.col("resolved_probability"),
        n_answer_options=pl.lit(2),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
    )

    multi_option = (
        df.filter(~pl.col("is_binary_single_yes"))
        .group_by(["discover_question_id", "scoring_day_et", "site_name", "answer_option_id"])
        .agg(site_logit=pl.col("bounded_logit").mean(), answer_order=pl.col("answer_order").first(), resolved_probability=pl.col("resolved_probability").first())
        .group_by(["discover_question_id", "scoring_day_et", "answer_option_id"])
        .agg(site_logit=pl.col("site_logit").mean(), answer_order=pl.col("answer_order").first(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(score=sigmoid_expr(pl.col("site_logit")))
        .with_columns(aggregate_probability=pl.col("score") / pl.col("score").sum().over(["discover_question_id", "scoring_day_et"]))
        .with_columns(n_answer_options=pl.col("answer_option_id").n_unique().over(["discover_question_id", "scoring_day_et"]))
        .join(qday_meta.lazy().select(meta_cols), on=["discover_question_id", "scoring_day_et"], how="left")
        .with_columns(method=pl.lit(method_name))
    )

    columns = [
        "discover_question_id",
        "scoring_day_et",
        "method",
        "answer_option_id",
        "answer_order",
        "aggregate_probability",
        "resolved_probability",
        "n_forecasters",
        "n_answer_options",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    output_cast = [
        pl.col("n_forecasters").cast(pl.Int64),
        pl.col("n_answer_options").cast(pl.Int64),
        pl.col("answer_order").cast(pl.Float64),
    ]
    return pl.concat(
        [
            binary_yes.with_columns(output_cast).select(columns),
            binary_no.with_columns(output_cast).select(columns),
            multi_option.with_columns(output_cast).select(columns),
        ],
        how="vertical",
    ).collect()

def aggregate_capped_gmo(vectors: pl.DataFrame, qday_meta: pl.DataFrame, cap: float) -> pl.DataFrame:
    bounded = (pl.col("forecast_probability").cast(pl.Float64).clip(cap, 1 - cap) / (1 - pl.col("forecast_probability").cast(pl.Float64).clip(cap, 1 - cap))).log()
    temp = vectors.with_columns(bounded_logit=bounded)
    yes = temp.lazy().filter(pl.col("is_binary_single_yes") & (pl.col("answer_option_id") == "Yes"))
    binary = (
        yes.group_by(["discover_question_id", "scoring_day_et"])
        .agg(mean_logit=pl.col("bounded_logit").mean(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(p_yes=sigmoid_expr(pl.col("mean_logit")))
        .join(qday_meta.lazy(), on=["discover_question_id", "scoring_day_et"], how="left")
    )
    binary_yes = binary.with_columns(
        method=pl.lit(f"capped_gmo_{cap:g}"),
        answer_option_id=pl.lit("Yes"),
        answer_order=pl.lit(1.0),
        aggregate_probability=pl.col("p_yes"),
        n_answer_options=pl.lit(2),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
    )
    binary_no = binary.with_columns(
        method=pl.lit(f"capped_gmo_{cap:g}"),
        answer_option_id=pl.lit("No_implied"),
        answer_order=pl.lit(2.0),
        aggregate_probability=1.0 - pl.col("p_yes"),
        resolved_probability=1.0 - pl.col("resolved_probability"),
        n_answer_options=pl.lit(2),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
    )
    multi = (
        temp.lazy()
        .filter(~pl.col("is_binary_single_yes"))
        .group_by(["discover_question_id", "scoring_day_et", "answer_option_id"])
        .agg(score=sigmoid_expr(pl.col("bounded_logit").mean()), answer_order=pl.col("answer_order").first(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(aggregate_probability=pl.col("score") / pl.col("score").sum().over(["discover_question_id", "scoring_day_et"]))
        .with_columns(n_answer_options=pl.col("answer_option_id").n_unique().over(["discover_question_id", "scoring_day_et"]))
        .join(qday_meta.lazy(), on=["discover_question_id", "scoring_day_et"], how="left")
        .with_columns(method=pl.lit(f"capped_gmo_{cap:g}"))
    )
    columns = [
        "discover_question_id",
        "scoring_day_et",
        "method",
        "answer_option_id",
        "answer_order",
        "aggregate_probability",
        "resolved_probability",
        "n_forecasters",
        "n_answer_options",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    output_cast = [
        pl.col("n_forecasters").cast(pl.Int64),
        pl.col("n_answer_options").cast(pl.Int64),
        pl.col("answer_order").cast(pl.Float64),
    ]
    return pl.concat(
        [
            binary_yes.with_columns(output_cast).select(columns),
            binary_no.with_columns(output_cast).select(columns),
            multi.with_columns(output_cast).select(columns),
        ],
        how="vertical",
    ).collect()

def aggregate_site_balanced_median(vectors: pl.DataFrame, qday_meta: pl.DataFrame) -> pl.DataFrame:
    meta_cols = [
        "discover_question_id",
        "scoring_day_et",
        "n_forecasters",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    yes = vectors.lazy().filter(pl.col("is_binary_single_yes") & (pl.col("answer_option_id") == "Yes"))
    binary = (
        yes.group_by(["discover_question_id", "scoring_day_et", "site_name"])
        .agg(site_probability=pl.col("forecast_probability").median(), resolved_probability=pl.col("resolved_probability").first())
        .group_by(["discover_question_id", "scoring_day_et"])
        .agg(p_yes=pl.col("site_probability").mean(), resolved_probability=pl.col("resolved_probability").first())
        .join(qday_meta.lazy().select(meta_cols), on=["discover_question_id", "scoring_day_et"], how="left")
    )
    binary_yes = binary.with_columns(
        method=pl.lit("site_balanced_median"),
        answer_option_id=pl.lit("Yes"),
        answer_order=pl.lit(1.0),
        aggregate_probability=pl.col("p_yes"),
        n_answer_options=pl.lit(2),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
    )
    binary_no = binary.with_columns(
        method=pl.lit("site_balanced_median"),
        answer_option_id=pl.lit("No_implied"),
        answer_order=pl.lit(2.0),
        aggregate_probability=1.0 - pl.col("p_yes"),
        resolved_probability=1.0 - pl.col("resolved_probability"),
        n_answer_options=pl.lit(2),
        is_binary_single_yes=pl.lit(True),
        is_ordinal=pl.lit(False),
        subset=pl.lit("binary only"),
        scoring_rule_used=pl.lit("binary_brier_0_2"),
    )
    multi_site = (
        vectors.lazy()
        .filter(~pl.col("is_binary_single_yes"))
        .group_by(["discover_question_id", "scoring_day_et", "site_name", "answer_option_id"])
        .agg(site_probability=pl.col("forecast_probability").median(), answer_order=pl.col("answer_order").first(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(site_probability=pl.col("site_probability") / pl.col("site_probability").sum().over(["discover_question_id", "scoring_day_et", "site_name"]))
        .with_columns(site_probability=pl.col("site_probability").fill_nan(None))
        .group_by(["discover_question_id", "scoring_day_et", "answer_option_id"])
        .agg(aggregate_probability=pl.col("site_probability").mean(), answer_order=pl.col("answer_order").first(), resolved_probability=pl.col("resolved_probability").first())
        .with_columns(aggregate_probability=pl.col("aggregate_probability") / pl.col("aggregate_probability").sum().over(["discover_question_id", "scoring_day_et"]))
        .with_columns(n_answer_options=pl.col("answer_option_id").n_unique().over(["discover_question_id", "scoring_day_et"]))
        .join(qday_meta.lazy().select(meta_cols), on=["discover_question_id", "scoring_day_et"], how="left")
        .with_columns(method=pl.lit("site_balanced_median"))
    )
    columns = [
        "discover_question_id",
        "scoring_day_et",
        "method",
        "answer_option_id",
        "answer_order",
        "aggregate_probability",
        "resolved_probability",
        "n_forecasters",
        "n_answer_options",
        "is_binary_single_yes",
        "is_ordinal",
        "subset",
        "scoring_rule_used",
        "correctness_known_date",
        "question_text",
    ]
    output_cast = [
        pl.col("n_forecasters").cast(pl.Int64),
        pl.col("n_answer_options").cast(pl.Int64),
        pl.col("answer_order").cast(pl.Float64),
    ]
    return pl.concat(
        [
            binary_yes.with_columns(output_cast).select(columns),
            binary_no.with_columns(output_cast).select(columns),
            multi_site.with_columns(output_cast).select(columns),
        ],
        how="vertical",
    ).collect()
