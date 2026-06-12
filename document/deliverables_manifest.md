# Deliverables Manifest

## Directory Layout

```text
analysis.py                 evaluator-facing wrapper; first line is WORKDIR
src/main.py                 pipeline orchestration
src/data.py                 raw-data loading and latest forecast-vector construction
src/aggregation.py          baseline and final aggregation rules
src/scoring.py              Brier/JNW scoring, summaries, cap sensitivity, bootstrap
data/raw/                   raw-data landing folder and download instructions
data/prompt/                original work-test prompt PDF
document/                   LaTeX source, PDF, and documentation
output/table/               calculation outputs written by analysis.py
output/figure/              static figures referenced by the memo
requirements.txt            Python package requirements
```

The previous exploratory experiments, old pandas generator, prototype scripts, and earlier submission-index folder were removed. The remaining repository contains the current calculation pipeline, memo PDF/LaTeX source, output tables, and instructions for obtaining the public raw data release.

## How to Rerun

The root `analysis.py` is the primary runnable entry point. Its first line remains:

```python
WORKDIR = "/path/to/raw/hfc/folder"
```

The raw CSV files are available as the GitHub Release asset `hfc-rcta-raw-data.zip`. There are two supported modes:

1. Download and unzip the raw data release into `data/raw/`, leave `WORKDIR` unchanged, and run from this repository.
2. Edit `WORKDIR` to point to another folder containing the raw HFC files.

Then run:

```bash
python analysis.py
```

The script is calculation-only. The root `analysis.py` calls `src.main.run_analysis`, and the source modules use Polars/vectorized group operations to write result tables to `output/table/`. The calculation pipeline does not generate the memo text or figures.

## Raw Data

| Location | Contents |
|---|---|
| `data/raw/README.md` | Instructions for downloading and unpacking the raw data release |
| GitHub Release asset `hfc-rcta-raw-data.zip` | Public raw RCTA CSV files used by the pipeline |
| `data/raw/rct-a-questions-answers.csv` after unzipping | RCTA question metadata, answer options, and resolved outcomes |
| `data/raw/rct-a-prediction-sets.csv` after unzipping | Individual forecast submissions used for the main analysis |
| `data/raw/rct-a-daily-forecasts.csv` after unzipping | Performer daily forecasts, retained as a raw reference file |
| `data/prompt/FRI 10-Hour Work Test (Data).pdf` | Original work-test prompt |

## Main Documents

| File | Purpose |
|---|---|
| `final_memo.pdf` | Root copy of the submitted memo PDF for quick reading |
| `document/final_memo.tex` | LaTeX source generated from the memo |
| `document/final_memo.pdf` | Compiled memo PDF |
| `document/deliverables_manifest.md` | This manifest |

## Tables Produced by `analysis.py`

| File | Purpose |
|---|---|
| `output/table/baseline_scores_question_balanced.csv` | Five baseline scores under question-balanced weighting |
| `output/table/baseline_scores_question_day_weighted.csv` | Five baseline scores under question-day weighting |
| `output/table/improved_scores_question_balanced.csv` | Final method score under question-balanced weighting |
| `output/table/improved_scores_question_day_weighted.csv` | Final method score under question-day weighting |
| `output/table/final_method_ablation_full.csv` | Median, original GMO, capped GMO, site-balanced median, and final method ablation |
| `output/table/final_method_bootstrap_vs_median.csv` | Paired bootstrap over questions for final method versus median |
| `output/table/site_balanced_cap_sensitivity.csv` | Cap sensitivity for site-balanced bounded evidence pooling |
| `output/table/latest_vectors.parquet` | Latest individual forecast vectors used by the calculation pipeline |
| `output/table/diagnostic_murphy_decomposition.csv` | Murphy decomposition supporting Section 6 |
| `output/table/diagnostic_brier_stratification.csv` | Error stratification by question type, forecaster count, dispersion, and forecast age |
| `output/table/diagnostic_tested_conditional_improvements_summary.csv` | Summary of tested conditional alternatives discussed in Section 7 |
| `output/table/diagnostic_gmo_tail_autopsy_summary.csv` | GMO-vs-median tail-risk summary supporting Section 8 |
| `output/table/diagnostic_gmo_vs_median_question_delta.csv` | Per-question GMO minus median loss diagnostics |
| `output/table/diagnostic_relative_to_individual_summary.csv` | Aggregate rank relative to individual forecasters |
| `output/table/diagnostic_relative_to_individual_question_day.csv` | Question-day level relative-to-individual diagnostics |
| `output/table/diagnostic_rolling_window_validation.csv` | Rolling-window temporal validation supporting Section 11 |

## Static Memo Assets

The current `analysis.py` does not generate figures. Existing figures in `output/figure/` are retained as static memo assets:

| File | Purpose |
|---|---|
| `output/figure/baseline_two_metrics.png` | Baseline scores under both estimands |
| `output/figure/step4_ablation.png` | Final method ablation |
| `output/figure/rolling_window_validation.png` | Rolling-window validation |

## Recommended Reader Path

1. `final_memo.pdf`
2. `analysis.py`
3. `output/table/baseline_scores_question_balanced.csv`
4. `output/table/baseline_scores_question_day_weighted.csv`
5. `output/table/improved_scores_question_balanced.csv`
6. `output/table/improved_scores_question_day_weighted.csv`
7. `output/table/final_method_ablation_full.csv`
