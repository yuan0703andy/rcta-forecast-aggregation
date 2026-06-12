WORKDIR = "/path/to/raw/hfc/folder"
#
# Put the original HFC/RCTA raw files in WORKDIR, or place them in
# data/raw/ and leave WORKDIR unchanged. Required files:
#
#   rct-a-questions-answers.csv
#   rct-a-prediction-sets.csv
#   rct-a-daily-forecasts.csv
#
# The main analysis uses rct-a-prediction-sets.csv for individual
# forecasts and rct-a-questions-answers.csv for outcomes/metadata.
# rct-a-daily-forecasts.csv is retained as a raw reference file but is
# not used as the source for individual-level aggregation.

from src.main import run_analysis


if __name__ == "__main__":
    run_analysis(WORKDIR)
