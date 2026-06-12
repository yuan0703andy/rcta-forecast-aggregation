# Raw Data

The raw HFC/RCTA CSV files are public but too large to store in normal git
history. Download the release asset:

```text
https://github.com/yuan0703andy/rcta-forecast-aggregation/releases/download/raw-data-v1/hfc-rcta-raw-data.zip
```

Then unzip it into this folder:

```bash
unzip hfc-rcta-raw-data.zip -d data/raw/
```

After unzipping, this folder should contain:

```text
rct-a-questions-answers.csv
rct-a-prediction-sets.csv
rct-a-daily-forecasts.csv
```

The analysis can then be run from the repository root:

```bash
python analysis.py
```
