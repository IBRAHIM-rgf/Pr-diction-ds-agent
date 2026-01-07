import json
import joblib
import pandas as pd
from pathlib import Path
from utils import train_tabular

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out_model", default="models/model.joblib")
    p.add_argument("--out_report", default="models/report.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    model, report = train_tabular(df, target=args.target)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, args.out_model)
    Path(args.out_report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("✅ Trained:", report)

if __name__ == "__main__":
    main()
