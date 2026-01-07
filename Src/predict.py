import json
import joblib
import pandas as pd

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--json", required=True, help="JSON d'une ligne ou liste de lignes")
    args = p.parse_args()

    pipe = joblib.load(args.model)
    payload = json.loads(args.json)
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        df = pd.DataFrame(payload)

    preds = pipe.predict(df)
    print(preds.tolist())

if __name__ == "__main__":
    main()
