import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from src.analyze import analyze_df
from src.utils import train_tabular

app = FastAPI(title="Predictor DS Agent")

MODEL_PATH = "models/model.joblib"
REPORT_PATH = "models/report.json"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return {"analysis": analyze_df(df)}

@app.post("/train")
async def train(file: UploadFile = File(...), target: str = Form(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    pipe, report = train_tabular(df, target=target)

    import os, json
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)

    return {"trained": True, "report": report}

@app.post("/predict")
async def predict(payload: dict):
    pipe = joblib.load(MODEL_PATH)
    df = pd.DataFrame([payload])
    pred = pipe.predict(df)[0]

    out = {"prediction": pred}
    # proba si classification
    try:
        proba = pipe.predict_proba(df)
        out["proba"] = proba[0].tolist()
    except Exception:
        pass

    return out
