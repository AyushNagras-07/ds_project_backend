# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import io

# -----------------------
# Load Model + Artifacts
# -----------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Maps frontend names → model training names
FEATURE_MAP = {
    "Blood_Type": "Blood Type",
    "Medical_Condition": "Medical Condition",
    "Billing_Amount": "Billing Amount",
    "Admission_Type": "Admission Type",
}

# Columns that MUST NOT be renamed with spaces
SAFE_COLUMNS = {
    "Symptom_Severity",
    "Prior_Hospital_Visits",
    "High_BP_Flag",
    "High_Sugar_Flag",
    "Fever_Flag",
    "Risk_Score",
}

app = FastAPI(title="Healthcare Test Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Request Schema
# -----------------------
class PredictRequest(BaseModel):
    Age: float
    Gender: int
    Blood_Type: int
    Medical_Condition: int
    Billing_Amount: float
    Admission_Type: int
    Medication: int
    Symptom_Severity: float
    Prior_Hospital_Visits: int
    High_BP_Flag: int
    High_Sugar_Flag: int
    Fever_Flag: int
    Risk_Score: float


@app.get("/")
def root():
    return {"status": "online"}


# -----------------------
# Single Prediction
# -----------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        data = req.dict()
        renamed = {}

        # Rename columns safely
        for k, v in data.items():

            if k in SAFE_COLUMNS:     # Keep EXACT NAME
                renamed[k] = v
                continue

            if k in FEATURE_MAP:      # Convert FE → Training
                renamed[FEATURE_MAP[k]] = v
                continue

            # Default: replace _ with space
            renamed[k.replace("_", " ")] = v

        df = pd.DataFrame([renamed])

        # Apply encoders
        for col, enc in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except:
                    pass

        # Validate features
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing required features: {missing}")

        df = df[feature_columns]

        pred = int(model.predict(df)[0])

        label_map = {0: "Abnormal", 1: "Inconclusive", 2: "Normal"}
        label = label_map.get(pred, pred)

        return {"prediction_encoded": pred, "prediction_label": label}

    except Exception as e:
        raise HTTPException(500, str(e))


# -----------------------
# Batch Prediction
# -----------------------
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Apply renaming
        df = df.rename(columns=FEATURE_MAP)

        # Apply encoders
        for col, enc in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except:
                    pass

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing required features in CSV: {missing}")

        preds = model.predict(df[feature_columns])
        df["prediction_encoded"] = preds

        label_map = {0: "Abnormal", 1: "Inconclusive", 2: "Normal"}
        df["prediction_label"] = [label_map[int(p)] for p in preds]

        buf = io.StringIO()
        df.to_csv(buf, index=False)

        return {"filename": file.filename, "content": buf.getvalue()}

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
