from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import io

# ------------------------------------------------------
# LOAD MODEL + ENCODERS + FEATURE COLUMNS
# ------------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ------------------------------------------------------
# FASTAPI APP SETUP
# ------------------------------------------------------
app = FastAPI(title="Healthcare Test Result Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Your React/Vercel frontend works
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# REQUEST SCHEMA (Matches EXACT model training)
# ------------------------------------------------------
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
    return {"status": "OK", "message": "Backend API running successfully!"}


# ------------------------------------------------------
# 🔥 SINGLE PREDICTION ENDPOINT
# ------------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):

    try:
        # Convert input to dataframe
        df = pd.DataFrame([req.dict()])

        # ----------------------------------------
        # Apply label encoders (ONLY for columns that need it)
        # ----------------------------------------
        for col in df.columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except Exception:
                    pass   # already encoded

        # ----------------------------------------
        # Ensure feature order matches training
        # ----------------------------------------
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing}"
            )

        df = df[feature_columns]

        # ----------------------------------------
        # Prediction
        # ----------------------------------------
        pred = model.predict(df)[0]

        # Decode test result label
        label_encoder_key = None
        for k in label_encoders:
            if "test" in k.lower():
                label_encoder_key = k
                break

        if label_encoder_key:
            decoded_label = label_encoders[label_encoder_key].inverse_transform([pred])[0]
        else:
            label_map = {0: "ABNORMAL", 1: "INCONCLUSIVE", 2: "NORMAL"}
            decoded_label = label_map.get(int(pred), str(pred))

        return {
            "prediction_encoded": int(pred),
            "prediction_label": decoded_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# 🔥 CSV BATCH PREDICTION
# ------------------------------------------------------
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        original = df.copy()

        # Apply encoders
        for col in df.columns:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except Exception:
                    pass

        # Ensure every required feature is present
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features in CSV: {missing}"
            )

        X = df[feature_columns]
        preds = model.predict(X)

        original["prediction_encoded"] = preds

        # Decode labels
        label_encoder_key = None
        for k in label_encoders:
            if "test" in k.lower():
                label_encoder_key = k
                break

        if label_encoder_key:
            original["prediction_label"] = label_encoders[label_encoder_key].inverse_transform(preds)
        else:
            label_map = {0: "ABNORMAL", 1: "INCONCLUSIVE", 2: "NORMAL"}
            original["prediction_label"] = [label_map[int(p)] for p in preds]

        buf = io.BytesIO()
        original.to_csv(buf, index=False)
        buf.seek(0)

        return {
            "filename": f"predictions_{file.filename}",
            "content": buf.getvalue().decode()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# Production entrypoint
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
