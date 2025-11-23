# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import io

# Load model + artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# label_encoders is a dict mapping column -> LabelEncoder (if you saved it)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

app = FastAPI(title="Healthcare Test Result Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for single prediction
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
    return {"status": "ok", "message": "Healthcare Test Prediction API"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.dict()])

        # apply encoders if your label_encoders contain encoders for some cols
        for col, enc in label_encoders.items():
            if col in df.columns:
                # if incoming value already numeric (encoded) skip transform
                try:
                    # assume encoder expects strings if it was trained on strings
                    # first try transform; if that fails, assume integer given and skip
                    df[col] = enc.transform(df[col].astype(str))
                except Exception:
                    # if transform fails, keep the value (likely already encoded)
                    df[col] = df[col]

        # Ensure all feature columns exist and in same order
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

        df = df[feature_columns]

        # Predict
        pred = model.predict(df)[0]

        # If you saved Test_Results encoder named 'Test_Results' or similar:
        label_encoder_name = None
        for k in label_encoders.keys():
            if k.lower().replace(" ", "_") in ("test_results", "test_results_encoded", "test result", "test_results"):
                label_encoder_name = k
                break

        if label_encoder_name:
            label = label_encoders[label_encoder_name].inverse_transform([pred])[0]
        else:
            # fallback mapping if numeric mapping used when training:
            label_map = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}
            label = label_map.get(int(pred), str(pred))

        return {"prediction_encoded": int(pred), "prediction_label": label}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch predict from uploaded CSV
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Keep copy to append predictions
        original = df.copy()

        # Apply encoders where necessary
        for col, enc in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = enc.transform(df[col].astype(str))
                except Exception:
                    df[col] = df[col]

        # Verify feature columns
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features in CSV: {missing}")

        X = df[feature_columns]
        preds = model.predict(X)
        original["prediction_encoded"] = preds

        # decode if possible
        label_encoder_name = None
        for k in label_encoders.keys():
            if k.lower().replace(" ", "_") in ("test_results", "test_results_encoded", "test result", "test_results"):
                label_encoder_name = k
                break
        if label_encoder_name:
            original["prediction_label"] = label_encoders[label_encoder_name].inverse_transform(preds.astype(int))
        else:
            label_map = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}
            original["prediction_label"] = [label_map.get(int(p), str(p)) for p in preds]

        # Return CSV bytes to download
        buf = io.BytesIO()
        original.to_csv(buf, index=False)
        buf.seek(0)
        return {
            "filename": f"predictions_{file.filename}",
            "content": buf.getvalue().decode("utf-8")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
