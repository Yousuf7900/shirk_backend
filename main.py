import os
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env vars when running locally.
# On Render, env vars are injected automatically but load_dotenv() doesn't hurt.
load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_KEY = os.getenv("API_KEY")

if HF_MODEL_ID is None:
    raise RuntimeError("HF_MODEL_ID not set in environment")
if HF_API_TOKEN is None:
    raise RuntimeError("HF_API_TOKEN not set in environment")
if API_KEY is None:
    raise RuntimeError("API_KEY not set in environment")

HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"

app = FastAPI(title="Bangla Shirk Detector API (HF Inference)")

# Allow browser frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    probs: dict

def verify_api_key(api_key: str | None):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def query_hf_inference(text: str):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text}

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        # HF may respond with {"error": "..."} while model is loading.
        raise HTTPException(
            status_code=500,
            detail=f"Hugging Face Inference API error: {response.status_code}, {response.text}",
        )

    return response.json()

@app.get("/")
def root():
    return {"message": "Shirk Detector API (HF Inference) is online 🚀"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)

    hf_result = query_hf_inference(input_data.text)

    # Expected formats:
    # 1) Single input: [ {"label": "...", "score": ...}, ... ]
    # 2) Batched: [ [ {"label": "...", "score": ...}, ... ], ... ]
    if isinstance(hf_result, list):
        if len(hf_result) == 0:
            raise HTTPException(status_code=500, detail="Empty response from Hugging Face")
        if isinstance(hf_result[0], dict):
            scores = hf_result                  # case 1: single input
        elif isinstance(hf_result[0], list):
            scores = hf_result[0]               # case 2: first element of batch
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected HF response format: {hf_result}")
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected HF response: {hf_result}")

    probs = {item["label"]: float(item["score"]) for item in scores}
    # Choose best label
    best_label, best_conf = max(probs.items(), key=lambda kv: kv[1])

    return PredictionOutput(
        label=best_label,
        confidence=best_conf,
        probs=probs,
    )
