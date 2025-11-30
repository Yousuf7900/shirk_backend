import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load env vars for local dev. On Render, env vars are injected directly.
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

# Create a lightweight HF Inference client
hf_client = InferenceClient(model=HF_MODEL_ID, token=HF_API_TOKEN)

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
    """
    Use HF InferenceClient for text classification.
    This handles correct routing internally.
    """
    try:
        # This returns a list of dicts or list of list of dicts
        result = hf_client.text_classification(text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hugging Face Inference error: {str(e)}",
        )

@app.get("/")
def root():
    return {"message": "Shirk Detector API (HF Inference) is online 🚀"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput, x_api_key: str = Header(default=None)):
    verify_api_key(x_api_key)

    hf_result = query_hf_inference(input_data.text)

    # Expected formats:
    # 1) [{"label": "...", "score": ...}, ...]
    # 2) [[{"label": "...", "score": ...}, ...], ...]
    if isinstance(hf_result, list):
        if len(hf_result) == 0:
            raise HTTPException(status_code=500, detail="Empty response from Hugging Face")
        if isinstance(hf_result[0], dict):
            scores = hf_result          # single input
        elif isinstance(hf_result[0], list):
            scores = hf_result[0]       # batched input
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected HF response format: {hf_result}")
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected HF response: {hf_result}")

    # Convert to {label: prob}
    probs = {item["label"]: float(item["score"]) for item in scores}
    best_label, best_conf = max(probs.items(), key=lambda kv: kv[1])

    return PredictionOutput(
        label=best_label,
        confidence=best_conf,
        probs=probs,
    )
