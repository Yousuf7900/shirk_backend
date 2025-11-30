import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load env (for local dev)
load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID")
API_KEY = os.getenv("API_KEY")

if HF_MODEL_ID is None:
    raise RuntimeError("HF_MODEL_ID not set in environment")
if API_KEY is None:
    raise RuntimeError("API_KEY not set in environment")

app = FastAPI(title="Bangla Shirk Detector API")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
model.eval()

# Could be dict or list
id2label = model.config.id2label


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    confidence: float
    probs: dict


def verify_api_key(api_key: str | None):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def root():
    return {"message": "Shirk Detector API is online 🚀"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()

    pred_id = int(torch.argmax(logits, dim=-1).item())

    # For dict id2label
    if isinstance(id2label, dict):
        pred_label = id2label[str(pred_id)]
    else:
        pred_label = id2label[pred_id]

    prob_dict = {}
    for i, p in enumerate(probs):
        if isinstance(id2label, dict):
            key = id2label[str(i)]
        else:
            key = id2label[i]
        prob_dict[key] = float(p)

    return PredictionOutput(
        label=pred_label,
        confidence=prob_dict[pred_label],
        probs=prob_dict,
    )
