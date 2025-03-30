import torch
from fastapi import FastAPI
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pydantic import BaseModel

# Load model & tokenizer
model = DistilBertForSequenceClassification.from_pretrained("text_model")
tokenizer = DistilBertTokenizer.from_pretrained("text_model")


# Put model in evaluation mode
model.eval()

app = FastAPI()


class Payload(BaseModel):
    text: str = ""


@app.post("/predict")
def predict(payload: Payload):
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs).logits
    return {"prediction": output.tolist()}


# Run the API using: uvicorn app:app --reload
