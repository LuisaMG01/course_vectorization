from transformers import AutoTokenizer, AutoModel
import torch
from config import Config
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModel.from_pretrained(Config.MODEL_NAME)
embedder = pipeline("feature-extraction", model=Config.MODEL_NAME, tokenizer=Config.MODEL_NAME)

def generate_embedding(text):
    return embedder(text)[0][0] 