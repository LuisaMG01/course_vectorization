import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "cursos")
    MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
