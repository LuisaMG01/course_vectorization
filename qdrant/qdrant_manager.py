from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import Config
import pandas as pd
from transformers import pipeline

embedder = pipeline("feature-extraction", model=Config.MODEL_NAME, tokenizer=Config.MODEL_NAME)

client = QdrantClient(
    url=f"https://{Config.QDRANT_HOST}",  
    port=Config.QDRANT_PORT,
    api_key=Config.QDRANT_API_KEY       
)

def initialize_qdrant():
    try:
        existing_collections = client.get_collections()
        print("Existing collections response:", existing_collections)
        
        if isinstance(existing_collections, dict) and 'collections' in existing_collections:
            existing_collection_names = [
                col['name'] for col in existing_collections['collections']
            ]  
        else:
            existing_collection_names = []  

        if Config.COLLECTION_NAME not in existing_collection_names:
            client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                timeout=60 
            )
            print("Collection created successfully")
        else:
            print(f"Collection '{Config.COLLECTION_NAME}' already exists. Using existing collection.")
    except Exception as e:
        print("Error initializing Qdrant:", e)



def upsert_course(course_id, name, description, vector):
    try:
        client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=course_id,
                    vector=vector,
                    payload={
                        "name": name,
                        "description": description
                    }
                )
            ]
        )
        print(f"Course {course_id} upserted successfully.")
    except Exception as e:
        print(f"Error upserting course {course_id}: {e}")


def search_similar_courses(vector, limit=5):
    try:
        results = client.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=vector,
            limit=limit
        )
        if not results:
            print("No similar courses found.")
        return results
    except Exception as e:
        print("Error searching similar courses:", e)
        return []

