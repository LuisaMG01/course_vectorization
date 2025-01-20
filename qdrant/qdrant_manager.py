from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import PointIdsList, Filter, FieldCondition, MatchValue, FilterSelector
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

def retrieve_course(id):
    try:
        response = client.retrieve(
            collection_name=Config.COLLECTION_NAME,
            ids=[id]
        )
        if response:
            result = [
                {
                    "id": record.id,
                    "vector": record.vector,
                    "payload": record.payload
                }
                for record in response
            ]
            return result[0] if result else None
        else:
            print(f"Failed to get point with ID {id}. Response: {response}")
            return None
    except Exception as e:
        print(f"Error getting course {id}: {e}")
        return None

def delete_course(point_id):
    try:
        client.delete(
            collection_name=Config.COLLECTION_NAME,
            points_selector=PointIdsList(
                points=[point_id]
            )
        )
        print(f"Course {point_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting course {point_id}: {e}")

def get_all_point_ids():
    try:
        all_points = []
        offset = None
        
        while True:
            results = client.scroll(
                collection_name=Config.COLLECTION_NAME,
                limit=100, 
                offset=offset
            )[0]
            
            if not results:
                break
                
            point_ids = [point.id for point in results]
            all_points.extend(point_ids)
            
            if len(results) < 100: 
                break
                
        return all_points
    except Exception as e:
        print(f"Error getting point IDs: {e}")
        raise e

def delete_all_courses():
    try:
        all_point_ids = get_all_point_ids()
        
        for point_id in all_point_ids:
            delete_course(point_id)
            
        return len(all_point_ids)
    except Exception as e:
        print(f"Error deleting all courses: {e}")
        raise e