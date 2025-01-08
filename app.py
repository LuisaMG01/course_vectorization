from uuid import uuid4
from flask import Flask, request, jsonify
from embeddings.embedder import generate_embedding
from qdrant.qdrant_manager import initialize_qdrant, upsert_course, search_similar_courses
import pandas as pd

app = Flask(__name__)

initialize_qdrant()

@app.route("/recommend", methods=["GET"])
def recommend():
    data = request.get_json()
    vacant_name = data.get("name", "")
    vacant_description = data.get("description", "")

    if not vacant_name or not vacant_description:
        return jsonify({"error": "Debe proporcionar nombre y descripción de la vacante"}), 400

    job_vector = generate_embedding(vacant_name + " " + vacant_description)

    search_results = search_similar_courses(job_vector, limit=5)

    recommendations = [
        result.id for result in search_results
    ]

    return jsonify({"recomendaciones": recommendations})

@app.route("/load-course", methods=["POST"])
def load_course():
    try:
        data = request.json
        if not data or "name" not in data or "description" not in data:
            return jsonify({"error": "data"}), 400

        name = data["name"]
        description = data["description"]
        
        text = f"{name}. {description}"
        
        vector = generate_embedding(text)
        course_id = str(uuid4())

        upsert_course(
            course_id=course_id,
            name=name,
            description=description,
            vector=vector
        )

        return jsonify({"message": "Curso almacenado exitosamente.", "course_id": course_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/ping", methods=["GET"])
def index():
    return "API de recomendación de cursos"

@app.route("/load-courses", methods=["POST"])
def load_courses():
    try:
        data = request.json

        if not data or "courses" not in data:
            return jsonify({"error": "Faltan datos o la clave 'courses' no está presente."}), 400

        courses = data["courses"]

        for course in courses:
            course_id = course["id"]
            name = course["name"]
            description = course["description"]
            text = f"{name}. {description}"
            vector = generate_embedding(text)

            upsert_course(course_id=course_id, name=name, description=description, vector=vector)

        print(f"Se almacenaron {len(courses)} cursos.")
        return jsonify({"message": "Cursos almacenados exitosamente."}), 201

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from config import Config
    app.run(debug=True, port=Config.SERVER_PORT)
