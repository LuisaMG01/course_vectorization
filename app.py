from uuid import uuid4
from flask import Flask, request, jsonify
from embeddings.embedder import generate_embedding
from qdrant.qdrant_manager import *
import pandas as pd
from utils.helpers import cleaner

app = Flask(__name__)

initialize_qdrant()

@app.route("/recommend", methods=["POST", "GET"])
def recommend():
    try:
        if request.method == "POST":
            data = request.get_json()
        else:
            data = {
                "vacant_name": request.args.get("vacant_name", ""),
                "vacant_description": request.args.get("vacant_description", "")
            }

        if not data:
            return jsonify({"error": "Missing request data"}), 400

        vacant_name = data.get("vacant_name", "")
        vacant_description = data.get("vacant_description", "")

        if not vacant_name or not vacant_description:
            return jsonify({"error": "Name and description for vacancy is not provided."}), 400

        job_vector = generate_embedding(vacant_name + " " + vacant_description)
        search_results = search_similar_courses(job_vector, limit=5)

        recommendations = [result.id for result in search_results]

        if not recommendations:
            return jsonify({"recommendations": []}), 200

        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        return jsonify({"message": "Course saved successfully.", "course_id": course_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/ping", methods=["GET"])
def index():
    return "Course Recommender API is running."

@app.route("/load-courses", methods=["POST"])
def load_courses():
    try:
        data = request.json
        
        if not data or "courses" not in data:
            return jsonify({"error": "Missing data or 'courses' key is not provided."}), 400

        courses = data["courses"]
        processed_courses = []
        errors = []

        for course in courses:
            try:
                if not all(key in course for key in ["id", "name", "description"]):
                    errors.append(f"Invalid course data: {course}")
                    continue

                course_id = course["id"]
                name = course["name"]
                description = course["description"]

                text = f"{name}. {description}"
                try:
                    vector = generate_embedding(text)
                except Exception as embed_error:
                    errors.append(f"Embedding generation failed for course {course_id}: {str(embed_error)}")
                    continue

                try:
                    upsert_course(
                        course_id=course_id,
                        name=name,
                        description=description,
                        vector=vector
                    )
                    processed_courses.append(course_id)
                except Exception as upsert_error:
                    errors.append(f"Upsert failed for course {course_id}: {str(upsert_error)}")

            except Exception as course_error:
                errors.append(f"Unexpected error processing course: {str(course_error)}")

        response = {
            "message": "Course processing completed",
            "processed_courses_count": len(processed_courses),
            "processed_course_ids": processed_courses
        }

        if errors:
            response["errors"] = errors
            print("Errors during course processing:", errors)

        print(f"Processed {len(processed_courses)} out of {len(courses)} courses")

        return jsonify(response), 201 if not errors else 206

    except Exception as e:
        print(f"Critical error in load_courses: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get-courses", methods=["GET"])
def get_courses():
    try:
        data = request.json
        if not data or "id_courses" not in data:
            return jsonify({"error": "Missing 'id' in the request parameters"}), 400
        
        courses = data["id_courses"]
        results = []
        for course in courses:
            result = retrieve_course(course)
            if result:
                results.append(result)
            else:
                results.append({"error": f"Course with ID {course} not found."})
        return jsonify({"courses": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete-course", methods=["DELETE"])
def delete_course_endpoint(): 
    try:
        data = request.json
        if not data or "point_id" not in data:
            return jsonify({"error": "Missing 'id' in the request parameters"}), 400

        course_id = data["point_id"]
        delete_course(course_id)  
        return jsonify({"message": f"Course with ID {course_id} deleted successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/delete-all-courses", methods=["DELETE"])
def delete_all_courses_endpoint():
    try:
        delete_all_courses()
        return jsonify({
            "message": "All courses have been deleted successfully."
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    from config import Config
    app.run(debug=True, port=Config.SERVER_PORT)