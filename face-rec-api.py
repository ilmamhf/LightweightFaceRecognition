from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import face_recognition
import pickle
import numpy as np
import cv2
import base64
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load known faces and embeddings once at startup
model_path = "encodings.pickle"  # Update the path if needed
with open(model_path, "rb") as f:
    data = pickle.load(f)

# Threshold for face distance to mark recognized faces
THRESHOLD = 0.4

def read_image_file(file_stream):
    # Convert file stream to numpy array image (BGR)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def read_image_base64(base64_str):
    # Decode base64 string and convert to numpy image (BGR)
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

@app.route("/recognize-face", methods=["POST"])
def recognize_face():
    """
    Accepts an image file upload or base64 json, performs face recognition,
    returns list of detected faces with bounding boxes, names, and similarity scores.
    """
    if "image" in request.files:
        # Image sent as a file part
        image_file = request.files["image"]
        image = read_image_file(image_file)
    else:
        # Try to parse base64 json field 'image_base64'
        data_json = request.get_json()
        if data_json is None or "image_base64" not in data_json:
            return jsonify({"error": "No image file or base64 image provided"}), 400
        image = read_image_base64(data_json["image_base64"])

    if image is None:
        return jsonify({"error": "Invalid image data"}), 400

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb_image, model="hog")
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    faces_output = []

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        face_distances = face_recognition.face_distance(data["encodings"], encoding)

        name = "Unknown"
        similarity = None
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            best_distance = face_distances[best_match_index]
            if best_distance <= THRESHOLD:
                name = data["names"][best_match_index]
                similarity = round(float(best_distance), 4)

        faces_output.append({
            "name": name,
            "similarity": similarity,
            "bounding_box": {"top": top, "right": right, "bottom": bottom, "left": left}
        })

    response = {
        "num_faces": len(faces_output),
        "faces": faces_output
    }

    return jsonify(response)

if __name__ == "__main__":
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
