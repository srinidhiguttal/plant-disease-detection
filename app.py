from flask import Flask, render_template, request, jsonify
import os
from predict import predict  # your predict.py function

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home page
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    file_path = os.path.join(UPLOAD_FOLDER, "temp_leaf.jpg")
    file.save(file_path)

    try:
        disease, confidence, severity, action = predict(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "action": action
    })

if __name__ == "__main__":
    app.run(debug=True)
