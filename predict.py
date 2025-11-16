import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1️⃣ Model path
MODEL_PATH = "best_model.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Check path.")
print("✅ Model exists")

# 2️⃣ Load TFLite model
print("🔄 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded!")

# 3️⃣ Class labels
CLASS_NAMES = [
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot", "Corn(maize)Common_rust", "Corn(maize)_Northern_Leaf_Blight",
    "Corn_(maize)healthy", "Grape_Black_rot", "Grape_Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape__healthy", "Orange_Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,bell_Bacterial_spot", "Pepper,_bell_healthy", "Potato_Early_blight", "Potato__Late_blight",
    "Potato__healthy", "Raspberry_healthy", "Soybean_healthy", "Squash__Powdery_mildew",
    "Strawberry__Leaf_scorch", "Strawberry_healthy", "Tomato_Bacterial_spot", "Tomato__Early_blight",
    "Tomato__Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato__healthy"
]

# 4️⃣ Preprocess image
def preprocess_image(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 5️⃣ Predict function
def predict(img_path):
    img = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id = int(np.argmax(preds))
    prob = float(np.max(preds)) * 100

    # Disease name
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        disease = "Unknown"
    else:
        disease = CLASS_NAMES[class_id]

    # Spray decision based on severity
    if "healthy" in disease.lower():
        severity = "None"
        action = "No Spray Needed ✅ (0 ml)"
    elif prob < 70:
        severity = "Low 🟢"
        action = "Preventive Spray 🌿 (100 ml / plant)"
    elif prob < 90:
        severity = "Medium 🟡"
        action = "Targeted Spray 🌿 (150 ml / plant)"
    else:
        severity = "High 🔴"
        action = "Immediate Heavy Spray 🚜 (200 ml / plant)"

    return disease, prob, severity, action

# 6️⃣ Example test
IMAGE_PATH = os.path.join("static", "uploads", "apple.jpeg")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("Test image not found! Check path.")

disease, prob, severity, action = predict(IMAGE_PATH)
print(f"Prediction     : {disease}")
print(f"Infection Level: {severity}")
print(f"Spray Decision : {action}")

