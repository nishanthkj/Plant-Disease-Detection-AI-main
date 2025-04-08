from flask import Flask, render_template, request, jsonify, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision import models
import google.generativeai as genai
import tempfile
import mimetypes
import io
import base64
# ✅ Set your Gemini API key
genai.configure(api_key="AIzaSyCJ1Zt3_Zyez3S1bS1EPFHpLF-qmzERqwE")  # 🔁 Replace with your actual Gemini API key

# ✅ Initialize Gemini model
gemini_model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction=(
        "You are AgriBot, an expert agriculture assistant. Only respond to questions related to agriculture, "
        "like farming, crop diseases, irrigation, soil health, fertilizers, pests, and related topics. "
        "If someone asks anything unrelated (e.g., politics, sports, tech, etc), politely respond: "
        "'❗ I'm AgriBot, your agriculture assistant. I can only help with farming and crop-related topics.'"
    )
)

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load PyTorch ResNet model for crop disease prediction
resnet_model = models.resnet18(weights='IMAGENET1K_V1')
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 15)
resnet_model.load_state_dict(torch.load('crop_disease_simple_undersample2.pth', map_location=torch.device('cpu')))
resnet_model.eval()

# ✅ Class labels
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ✅ Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message")
    try:
        response = gemini_model.generate_content(user_msg)
        reply = response.text
    except Exception as e:
        print("Gemini Error:", e)
        reply = "⚠️ Oops! Something went wrong. Please try again."
    return jsonify({"response": reply})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if file:
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=".")
        filepath = temp_file.name
        file.save(filepath)

        # Predict disease
        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = resnet_model(image)
            _, predicted = torch.max(output, 1)
            prediction = class_names[predicted.item()]

        # Ask Gemini for explanation
        prompt = f"What is {prediction}? Explain it in simple terms, symptoms, and treatments."
        try:
            gemini_response = gemini_model.generate_content(prompt)
            disease_info = gemini_response.text
        except Exception as e:
            print("Gemini error:", e)
            disease_info = "⚠️ Could not fetch disease information. Please try again."

        filename = os.path.basename(filepath)
        return jsonify({
            "prediction": prediction,
            "image_url": f"/temp-image/{filename}",
            "disease_info": disease_info
        })

    return jsonify({"error": "No image uploaded"}), 400

@app.route("/temp-image/<filename>")
def serve_temp_image(filename):
    path = os.path.join(".", filename)
    mimetype = mimetypes.guess_type(path)[0] or 'image/jpeg'
    return send_file(path, mimetype=mimetype)

if __name__ == "__main__":
    app.run(debug=True)
