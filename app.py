"""
PlantPulse: Cotton Crop Foliar Disease Detection Platform
Flask web application serving real-time neural inference, interactive model comparison,
agronomy treatment recommendations, and localized agricultural weather forecast.
"""
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
# pyrefly: ignore[missing-import]
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import json
import uuid
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Class names based on the dataset (include 'Unknown' for unrelated images)
class_names = ['Aphids', 'Army worm', 'Bacterial blight', 
               'Healthy leaf', 'Powdery mildew', 'Target spot', 'Unknown']

# Solutions dictionary
solutions = {
    'Aphids': {
        'organic': 'Spray neem oil or use ladybugs as natural predators.',
        'chemical': 'Use insecticides like Imidacloprid or Pyrethroids.',
        'products': [
            {'name': 'Neem Oil Spray', 'url': 'https://www.amazon.in/gp/aw/d/B09ZRP5VMM/', 'image_url': '/static/images/neem-oil.jpg'},
            {'name': 'Imidacloprid 17.8% SL', 'url': 'https://www.amazon.in/s?k=Ulala+upl+comp', 'image_url': '/static/images/ulala.jpg'}
        ]
    },
    'Army worm': {
        'organic': 'Apply Bacillus thuringiensis (Bt) or use Trichogramma wasps.',
        'chemical': 'Spray Chlorantraniliprole or Lambda-cyhalothrin.',
        'products': [
            {'name': 'Bt-based Spray', 'url': 'https://www.amazon.in/PLANTS-BUDDY-Thuringiensis-Lepidopteran-Bio-Insecticide/dp/B0CQQSW51X/ref=sr_1_6?crid=3S40XHEOWANW5&dib=eyJ2IjoiMSJ9.VjAWKofo7OsR4DhFPI9BkXpg923wblwvpGk7FMSs0JSn3gj3woklh2fP3_ggzTy2l09Rpts98b7MztOcs94T5nuhtWpHygrX9kbAjcEXSnqLMxF8JOROpcFOmNu0qe1pTjQZ4Zl9VP-_lDpHu8yIX3IU2a1xRFbQ7w1ETzJLxo1xlm0wqHUNbfMAxCKtnFemgjAVvhwX5AqUbw0EHL1b4g.ELUyK5wJrd_RlAvH4qXrvCzOyr0ozGdubH8Ra4cIZAk&dib_tag=se&keywords=Bacillus+thuringiensis+%28Bt%29&qid=1736697698&sprefix=bacillus+thuringiensis+bt+%2Caps%2C247&sr=8-6', 'image_url': '/static/images/bacillus.jpg'},
            {'name': 'Chlorantraniliprole 18.5% SC', 'url': 'https://www.amazon.in/SWASTIK-ICON-Insecticide-Lambda-Cyhalothrin/dp/B07ZTJM7VR/ref=sr_1_2_sspa?crid=2HE2900BYUG8W&dib=eyJ2IjoiMSJ9.IpQg5xuz9SFcvExt_OZZzXUzDLcK77_fRFZ0IunsccIQGBV5nqWEYVlJhf-9ZI8J7z9XlTNemm1zE95Ilh7Y19KWBkZ89RT8JUM4QvQ-lygTw2BW1l-FgPff6wt90HviB46W8Y0rcVwfGXebU4OHrHHTEHv3oTYM50DMDRlw1g403YPnRztNF40C3p3XE-2qcNzO9ngnjJeOpW9A0i7SsgDHnfrRhXlsiEPLydEFzdWPr6NvWEmlMY5mHgRwqb4Q3rOAVDxWDaGmSwLgRtW99UN2z5emYXKepP71f06bWmg.Z3GyeFmHEPoRLlpCsPrVWstK7HVKFE7QF6w3qmP6z3I&dib_tag=se&keywords=lambda+cyhalothrin&qid=1739504559&sprefix=cyhalothrin.%2Caps%2C350&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1', 'image_url': '/static/images/coragen.jpg'}
        ]
    },
    'Bacterial blight': {
        'organic': 'Use copper-based fungicides or compost tea sprays.',
        'chemical': 'Apply Streptomycin sulfate or Copper oxychloride.',
        'products': [
            {'name': 'Copper Fungicide', 'url': 'https://www.desertcart.in/products/2343720-bonidecaptain-jack-s-copper-fungicide', 'image_url': '/static/images/liquid-copper-concentrate.jpg'},
            {'name': 'Streptomycin Sulfate Powder', 'url': 'https://www.amazon.in/s?k=streptocycline+vector', 'image_url': '/static/images/vector.jpg'}
        ]
    },
    'Healthy leaf': {
        'organic': 'No treatment required. Maintain proper care.',
        'chemical': 'Not applicable.',
        'products': []
    },
    'Powdery mildew': {
        'organic': 'Spray sulfur or a milk solution (1:9 milk to water).',
        'chemical': 'Apply fungicides like Propiconazole or Myclobutanil.',
        'products': [
            {'name': 'Sulfur Powder', 'url': 'https://www.amazon.in/SENTALAB-Sulphur-Powder-NO-7704-34-9-Purity/', 'image_url': '/static/images/sulfur.jpg'},
            {'name': 'Propiconazole 25% EC', 'url': 'https://www.bighaat.com/collections/carbendazim-50', 'image_url': '/static/images/sprint.jpg'}
        ]
    },
    'Powder mildew': {
        'organic': 'Spray sulfur or a milk solution (1:9 milk to water).',
        'chemical': 'Apply fungicides like Propiconazole or Myclobutanil.',
        'products': [
            {'name': 'Sulfur Powder', 'url': 'https://www.amazon.in/SENTALAB-Sulphur-Powder-NO-7704-34-9-Purity/', 'image_url': '/static/images/sulfur.jpg'},
            {'name': 'Propiconazole 25% EC', 'url': 'https://www.bighaat.com/collections/carbendazim-50', 'image_url': '/static/images/sprint.jpg'}
        ]
    },
    'Target spot': {
        'organic': 'Apply compost tea or neem oil regularly.',
        'chemical': 'Use Chlorothalonil or Mancozeb fungicides.',
        'products': [
            {'name': 'Neem Oil Spray', 'url': 'https://www.amazon.in/gp/aw/d/B09ZRP5VMM/', 'image_url': '/static/images/neem2.jpeg'},
            {'name': 'Mancozeb 75% WP', 'url': 'https://www.crystalcropprotection.com/cropprotection/detail/blue-copper', 'image_url': '/static/images/blue-copper.jpg'}
        ]
    },
    'Unknown': {
        'message': 'Unrelated image detected. Please upload an image related to cotton plants.'
    }
}

# ---------------------------------------------------------------------------
# MODEL REGISTRY & INITIALIZATION
# ---------------------------------------------------------------------------
def read_accuracy_from_metrics(metrics_filename, fallback="--"):
    """Reads test accuracy dynamically from results JSON."""
    metrics_path = os.path.join(RESULTS_DIR, metrics_filename)
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                acc = data.get("test_accuracy")
                if acc is not None:
                    return f"{acc * 100:.2f}%"
        except Exception:
            pass
    return fallback

MODEL_DEFINITIONS = [
    {
        "id": "mobilenetv2",
        "name": "MobileNetV2 Transfer Learning",
        "path": os.path.join(BASE_DIR, "models", "algo2.h5"),
        "metrics_file": "algo2_metrics.json",
        "badge": "Recommended",
        "is_default": True,
        "default_acc": "88.54%"
    },
    {
        "id": "custom_cnn",
        "name": "Sequential 3-Block CNN",
        "path": os.path.join(BASE_DIR, "models", "algo1.h5"),
        "metrics_file": "algo1_metrics.json",
        "badge": "From Scratch",
        "is_default": False,
        "default_acc": "74.74%"
    }
]

LOADED_MODELS = {}

def initialize_models():
    """Loads all existing models at startup to avoid re-loading on each request."""
    print("=" * 60)
    print("Initializing PlantPulse Model Registry...")
    for item in MODEL_DEFINITIONS:
        path = item["path"]
        if os.path.exists(path):
            try:
                print(f"Loading {item['name']} from {path}...")
                model = tf.keras.models.load_model(path)
                LOADED_MODELS[item["id"]] = {
                    "id": item["id"],
                    "name": item["name"],
                    "badge": item["badge"],
                    "metrics_file": item["metrics_file"],
                    "default_acc": item["default_acc"],
                    "model": model,
                    "is_default": item["is_default"]
                }
                print(f"Successfully loaded {item['name']}.")
            except Exception as e:
                print(f"Warning: Failed to load {item['name']} ({path}): {e}")
        else:
            print(f"Info: Model file not found (skipping): {path}")
    print(f"Active models loaded: {list(LOADED_MODELS.keys())}")
    print("=" * 60)

initialize_models()

def get_available_models_info():
    """Returns available models metadata with up-to-date accuracy."""
    info_list = []
    for item in MODEL_DEFINITIONS:
        m_id = item["id"]
        if m_id in LOADED_MODELS:
            metrics_file = item["metrics_file"]
            acc = read_accuracy_from_metrics(metrics_file, fallback=item["default_acc"]) if metrics_file else item["default_acc"]
            info_list.append({
                "id": m_id,
                "name": item["name"],
                "badge": item["badge"],
                "accuracy": acc,
                "is_default": item["is_default"]
            })
    return info_list

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    models_info = get_available_models_info()
    return render_template("index.html", available_models=models_info)

@app.route("/learn")
def learn():
    return render_template("learn.html")

@app.route("/al")
def AI():
    return render_template("al.html")

@app.route('/weather', methods=['GET', 'POST'])
def home():
    weather_data = None
    if request.method == 'POST':
        city = request.form.get('city', '').strip()
        api_key = os.getenv("OPENWEATHER_API_KEY", "")  # Set OPENWEATHER_API_KEY in your .env file
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    "city": data["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"].capitalize(),
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "pressure": data["main"]["pressure"],
                    "icon": data["weather"][0]["icon"]
                }
            else:
                weather_data = {"error": "City not found. Please try again."}
        except Exception:
            weather_data = {"error": "Could not connect to weather service."}

    return render_template('weather.html', weather=weather_data)

@app.route("/comparison")
def comparison():
    m1 = {}
    m2 = {}
    try:
        p1 = os.path.join(RESULTS_DIR, "algo1_metrics.json")
        p2 = os.path.join(RESULTS_DIR, "algo2_metrics.json")
        if os.path.exists(p1):
            with open(p1, "r", encoding="utf-8") as f:
                m1 = json.load(f)
        if os.path.exists(p2):
            with open(p2, "r", encoding="utf-8") as f:
                m2 = json.load(f)
    except Exception as e:
        print(f"Error loading metrics in /comparison: {e}")
        
    return render_template("comparison.html", metrics_m1=m1, metrics_m2=m2)

@app.route("/results-image/<path:filename>")
def results_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("error.html", message="No file uploaded. Please upload an image.")

    file = request.files['file']
    if not file.filename:
        return render_template("error.html", message="No file selected. Please choose a valid image file.")

    if not allowed_file(file.filename):
        return render_template("error.html", message="Unsupported file type. Please upload a valid image (PNG, JPG, JPEG, or WEBP).")

    # Determine requested algorithm
    selected_id = request.form.get("model_id", "mobilenetv2")
    if selected_id not in LOADED_MODELS:
        # Fallback to default or first available loaded model
        if "mobilenetv2" in LOADED_MODELS:
            selected_id = "mobilenetv2"
        elif LOADED_MODELS:
            selected_id = list(LOADED_MODELS.keys())[0]
        else:
            return render_template("error.html", message="No active trained models found on the server.")

    model_entry = LOADED_MODELS[selected_id]
    active_model = model_entry["model"]
    model_name = model_entry["name"]
    metrics_file = model_entry["metrics_file"]
    model_acc = read_accuracy_from_metrics(metrics_file, fallback=model_entry["default_acc"]) if metrics_file else model_entry["default_acc"]

    try:
        # Generate secure unique filename and save to static/uploads
        file_ext = os.path.splitext(file.filename)[1].lower()
        base_name = secure_filename(os.path.splitext(file.filename)[0]) or "leaf"
        safe_filename = f"{uuid.uuid4().hex[:8]}_{base_name}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction with the chosen model
        predictions = active_model.predict(img_array)
        predicted_idx = int(np.argmax(predictions))
        predicted_class = class_names[predicted_idx]
        confidence_val = float(np.max(predictions))
        confidence_percent = f"{confidence_val * 100:.1f}%"

        # Web-accessible relative static path
        display_img_path = f"uploads/{safe_filename}"

        # Handle unrelated images with a confidence threshold
        if predicted_class == 'Unknown' or confidence_val < 0.60:
            return render_template(
                "result.html",
                image_path=display_img_path,
                model_name=model_name,
                model_accuracy=model_acc,
                confidence=confidence_percent,
                predicted_class='Unrelated image detected. Please upload an image related to cotton plants.',
                solutions={'message': 'Unrelated image detected. Please upload an image related to cotton plants.'}
            )

        # Get solutions for the predicted class
        solution = solutions.get(predicted_class, {})
        return render_template(
            "result.html",
            image_path=display_img_path,
            model_name=model_name,
            model_accuracy=model_acc,
            confidence=confidence_percent,
            predicted_class=predicted_class,
            solutions=solution
        )

    except Exception as e:
        return render_template("error.html", message=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)