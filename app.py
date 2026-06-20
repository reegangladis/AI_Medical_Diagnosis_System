import os
import io
import sqlite3
from datetime import datetime
import ollama
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from flask import Flask, render_template, request, url_for, jsonify, Response, send_file, redirect, session
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import csv
from flask import Response
import zipfile
from flask import send_file
import google.generativeai as genai
from flask import jsonify
import os
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = "ai_healthcare_secret_123"



DOCTOR_MAP = {
    "pneumonia": "Pulmonologist",
    "tb": "Chest Specialist",
    "brain": "Neurologist",
    "skin": "Dermatologist",
    "bone": "Orthopedic Specialist",
    "lung_cancer": "Oncologist",
    "malaria": "Pathologist"
}
# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "database", "app.db")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")
HEATMAPS_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(HEATMAPS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

IMG_SIZE = (224, 224)

# ----------------------------
# DB init (auto creates table)
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            disease TEXT,
            prediction TEXT,
            confidence REAL,
            severity TEXT,
            emergency INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            report_path TEXT,
            heatmap_path TEXT,
            patient_name TEXT,
            patient_id TEXT,
            age INTEGER,
            gender TEXT,
            user_id INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'patient'
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT,
            hospital_name TEXT,
            doctor_type TEXT,
            appointment_date TEXT,
            appointment_time TEXT,
            location TEXT,
            map_link TEXT,
            status TEXT DEFAULT 'Booked',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Load models safely
# ----------------------------
def safe_load_model(path, name):
    if not os.path.exists(path):
        print(f"⚠ {name} not found: {path}")
        return None
    try:
        print(f"✅ Loading {name}...")
        m = tf.keras.models.load_model(path, compile=False)
        # warm-up
        _ = m(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32), training=False)
        return m
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        return None

PNEUMONIA_MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_model.h5")
TB_MODEL_PATH = os.path.join(BASE_DIR, "models", "tb_model.h5")
BRAIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_model.h5")
SKIN_MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_cancer_model.h5")
BONE_MODEL_PATH = os.path.join(BASE_DIR, "models", "bone_model.h5")
LUNG_MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_cancer_model.h5")
MALARIA_MODEL_PATH = os.path.join(BASE_DIR, "models", "malaria_model.h5")

malaria_model = safe_load_model(MALARIA_MODEL_PATH, "Malaria Model")
pneumonia_model = safe_load_model(PNEUMONIA_MODEL_PATH, "Pneumonia Model")
tb_model = safe_load_model(TB_MODEL_PATH, "TB Model")
brain_model = safe_load_model(BRAIN_MODEL_PATH, "Brain Tumor Model")
skin_model = safe_load_model(SKIN_MODEL_PATH, "Skin Cancer Model")
bone_model = safe_load_model(BONE_MODEL_PATH, "Bone Fracture Model")
lung_model = safe_load_model(LUNG_MODEL_PATH, "Lung Cancer Model")

print("✅ Model loading finished")

# ----------------------------
# Helpers
# ----------------------------
def get_severity(conf: float) -> str:
    if conf >= 85:
        return "Severe"
    elif conf >= 70:
        return "Moderate"
    return "Mild"

def is_emergency(conf: float) -> bool:
    return conf >= 90

# ----------------------------
# PDF report
# ----------------------------
def generate_pdf_report(filename, disease, prediction, confidence, severity, emergency, heatmap_path=None,
                        patient_name="", patient_id="", age="", gender=""):

    report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    full_report_path = os.path.join(REPORTS_FOLDER, report_name)

    c = canvas.Canvas(full_report_path, pagesize=A4)
    width, height = A4

    # =========================
    # PAGE 1 : HEADER SUMMARY
    # =========================

    c.setFillColor(colors.HexColor("#0f4c81"))
    c.rect(0, height-70, width, 70, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(130, height-42, "AI HEALTHCARE DIAGNOSIS REPORT")

    c.setFillColor(colors.black)

    # patient box
    c.setStrokeColor(colors.HexColor("#0f4c81"))
    c.rect(40, height-210, width-80, 120, fill=0)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(55, height-110, "Patient Information")

    c.setFont("Helvetica", 11)
    c.drawString(55, height-130, f"Patient Name : {patient_name}")
    c.drawString(55, height-148, f"Patient ID   : {patient_id}")
    c.drawString(55, height-166, f"Age / Gender : {age} / {gender}")
    c.drawString(55, height-184, f"Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # diagnosis summary
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor("#0f4c81"))
    c.drawString(50, height-250, "AI Diagnosis Summary")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    c.drawString(60, height-275, f"Disease Type       : {disease.upper()}")
    c.drawString(60, height-295, f"Prediction Result  : {prediction}")
    c.drawString(60, height-315, f"Confidence Score   : {confidence:.2f}%")
    c.drawString(60, height-335, f"Severity Level     : {severity}")
    c.drawString(60, height-355, f"Emergency Alert    : {'YES' if emergency else 'NO'}")

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor("#0f4c81"))
    c.drawString(50, height-395, "AI Interpretation")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)

    interpretation = [
        f"The uploaded scan image was processed using a deep learning {disease} detection model.",
        f"The AI system identified the scan as {prediction} with a high confidence score of {confidence:.2f}%.",
        f"Severity analysis indicates a {severity.lower()} level condition based on learned pathological patterns.",
        "This report should be used as an AI-assisted screening document and clinically verified."
    ]

    y = height - 420
    for line in interpretation:
        c.drawString(60, y, line)
        y -= 18

    # footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(150, 25, "AI Healthcare Assistant • Generated Diagnostic Summary")

    # =========================
    # PAGE 2 : IMAGES
    # =========================
    c.showPage()

    c.setFillColor(colors.HexColor("#0f4c81"))
    c.rect(0, height-60, width, 60, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, height-38, "MEDICAL IMAGE ANALYSIS")

    c.setFillColor(colors.black)

    original_img_path = os.path.join(UPLOAD_FOLDER, filename)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height-100, "Uploaded Scan Image")

    if os.path.exists(original_img_path):
        c.drawImage(original_img_path, 50, height-380, width=220, height=220, preserveAspectRatio=True)

    c.drawString(330, height-100, "Grad-CAM Heatmap")

    if heatmap_path:
        heatmap_full = os.path.join(BASE_DIR, "static", heatmap_path)
        if os.path.exists(heatmap_full):
            c.drawImage(heatmap_full, 300, height-380, width=220, height=220, preserveAspectRatio=True)

    c.setFont("Helvetica", 10)
    c.drawString(60, height-420, "Heatmap Explanation:")
    c.drawString(70, height-438, "Red/Yellow highlighted regions show where the AI model focused most strongly")
    c.drawString(70, height-454, "while identifying suspicious pathological abnormalities in the scan.")

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(180, 25, "AI Healthcare Assistant • Explainable Medical Imaging")

    # =========================
    # PAGE 3 : RECOMMENDATION
    # =========================
    c.showPage()

    c.setFillColor(colors.HexColor("#0f4c81"))
    c.rect(0, height-60, width, 60, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, height-38, "MEDICAL RECOMMENDATIONS")

    c.setFillColor(colors.black)

    specialist = DOCTOR_MAP.get(disease, "General Physician")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height-100, f"Recommended Specialist : {specialist}")

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor("#0f4c81"))
    c.drawString(60, height-140, "Suggested Next Steps")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)

    steps = [
        "1. Visit the recommended specialist doctor with this AI report.",
        "2. Carry original scan/X-ray for further radiological review.",
        "3. Follow laboratory tests and physician advice if symptoms persist.",
        "4. In severe or emergency-positive cases, immediate hospital admission is recommended."
    ]

    y = height - 170
    for s in steps:
        c.drawString(70, y, s)
        y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.red)
    c.drawString(60, height-300, "AI Disclaimer")

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(70, height-325, "This document is generated by an AI-based healthcare assistant.")
    c.drawString(70, height-340, "It is intended for preliminary screening support only.")
    c.drawString(70, height-355, "Final diagnosis must be confirmed by a licensed medical professional.")

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(180, 25, "AI Healthcare Assistant • Clinical Recommendation Summary")

    c.save()
    return f"reports/{report_name}"

# ----------------------------
# Grad-CAM for Pneumonia only
# ----------------------------
BASE_MODEL_NAME = "mobilenetv2_1.00_224"
LAST_CONV_LAYER = "Conv_1"


def make_gradcam_heatmap(img_array, model):

    # Use LAST CONV layer directly
    last_conv_layer = model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def save_gradcam_overlay(original_path, heatmap, output_name):
    if heatmap is None:
        return None

    img = cv2.imread(original_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    save_path = os.path.join(HEATMAPS_FOLDER, output_name)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return f"heatmaps/{output_name}"

def find_nearby_hospitals(city="Bangalore", disease=None):
    doctor_map = {
        "pneumonia": "Pulmonologist",
        "tb": "Chest Specialist",
        "brain": "Neurologist",
        "skin": "Dermatologist"
    }

    doctor = doctor_map.get(disease, "General Physician")

    hospital_images = [
        "https://images.unsplash.com/photo-1586773860418-d37222d8fce3?w=1200",
        "https://images.unsplash.com/photo-1538108149393-fbbd81895907?w=1200",
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200",
        "https://images.unsplash.com/photo-1580281657527-47b7aa4c1b52?w=1200",
        "https://images.unsplash.com/photo-1504439468489-c8920d796a29?w=1200",
        "https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=1200",
        "https://images.unsplash.com/photo-1512678080530-7760d81faba6?w=1200",
        "https://images.unsplash.com/photo-1579154204601-01588f351e67?w=1200",
        "https://images.unsplash.com/photo-1526256262350-7da7584cf5eb?w=1200",
        "https://images.unsplash.com/photo-1516549655169-df83a0774514?w=1200"
    ]

    hospitals = [
        {
            "name": "Apollo Hospital",
            "location": "Bannerghatta Road, Bangalore",
            "map": "https://www.google.com/maps?q=Apollo+Hospital+Bannerghatta+Road+Bangalore",
            "image": "apollo.jpeg",  # Matches your file exactly
            "doctor": doctor,
            "rating": "4.8 ⭐",
            "contact": "+91 9876543210",
            "slots": "Available Today"
        },
        {
            "name": "Fortis Hospital",
            "location": "Nagarbhavi, Bangalore",
            "map": "https://www.google.com/maps?q=Fortis+Hospital+Bangalore",
            "image": "fortis.jpeg",
            "doctor": doctor,
            "rating": "4.7 ⭐",
            "contact": "+91 9876543211",
            "slots": "Available Today"
        },
        {
            "name": "Manipal Hospital",
            "location": "Old Airport Road, Bangalore",
            "map": "https://www.google.com/maps?q=Manipal+Hospital+Bangalore",
            "image": "manipal.jpeg",
            "doctor": doctor,
            "rating": "4.9 ⭐",
            "contact": "+91 9876543212",
            "slots": "Next Slot 4 PM"
        },
        {
            "name": "Narayana Health City",
            "location": "Bommasandra, Bangalore",
            "map": "https://www.google.com/maps?q=Narayana+Health+City+Bangalore",
            "image": "narayana.jpeg",
            "doctor": doctor,
            "rating": "4.8 ⭐",
            "contact": "+91 9876543213",
            "slots": "Available Tomorrow"
        },
        {
            "name": "Aster CMI Hospital",
            "location": "Hebbal, Bangalore",
            "map": "https://www.google.com/maps?q=Aster+CMI+Hospital+Bangalore",
            "image": "aster.jpeg",
            "doctor": doctor,
            "rating": "4.6 ⭐",
            "contact": "+91 9876543214",
            "slots": "Available Today"
        },
        {
            "name": "Sakra World Hospital",
            "location": "Marathahalli, Bangalore",
            "map": "https://www.google.com/maps?q=Sakra+World+Hospital+Bangalore",
            "image": "sakra.jpeg",
            "doctor": doctor,
            "rating": "4.7 ⭐",
            "contact": "+91 9876543215",
            "slots": "Next Slot 6 PM"
        },
        {
            "name": "Columbia Asia Hospital",
            "location": "Yeshwanthpur, Bangalore",
            "map": "https://www.google.com/maps?q=Columbia+Asia+Hospital+Bangalore",
            "image": "columbia.jpeg",
            "doctor": doctor,
            "rating": "4.5 ⭐",
            "contact": "+91 9876543216",
            "slots": "Available Today"
        },
        {
            "name": "Kauvery Hospital",
            "location": "Electronic City, Bangalore",
            "map": "https://www.google.com/maps?q=Kauvery+Hospital+Bangalore",
            "image": "kauvery.jpeg",
            "doctor": doctor,
            "rating": "4.6 ⭐",
            "contact": "+91 9876543217",
            "slots": "Available Tomorrow"
        },
        {
            "name": "Rainbow Children's Hospital",
            "location": "Bannerghatta Road, Bangalore",
            "map": "https://www.google.com/maps?q=Rainbow+Hospital+Bangalore",
            "image": "rainbow.jpeg",
            "doctor": doctor,
            "rating": "4.8 ⭐",
            "contact": "+91 9876543218",
            "slots": "Available Today"
        },
        {
            "name": "Bangalore Baptist Hospital",
            "location": "Hebbal, Bangalore",
            "map": "https://www.google.com/maps?q=Bangalore+Baptist+Hospital",
            "image": "baptist.jpeg",
            "doctor": doctor,
            "rating": "4.5 ⭐",
            "contact": "+91 9876543219",
            "slots": "Next Slot 5 PM"
        }
    ]
    return hospitals
# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    if "user_id" not in session:
        return redirect("/login")

    return render_template("upload.html", user_name=session.get("user_name"))
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400

    disease = request.form.get("disease")
    file = request.files["image"]

    patient_name = request.form.get("patient_name")
    patient_id = request.form.get("patient_id")
    age = request.form.get("age")
    gender = request.form.get("gender")

    if not patient_name or not patient_id or not age or not gender:
        return "Patient details missing", 400

    try:
        age = int(age)
    except:
        age = 0

    if not disease:
        return "Disease not selected", 400
    if file.filename == "":
        return "No selected file", 400

    # save upload
    safe_name = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)

    # preprocess (same for all)
    img = image.load_img(save_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # pick model
    model_map = {
    "pneumonia": pneumonia_model,
    "tb": tb_model,
    "brain": brain_model,
    "skin": skin_model,
    "bone": bone_model,
    "lung_cancer": lung_model,
    "malaria": malaria_model
}

    model = model_map.get(disease)

    if model is None:
        return f"Model not loaded for {disease}", 500


# =========================
# LUNG CANCER PREDICTION
# =========================
    if disease == "lung_cancer":

        img = cv2.imread(save_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = lung_model.predict(img)

        class_index = np.argmax(prediction)

        classes = [
        "🫁 Adenocarcinoma",
        "🫁 Large Cell Carcinoma",
        "✅ Normal Lung",
        "🫁 Squamous Cell Carcinoma"
    ]

        prediction_label = classes[class_index]

        confidence = float(np.max(prediction)) * 100

        severity = get_severity(confidence)

        emergency = True if confidence > 90 else False


# =========================
# OTHER DISEASES
# =========================
    else:

       pred = float(model.predict(img_array, verbose=0)[0][0])

       if pred >= 0.5:
            prediction_label = "POSITIVE"
            confidence = pred * 100
       else:
            prediction_label = "NEGATIVE"
            confidence = (1 - pred) * 100

       label_map = {
        "pneumonia": ("PNEUMONIA", "NORMAL"),
        "tb": ("TUBERCULOSIS", "NORMAL"),
        "brain": ("TUMOR", "NO TUMOR"),
        "skin": ("MALIGNANT", "BENIGN"),
        "bone": ("FRACTURE", "NORMAL"),
        "malaria": ("MALARIA INFECTED", "NORMAL"),
    }

       pos, neg = label_map[disease]

       prediction_label = pos if pred >= 0.5 else neg

       severity = get_severity(confidence)

       emergency = is_emergency(confidence) if prediction_label == pos else False

    # ----------------------------
    # Grad-CAM Heatmap (Pneumonia)
    # ----------------------------
    heatmap_rel_path = None

    if disease == "pneumonia":
        try:
            heatmap = make_gradcam_heatmap(img_array, model)
            heatmap_name = f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            heatmap_rel_path = save_gradcam_overlay(save_path, heatmap, heatmap_name)
            print("🔥 Heatmap saved:", heatmap_rel_path)
        except Exception as e:
            print("❌ Heatmap error:", e)
            heatmap_rel_path = None

    # pdf
    report_rel_path = generate_pdf_report(
    safe_name,
    disease,
    prediction_label,
    confidence,
    severity,
    emergency,
    heatmap_rel_path,
    patient_name,
    patient_id,
    age,
    gender
)

    # save in DB (IMPORTANT: save disease too)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
INSERT INTO predictions (
    filename, disease, prediction, confidence, severity, emergency,
    report_path, heatmap_path,
    patient_name, patient_id, age, gender, user_id
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    safe_name, disease, prediction_label, float(confidence), severity, int(emergency),
    report_rel_path, heatmap_rel_path,
    patient_name, patient_id, age, gender, session.get("user_id")
))
    conn.commit()
    conn.close()

    image_url = url_for("static", filename=f"uploads/{safe_name}")

    return render_template(
        "result.html",
        disease=disease,
        prediction=prediction_label,
        confidence=f"{confidence:.2f}",
        severity=severity,
        emergency=emergency,
        image_url=image_url,
        report_path=report_rel_path,
        heatmap_path=heatmap_rel_path
    )

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")
    patient_id = request.args.get("patient_id", "").strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if patient_id:
        cursor.execute("""
            SELECT id, patient_name, patient_id, age, gender,
                   filename, disease, prediction, confidence,
                   severity, emergency, created_at, report_path, heatmap_path
            FROM predictions
            WHERE user_id=? AND patient_id LIKE ?
            ORDER BY id DESC
        """, (session.get("user_id"), f"%{patient_id}%"))
    else:
        cursor.execute("""
            SELECT id, patient_name, patient_id, age, gender,
                   filename, disease, prediction, confidence,
                   severity, emergency, created_at, report_path, heatmap_path
            FROM predictions
            WHERE user_id=?
            ORDER BY id DESC
        """, (session.get("user_id"),))

    rows = cursor.fetchall()
    conn.close()

    return render_template("history.html", rows=rows, patient_id=patient_id, user_name=session.get("user_name"))




@app.route("/risk_summary")
def risk_summary():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (session["user_id"],))
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND emergency=1", (session["user_id"],))
    emergency_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND heatmap_path IS NOT NULL AND heatmap_path != ''", (session["user_id"],))
    heatmap_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE user_id=?", (session["user_id"],))
    avg_conf = cursor.fetchone()[0] or 0
    avg_conf = round(avg_conf, 2)

    cursor.execute("""
        SELECT disease, COUNT(*) FROM predictions
        WHERE user_id=?
        GROUP BY disease
        ORDER BY COUNT(*) DESC
    """, (session["user_id"],))
    disease_data = cursor.fetchall()
    disease_labels = [x[0].upper() for x in disease_data]
    disease_counts = [x[1] for x in disease_data]

    cursor.execute("""
        SELECT severity, COUNT(*) FROM predictions
        WHERE user_id=?
        GROUP BY severity
        ORDER BY COUNT(*) DESC
    """, (session["user_id"],))
    severity_data = cursor.fetchall()
    severity_labels = [x[0] for x in severity_data]
    severity_counts = [x[1] for x in severity_data]

    cursor.execute("""
        SELECT DATE(created_at), COUNT(*) FROM predictions
        WHERE user_id=?
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at) ASC
    """, (session["user_id"],))
    trend_data = cursor.fetchall()
    trend_labels = [x[0] for x in trend_data]
    trend_counts = [x[1] for x in trend_data]

    cursor.execute("""
        SELECT DATE(created_at), COUNT(*) FROM predictions
        WHERE user_id=? AND heatmap_path IS NOT NULL AND heatmap_path != ''
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at) ASC
    """, (session["user_id"],))
    heatmap_trend_data = cursor.fetchall()
    heatmap_trend_labels = [x[0] for x in heatmap_trend_data]
    heatmap_trend_counts = [x[1] for x in heatmap_trend_data]

    conn.close()

    return render_template(
        "risk_summary.html",
        total=total,
        emergency_count=emergency_count,
        avg_conf=avg_conf,
        heatmap_count=heatmap_count,
        disease_labels=disease_labels,
        disease_counts=disease_counts,
        severity_labels=severity_labels,
        severity_counts=severity_counts,
        trend_labels=trend_labels,
        trend_counts=trend_counts,
        heatmap_trend_labels=heatmap_trend_labels,
        heatmap_trend_counts=heatmap_trend_counts
    )

@app.route("/ai_doctor")
def ai_doctor():
    if "user_id" not in session:
        return redirect("/login")
    disease = request.args.get("disease", "")
    prediction = request.args.get("prediction", "")
    confidence = request.args.get("confidence", "")
    severity = request.args.get("severity", "")
    emergency = request.args.get("emergency", "")

    return render_template(
        "ai_doctor.html",
        disease=disease,
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        emergency=emergency,
        user_name=session.get("user_name")
    )

import re
from flask import request, jsonify

@app.route("/ollama_chat", methods=["POST"])
def ollama_chat():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()

    disease = data.get("disease", "")
    prediction = data.get("prediction", "")
    confidence = data.get("confidence", "")
    severity = data.get("severity", "")
    emergency = data.get("emergency", "")

    if not user_message:
        return jsonify({"reply": "Please enter a medical question."})

    context_block = ""
    if disease or prediction or confidence or severity or emergency:
        context_block = f"""
Current diagnosis context:
- Disease: {disease}
- Prediction: {prediction}
- Confidence: {confidence}
- Severity: {severity}
- Emergency: {emergency}
"""

    system_prompt = f"""
You are an advanced AI healthcare voice doctor assistant inside a smart medical diagnosis platform.

Your job:
- Talk naturally like a caring doctor.
- Give practical medical guidance in simple English.
- Always explain according to the current diagnosis context.
- If condition is severe/emergency, clearly warn user.
- If user asks for doctor/hospital/treatment/where to go, encourage specialist consultation.

Current Patient Diagnosis:
Disease: {disease}
Prediction: {prediction}
Confidence: {confidence}
Severity: {severity}
Emergency: {emergency}

Doctor Specialist Mapping:
pneumonia -> Pulmonologist
tb -> Chest Specialist
brain -> Neurologist
skin -> Dermatologist

Rules:
- reply in 4 to 6 lines only
- sound human and supportive
- never say "I am just AI"
- never refuse hospital suggestions
- if user asks anything about treatment/hospital/doctor, mention specialist type
"""
    hospital_keywords = [
    "hospital", "doctor", "clinic", "nearby", "treatment",
    "where should i go", "which hospital", "specialist", "admit",
    "emergency", "consultation"
]

    show_hospitals = any(word in user_message.lower() for word in hospital_keywords)
    try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        reply = response["message"]["content"]
        return jsonify({
    "reply": reply,
    "show_hospitals": show_hospitals
})
    except Exception as e:
        print("Ollama error:", e)
        return jsonify({"reply": "AI Doctor is unavailable right now. Please make sure Ollama is running."})



@app.route("/explain_result", methods=["POST"])
def explain_result():
    data = request.get_json() or {}

    disease = data.get("disease", "")
    prediction = data.get("prediction", "")
    confidence = data.get("confidence", "")
    severity = data.get("severity", "")
    emergency = data.get("emergency", False)

    prompt = f"""
    You are an AI medical assistant.
    Explain this diagnosis result in simple, patient-friendly language.

    Disease Type: {disease}
    Prediction: {prediction}
    Confidence: {confidence}
    Severity: {severity}
    Emergency: {emergency}

    Rules:
    - Keep answer short and clear
    - Do not give final diagnosis
    - If emergency is true, strongly advise immediate doctor consultation
    - Mention that this is AI-assisted only
    - Always encourage seeing a specialist doctor for accurate diagnosis and treatment
    - Never say "I am just an AI". Always sound supportive and caring.
    - If confidence is low, mention that AI is uncertain and doctor consultation is important.
    - If severity is moderate or severe, clearly explain the risks and why specialist care is recommended.
    - If user asks about treatment, always say that treatment depends on doctor's evaluation and tests, and encourage specialist consultation.
    - If user asks about hospitals, always suggest seeing a specialist doctor for accurate diagnosis and treatment, and mention the specialist type based on disease.
    - Never refuse to give guidance on where to go or what to do next. Always encourage seeing a doctor, especially if severity is moderate/severe or emergency is true.
    - Always explain that this AI result is a preliminary screening and not a final diagnosis, and that clinical verification by a licensed medical professional is essential for accurate diagnosis and treatment.
    - Always sound empathetic, supportive, and encouraging towards the patient, regardless of the diagnosis result.
    - If the prediction is positive and confidence is high, explain that the AI model has identified patterns in the scan that are strongly associated with the disease, but that this should be confirmed by a doctor through further tests and evaluation.
    - If the prediction is negative but confidence is low, explain that the AI model is uncertain and that it is still important to consult a doctor if symptoms persist or worsen.
    - dont use * symbols in the answer, just plain text. Avoid markdown formatting.
    - dont use astrits or any special characters in the answer. Just simple text.
    
    """

    try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": "You are a safe medical AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = response["message"]["content"]
        return jsonify({"reply": reply})

    except Exception as e:
        print("Explain result error:", e)
        return jsonify({"reply": "Unable to generate explanation right now."})

@app.route("/nearby_hospitals")
def nearby_hospitals():
    city = request.args.get("city", "Bangalore")
    disease = request.args.get("disease", "")
    hospitals = find_nearby_hospitals(city, disease)
    return jsonify({"hospitals": hospitals})

@app.route("/hospital_directory")
def hospital_directory():
    if "user_id" not in session:
        return redirect("/login")

    disease = request.args.get("disease", "general")

    hospitals = find_nearby_hospitals("Bangalore", disease)

    return render_template(
        "hospital_directory.html",
        hospitals=hospitals,
        disease=disease,
        user_name=session.get("user_name")
    )

@app.route("/book_appointment", methods=["GET", "POST"])
def book_appointment():
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":
        hospital_name = request.form.get("hospital_name")
        doctor_type = request.form.get("doctor_type")
        appointment_date = request.form.get("appointment_date")
        appointment_time = request.form.get("appointment_time")
        location = request.form.get("location")
        map_link = request.form.get("map_link")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO appointments (
                user_id, patient_name, hospital_name, doctor_type,
                appointment_date, appointment_time, location, map_link
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            session["user_name"],
            hospital_name,
            doctor_type,
            appointment_date,
            appointment_time,
            location,
            map_link
        ))

        conn.commit()
        conn.close()

        return redirect("/my_appointments")

    return render_template(
        "book_appointment.html",
        hospital_name=request.args.get("hospital"),
        doctor_type=request.args.get("doctor"),
        location=request.args.get("location"),
        map_link=request.args.get("map")
    )

@app.route("/my_appointments")
def my_appointments():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, hospital_name, doctor_type, appointment_date,
       appointment_time, location, map_link, status, created_at
FROM appointments
WHERE user_id=?
ORDER BY id DESC
    """, (session["user_id"],))

    appointments = cur.fetchall()
    conn.close()

    return render_template(
        "my_appointments.html",
        appointments=appointments,
        user_name=session.get("user_name")
    )

@app.route("/cancel_appointment/<int:aid>")
def cancel_appointment(aid):
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("UPDATE appointments SET status='Cancelled' WHERE id=? AND user_id=?", (aid, session["user_id"]))
    conn.commit()
    conn.close()

    return redirect("/my_appointments")

import os
from flask import send_file, session, redirect
import sqlite3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

@app.route("/appointment_slip/<int:appointment_id>")
def appointment_slip(appointment_id):
    if "user_id" not in session:
        return redirect("/login")
        
    # 1. Fetch appointment details from database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT hospital_name, doctor_type, appointment_date, appointment_time, location, status 
        FROM appointments 
        WHERE id = ? AND user_id = ?
    """, (appointment_id, session["user_id"]))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return "Appointment not found", 404
        
    hospital_name, doctor_type, appt_date, appt_time, location, status = row

    # 2. Map the hospital name to its corresponding filename
    # Make sure these filenames match what you have inside your static/images/hospitals/ folder!
    HOSPITAL_IMAGE_MAP = {
        "Apollo Hospital": "apollo.jpeg",
        "Fortis Hospital": "fortis.jpeg",
        "Manipal Hospital": "manipal.jpeg",
        "Narayana Health City": "narayana.jpeg",
        "Aster CMI Hospital": "aster.jpeg",
        "Sakra World Hospital": "sakra.jpeg",
        "Columbia Asia Hospital": "columbia.jpeg",
        "Kauvery Hospital": "kauvery.jpeg",
        "Rainbow Children's Hospital": "rainbow.jpeg",
        "Bangalore Baptist Hospital": "baptist.jpeg"
    }
    
    image_filename = HOSPITAL_IMAGE_MAP.get(hospital_name)
    
    # Define PDF creation paths
    pdf_filename = f"appointment_{appointment_id}.pdf"
    pdf_path = os.path.join(BASE_DIR, "static", "reports", pdf_filename)
    
    # 3. Generate structured PDF using ReportLab
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    
    # Top Aesthetic Brand Header Band
    c.setFillColor(colors.HexColor("#0f4c81"))
    c.rect(0, height - 80, width, 80, fill=1)
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 48, "AI HEALTHCARE APPOINTMENT SLIP")
    
    # Handle Image Insertion if mapping exists and file is valid
    current_y = height - 130
    if image_filename:
        img_path = os.path.join(BASE_DIR, "static", "images", "hospitals", image_filename)
        if os.path.exists(img_path):
            # Center the image horizontally on the A4 page (width = 595.27 points)
            img_width = 240
            img_height = 140
            img_x = (width - img_width) / 2
            
            c.drawImage(img_path, img_x, height - 260, width=img_width, height=img_height, preserveAspectRatio=True)
            current_y = height - 290  # Push typography values down below the photo box
            
    # Appointment Data Layout Text Blocks
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    
    details = [
        f"Hospital: {hospital_name}",
        f"Doctor Type: {doctor_type}",
        f"Date: {appt_date}",
        f"Time: {appt_time}",
        f"Location: {location}",
        f"Status: {status}"
    ]
    
    for item in details:
        c.drawString(100, current_y, item)
        current_y -= 25
        
    # Document Verification Footer Notice
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawCentredString(width / 2, 40, "Please display this digital summary or print version at the desk counter upon arrival.")
    
    c.save()
    return send_file(pdf_path, as_attachment=False)
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")   # ← THIS WAS MISSING
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        try:
            cur.execute(
                "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                (name, email, hashed_password, "patient")
            )
            conn.commit()
            conn.close()
            return redirect("/login")
        except:
            conn.close()
            return "Email already exists"

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("SELECT id, name, password, role FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
    
            session["user_id"] = user[0]
            session["user_name"] = user[1]

            return redirect("/home")
        return "Invalid login credentials"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")




@app.route("/home")
def upload_page():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (session["user_id"],))
    total_scans = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND emergency=1", (session["user_id"],))
    emergency_cases = cur.fetchone()[0]

    cur.execute("""
        SELECT disease, prediction, confidence, created_at
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC LIMIT 5
    """, (session["user_id"],))
    recent_reports = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM appointments WHERE user_id=? AND status='Booked'", (session["user_id"],))
    appointment_count = cur.fetchone()[0]
    conn.close()

    return render_template(
        "home.html",
        user_name=session["user_name"],
        total_scans=total_scans,
        emergency_cases=emergency_cases,
        recent_reports=recent_reports,
        appointment_count=appointment_count
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)