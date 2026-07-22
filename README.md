<<<<<<< HEAD
# 🏥 AI-Based Healthcare Diagnosis System & Medical Imaging Platform

## 📌 Overview

The AI-Based Healthcare Diagnosis System & Medical Imaging Platform is an intelligent healthcare solution that leverages Artificial Intelligence (AI), Deep Learning, and Medical Image Analysis to assist in the early detection of diseases.

The platform allows users to upload medical images and receive AI-generated diagnostic predictions, confidence scores, severity analysis, and automated medical reports. It also provides specialist recommendations and healthcare decision-support features.

---

## 🚀 Features

### 🔍 Disease Detection

The system can detect and analyze:

* Pneumonia
* Tuberculosis (TB)
* Brain Tumor
* Skin Cancer
* Bone Fracture
* Lung Cancer
* Malaria

### 🧠 AI-Powered Analysis

* Deep Learning-based disease prediction
* Medical image preprocessing
* Confidence score generation
* Severity assessment
* Emergency alert detection

### 📊 Explainable AI

* Grad-CAM Heatmap Visualization
* Highlights affected regions in medical images
* Improves transparency of AI predictions

### 📄 Automated Report Generation

* PDF diagnostic reports
* Patient information summary
* Prediction results
* Confidence scores
* Medical recommendations

### 👨‍⚕️ Healthcare Support

* Specialist doctor recommendations
* Hospital suggestions
* Patient history tracking
* Appointment management

### 🔐 User Management

* Secure authentication
* User registration and login
* Patient record management
* Prediction history dashboard

---

## 🛠️ Technologies Used

### Programming Languages

* Python

### AI & Machine Learning

* TensorFlow
* Keras
* Deep Learning
* Machine Learning

### Computer Vision

* OpenCV
* Medical Image Processing

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Database

* SQLite

### Libraries

* NumPy
* PIL (Pillow)
* ReportLab
* Requests

---

## 🏗️ System Architecture

1. User uploads a medical image.
2. Image preprocessing is performed.
3. Deep Learning model analyzes the image.
4. Disease prediction is generated.
5. Confidence score and severity are calculated.
6. Grad-CAM heatmap is created.
7. PDF report is generated.
8. Results are stored in the database.
9. Doctor and hospital recommendations are provided.

---

## 📂 Project Structure

AI-Based-Healthcare-Diagnosis-System/

├── app.py

├── models/

├── static/

│ ├── uploads/

│ ├── reports/

│ └── heatmaps/

├── templates/

├── database/

├── requirements.txt

└── README.md

---

## 📸 Screenshots

### Login Page

<img width="612" height="872" alt="image" src="https://github.com/user-attachments/assets/8be4af3b-76d7-41da-a343-356cd18b266e" />


### Disease Selection Dashboard

<img width="1661" height="901" alt="image" src="https://github.com/user-attachments/assets/12149038-cffd-49c5-a410-e03889520dc2" />


### Medical Image Upload

<img width="1877" height="891" alt="image" src="https://github.com/user-attachments/assets/bdc1d696-e79e-4ca1-b6da-0190e40deded" />


### Prediction Results

<img width="1830" height="901" alt="image" src="https://github.com/user-attachments/assets/c89eb6ef-bfc9-4ebe-b832-177c6dfa8772" />


### Grad-CAM Heatmap Visualization

<img width="542" height="662" alt="image" src="https://github.com/user-attachments/assets/06f4b690-0762-4a5e-a0e3-f0f5b30b7106" />


### Generated Medical Report

<img width="1006" height="827" alt="image" src="https://github.com/user-attachments/assets/162c27a4-ae0d-4634-8aa7-e95722275ca2" />
<img width="992" height="800" alt="image" src="https://github.com/user-attachments/assets/06d0f5fd-2fcc-446a-9f67-89e1f05ef3d0" />

### AI Clinical Assistant

<img width="1742" height="892" alt="ai docort" src="https://github.com/user-attachments/assets/2fe4f68b-5ba0-418d-b813-1af0996966bb" />

### Prediction History Dashboard

<img width="1816" height="901" alt="image" src="https://github.com/user-attachments/assets/20302b1b-7f51-4063-b807-27098cbec6b7" />

### Hospital Recommendation Page

<img width="1797" height="897" alt="hos ital" src="https://github.com/user-attachments/assets/109bc146-5988-4cda-a6b6-42d7529ac0bb" />


### Appointment Booking Page

<img width="1747" height="876" alt="appontement" src="https://github.com/user-attachments/assets/b7e2874f-02a7-4a90-b52a-dffb59258b9d" />

---

## 🎯 Project Outcomes

* Improved healthcare accessibility through AI-assisted diagnosis.
* Automated disease detection workflow.
* Enhanced medical image interpretation.
* Generated explainable AI visualizations.
* Streamlined healthcare reporting process.

---

## 🔮 Future Enhancements

* Real-time doctor consultation
* Cloud deployment
* Mobile application support
* Electronic Health Record (EHR) integration
* Multi-language support
* Advanced disease prediction models

---

## 👨‍💻 Author

### Reegan Gladis P

BCA Student | AI & Data Science Enthusiast | Machine Learning | Python Developer

📧 Email: reegangladis@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/reegan2806

🔗 GitHub: https://github.com/reegangladis

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub and connect with me on LinkedIn.
=======
# Medical AI Diagnosis System (MediAI Suite)

Medical AI Diagnosis System is a deep learning-based clinical intelligence platform. It analyzes medical scan images to screen for key conditions, generates clinical reports, offers diagnostic explanations, and allows patients to book specialist appointments.

---

## 🌟 Key Features

1. **Deep Learning Diagnosis**: Detects 7 major pathologies from scan uploads:
   * **Pneumonia** (X-ray scan)
   * **Tuberculosis** (X-ray scan)
   * **Brain Tumor** (MRI scan)
   * **Skin Cancer** (Dermatoscopic image)
   * **Bone Fracture** (X-ray scan)
   * **Lung Cancer** (Categorical: Adenocarcinoma, Large Cell, Squamous, or Normal)
   * **Malaria** (Blood smear cell slide)
2. **Explainable AI (Grad-CAM)**: Generates a colorized attention heatmap overlay showing exactly where the neural network focused to highlight anomalies.
3. **Medical PDF Reports**: Automatically generates download-ready clinical PDF reports with patient info, AI interpretation, and Grad-CAM scans.
4. **Dual LLM Voice Doctor Chatbot**:
   * Powered primarily by the **Gemini API** (`gemini-1.5-flash`) for cloud hosting speed and reliability.
   * Graceful fallback to local **Ollama** (`gemma:2b`) or rule-based medical dialogue engines if keys are not provided.
5. **Hospital Directory & Appointment Booking**: Recommends local specialist doctors (e.g. Pulmonologist, Neurologist) and provides booking slips with interactive maps.
6. **Graceful Simulation/Demo Mode**: Since model files (`.h5`) are large and gitignored, the app automatically runs in a fully-functional simulation mode if model files are missing on live servers (e.g., Render/Heroku).

---

## 💻 Tech Stack

* **Backend**: Flask, TensorFlow/Keras, OpenCV (CV2), sqlite3, ReportLab, python-dotenv
* **Frontend**: HTML5 (Semantic Structure), Vanilla CSS (Responsive Glassmorphism & Mesh Backgrounds), JavaScript (Number counters, mobile nav toggles, buttons ripple effects)
* **Hosting Configuration**: WSGI, Gunicorn, Procfile

---

## 🚀 Local Installation & Quickstart

### Prerequisites
* Python 3.9+
* pip

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <your-repository-url>
   cd medical-ai-diagnosis
   ```

2. **Initialize Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Create a `.env` file in the root folder:
   ```env
   GOOGLE_MAPS_API_KEY=your_google_maps_key
   GEMINI_API_KEY=your_gemini_api_key
   ```
   *(Note: If no `GEMINI_API_KEY` is provided, the chat feature will fallback to Ollama or rule-based replies).*

5. **Run the Application**:
   ```bash
   python wsgi.py
   ```
   Open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Deployment Configuration

This repository is optimized for deployment to hosts like **Render**, **Heroku**, or private VPS:
* **Procfile**: Declares process types for cloud servers: `web: gunicorn wsgi:app`
* **wsgi.py**: The WSGI entrypoint that automatically instantiates the database schema before starting the app process.
* **Database & Uploads Ignored**: Local SQLite databases (`*.db`) and generated uploads, reports, and heatmaps are ignored in `.gitignore` to keep repositories clean.
>>>>>>> 8a508842 (Updated login page and fixed dashboard)
