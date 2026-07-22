# AI Medical Diagnosis System (MediAI Suite)

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

* **Backend**: Flask, TensorFlow/Keras, OpenCV (CV2), PostgreSQL (production) / SQLite (local fallback), Flask-SQLAlchemy, Flask-Migrate, ReportLab, python-dotenv
* **Frontend**: HTML5 (Semantic Structure), Vanilla CSS (Responsive Glassmorphism & Mesh Backgrounds), JavaScript (Number counters, mobile nav toggles, buttons ripple effects)
* **Hosting Configuration**: WSGI, Gunicorn, Procfile

---

## 🚀 Local Installation & Quickstart

### Prerequisites
* Python 3.9+
* pip
* PostgreSQL (Optional - falls back to SQLite automatically if `DATABASE_URL` is not set)

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
   # API Keys
   GOOGLE_MAPS_API_KEY=your_google_maps_key
   GEMINI_API_KEY=your_gemini_api_key

   # Database URL (Omit to fallback to local SQLite database automatically)
   # Format: postgresql://user:password@host:port/dbname
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mediai_db
   ```

5. **Run Database Migrations (Optional)**:
   ```bash
   flask db upgrade
   ```
   *(Note: The server will automatically run `db.create_all()` on startup to instantiate tables if migrations aren't executed).*

6. **Run the Application**:
   ```bash
   python wsgi.py
   ```
   Open your browser to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Deployment Configuration (Render)

This repository is pre-configured and optimized for deployment to **Render**:

### Render Web Service Setup

1. **Create Web Service**: Connect your GitHub repository to a new Web Service on Render.
2. **Environment**: Choose `Python 3` environment.
3. **Environment Variables**: Define the following under Environment settings:
   * `DATABASE_URL`: Set to the External Database URL from your Render PostgreSQL instance. (Render PostgreSQL URLs automatically configure SSL).
   * `SECRET_KEY`: Set to a secure random string.
   * `GEMINI_API_KEY`: (Optional) Your Gemini AI developer key.
   * `GOOGLE_MAPS_API_KEY`: (Optional) Your Google Maps embed key.
4. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
   *If using Alembic database migrations, you can combine:*
   ```bash
   pip install -r requirements.txt && flask db upgrade
   ```
5. **Start Command**:
   ```bash
   gunicorn wsgi:app
   ```

### Alembic Migrations Commands
If you need to make changes to database schemas, use:
* **Generate a migration**: `flask db migrate -m "migration description"`
* **Apply migrations**: `flask db upgrade`
