import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# -------------------------------
# Add disease column
# -------------------------------
try:
    cur.execute("ALTER TABLE predictions ADD COLUMN disease TEXT")
    print("✅ Added column: disease")
except Exception:
    print("⚠️ disease column already exists")

# -------------------------------
# Add heatmap_path column
# -------------------------------
try:
    cur.execute("ALTER TABLE predictions ADD COLUMN heatmap_path TEXT")
    print("✅ Added column: heatmap_path")
except Exception:
    print("⚠️ heatmap_path column already exists")

conn.commit()
conn.close()

print("✅ DB update completed")