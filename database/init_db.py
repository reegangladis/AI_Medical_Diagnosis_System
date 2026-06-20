import sqlite3

conn = sqlite3.connect("database/app.db")
cursor = conn.cursor()

cursor.execute("""
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
        heatmap_path TEXT
    )
""")

conn.commit()
conn.close()

print("Database initialized.")
