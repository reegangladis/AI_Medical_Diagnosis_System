from app import app, init_db

# Initialize database on startup
init_db()

if __name__ == "__main__":
    app.run()
