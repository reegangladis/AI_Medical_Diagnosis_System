import os
import logging
from app import app

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Database is initialized at app module level (init_db() is called in app.py).
# No need to call it again here, which would cause duplicate table-creation attempts.

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
