import os
from app import interface

port = int(os.environ.get("PORT", 7860))
app = interface.app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
