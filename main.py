# This is the entrypoint for the modular backend.
# If you add new routes or blueprints, restart the backend to apply changes.
from app import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)