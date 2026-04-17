"""dashboard/app.py — Built in Phase 9."""
from flask import Flask, jsonify
def create_app():
    app = Flask(__name__)
    @app.route("/api/status")
    def status():
        return jsonify({"status": "running", "mode": "paper"})
    return app
