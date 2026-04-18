"""dashboard/app.py — Built in Phase 9."""
from flask import Flask, jsonify


def create_app():
    app = Flask(__name__)

    @app.route("/api/status")
    def status():
        from config import settings
        return jsonify({
            "status": "running",
            "mode": "paper" if settings.paper_trade else "live",
            "broker": settings.broker,
            "strategy": settings.strategy,
        })

    return app
