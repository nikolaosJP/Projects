import os
from flask import Flask, render_template


def create_app():
    """Application factory function"""
    app = Flask(__name__,
                template_folder='.',
                static_folder='static')

    @app.route("/")
    def index():
        return render_template('index.html')

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("XAI_HOST", "127.0.0.1")
    port = int(os.environ.get("XAI_PORT", "5000"))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"Machine Learning & XAI Platform running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
