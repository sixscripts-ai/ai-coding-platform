from flask import Flask
from flask_cors import CORS
import os

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_folder='../frontend/build', static_url_path='/')
    CORS(app)
    
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DATABASE=os.path.join(app.instance_path, 'ai_platform.sqlite'),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Register blueprints
    from app.api import api_bp
    app.register_blueprint(api_bp)
    
    # Serve React App
    @app.route('/')
    def serve():
        return app.send_static_file('index.html')
    
    @app.errorhandler(404)
    def not_found(e):
        return app.send_static_file('index.html')
    
    return app
