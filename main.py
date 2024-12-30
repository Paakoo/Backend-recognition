from flask import Flask
from routes.face_recognition import face_recognition_bp, configure_jwt, init_jwt
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()

# Initialize Flask
app = Flask(__name__)

configure_jwt(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH'))
CORS(app) 
app.register_blueprint(face_recognition_bp)
init_jwt(app)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST')
    port = int(os.getenv('FLASK_PORT'))
    debug = os.getenv('DEBUG')
    app.run(debug=debug, host=host, port=port)
