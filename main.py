from flask import Flask
from routes.face_recognition import face_recognition_bp, configure_jwt
import os

app = Flask(__name__)

# Configure JWT
configure_jwt(app)

# Register Blueprints
app.register_blueprint(face_recognition_bp)

if __name__ == '__main__':
    # Ensure the JWT secret key is set
    jwt_secret_key = os.getenv('JWT_SECRET_KEY')
    if not jwt_secret_key:
        raise ValueError("JWT_SECRET_KEY environment variable is not set")
    else:
        print(f"JWT_SECRET_KEY is set to: {jwt_secret_key}")
    
    app.run(debug=True)
