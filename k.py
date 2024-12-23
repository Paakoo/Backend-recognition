from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import datetime

app = Flask(__name__)

# Secret key untuk mengenkripsi token
app.config['JWT_SECRET_KEY'] = 'your_secret_key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)  # Token kedaluwarsa dalam 1 jam

jwt = JWTManager(app)

# Dummy data untuk pengguna
users = {
    "user@example.com": {
        "password": "password123",
        "id": 1
    },
    "kntl@gmail.com": {
        "password": "123",
        "id": 2
    }
}

# Route untuk login dan menghasilkan token
@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email', None)
    password = request.json.get('password', None)

    if email not in users or users[email]['password'] != password:
        return jsonify({"message": "Invalid credentials"}), 401

    access_token = create_access_token(identity=email)
    return jsonify(access_token=access_token), 200

# Protected route yang memerlukan token
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.11')






return jsonify({
                                'status': 'success',
                                'message': 'Face matched with authenticated user',
                                'data': {
                                    'matched_name': matched_name,
                                    'confidence': float(similarity),
                                    'filename': filename
                                },
                                'timing': {
                                    'spoofing': f"{spoof_time:.3f}s",
                                    'detection': f"{detection_time:.3f}s",
                                    'processing': f"{process_time:.3f}s",
                                    'embedding': f"{embedding_time:.3f}s",
                                    'matching': f"{matching_time:.3f}s",
                                    'total': f"{time.time() - start_time:.3f}s"
                                }
                            }), 200