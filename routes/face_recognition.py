from flask import Blueprint, request, jsonify, redirect, render_template
from werkzeug.utils import secure_filename
import os
from deepface import DeepFace
import cv2
from mtcnn import MTCNN
from dotenv import load_dotenv
import numpy as np
import h5py
import time
import base64
from scipy.spatial.distance import cosine
import pymysql
from datetime import datetime, timedelta
import json
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from functools import wraps
import jwt







# Load environment variables
load_dotenv()

jwt = JWTManager()

def init_jwt(app):
    jwt.init_app(app)

# Blueprint Initialization
face_recognition_bp = Blueprint('face_recognition_bp', __name__)


# Get configurations from environment
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
BASE_FOLDER = os.getenv('BASE_FOLDER')
FACE_DETECTION_MODEL = os.getenv('FACE_DETECTION_MODEL')
FACE_RECOGNITION_MODEL = os.getenv('FACE_RECOGNITION_MODEL')
FACE_IMAGE_SIZE = int(os.getenv('FACE_IMAGE_SIZE'))
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH')

db_config = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "pass"),
    "database": os.getenv("MYSQL_DB", "tugas_akhir"),
    "cursorclass": pymysql.cursors.DictCursor
}

def load_h5_embeddings():
    try:
        embeddings = {}
        with h5py.File(EMBEDDINGS_PATH, 'r') as hf:
            for username in hf.keys():
                clean_username = username.replace('user_', '')
                embeddings[clean_username] = np.array(hf[username])
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return {}

def compare_embeddings(embedding1, embedding2):
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def find_matching_face(new_embedding, stored_embeddings, threshold=0.75):
    max_similarity = -1
    matched_name = None
    
    for username, user_embeddings in stored_embeddings.items():
        # Handle both single embedding and multiple embeddings per user
        if len(user_embeddings.shape) == 1:
            user_embeddings = np.array([user_embeddings])
            
        for stored_embedding in user_embeddings:
            similarity = compare_embeddings(new_embedding, stored_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_name = username
    
    if max_similarity >= threshold:
        return matched_name, max_similarity
    return None, max_similarity

def save_h5_embeddings(embeddings):
    try:
        with h5py.File(EMBEDDINGS_PATH, 'w') as hf:
            for username, embedding in embeddings.items():
                hf.create_dataset(f"user_{username}", data=embedding)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False
        
def configure_jwt(app):
    """Configure JWT settings"""
    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_TOKEN_LOCATION'] = ['headers']
    app.config['JWT_HEADER_NAME'] = 'Authorization'
    app.config['JWT_HEADER_TYPE'] = 'Bearer'

# API Route to test the API
@face_recognition_bp.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'API is running'})

@face_recognition_bp.route('/capture')
def capture():
    return render_template('captureWajah.html')

def crop_and_save_face(image_data, output_path, filename):
    try:
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Convert image data to cv2 format
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_data

        # Detect faces
        faces = detector.detect_faces(image)

        if faces:
            # Get first face detected
            x, y, width, height = faces[0]['box']
            
            # Add margin
            margin = 20
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            width = width + margin * 2
            height = height + margin * 2

            # Crop and resize
            cropped_face = image[y:y+height, x:x+width]
            cropped_face_resized = cv2.resize(cropped_face, (250, 250))

            # Save cropped face
            output_file = os.path.join(output_path, filename)
            cv2.imwrite(output_file, cropped_face_resized)
            
            return filename
        else:
            # If no face detected, crop center
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            x = max(center_x - 125, 0)
            y = max(center_y - 125, 0)
            
            cropped_face = image[y:y+250, x:x+250]
            cropped_face_resized = cv2.resize(cropped_face, (250, 250))

            # Save cropped image
            output_file = os.path.join(output_path, f'center_crop_{filename}')
            cv2.imwrite(output_file, cropped_face_resized)
            
            return f'center_crop_{filename}'

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Update save_image_route to use face cropping
@face_recognition_bp.route('/api/save_image', methods=['POST'])
def save_image_route():
    data = request.get_json()
    angle = data.get('angle')
    count = data.get('count')
    image_data = data.get('image')
    username = data.get('username')

    user_folder = os.path.join(BASE_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    try:
        filename = f"{username}_{angle}_{count}.jpg"
        processed_filename = crop_and_save_face(image_data, user_folder, filename)
        
        if processed_filename:
            return jsonify({
                'status': 'success',
                'message': f'Image saved as {processed_filename}',
                'filename': processed_filename
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process image'
            }), 500
            
    except Exception as e:
        print(f"Error saving image: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error saving image'
        }), 500
    

def get_incremental_filename(folder, filename):
    """Generate an incremental filename to avoid overwriting existing files."""
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{name}_{counter}{ext}"
    new_file_path = os.path.join(folder, new_filename)

    # Increment filename until a unique one is found
    while os.path.exists(new_file_path):
        counter += 1
        new_filename = f"{name}_{counter}{ext}"
        new_file_path = os.path.join(folder, new_filename)
        
    return new_filename

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def get_current_user():
    try:
        current_user = get_jwt_identity()
        return current_user
    except Exception as e:
        return None
    
@face_recognition_bp.route('/api/upload', methods=['POST'])
@jwt_required()
def upload():
    start_time = time.time()
    
    current_user = get_current_user()
    if not current_user:
        return jsonify({
            'status': 'error',
            'message': 'Could not identify user'
        }), 401
    
    # Get token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({
            'status': 'error',
            'message': 'Missing Authorization header'
        }), 401
    
    # Parse Bearer token
    parts = auth_header.split()
    if parts[0].lower() != 'bearer' or len(parts) != 2:
        return jsonify({
            'status': 'error',
            'message': 'Invalid Authorization format'
        }), 401
    
    token = parts[1]
    
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided'
            }), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Empty file'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type'
            }), 400
            
        # Save and process file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)
        
        # Process image
        try:
            # Anti-spoofing check first
            spoof_start = time.time()
            spoof_result = DeepFace.extract_faces(
                img_path=temp_path,
                anti_spoofing=True
            )
            spoof_time = time.time() - spoof_start

            if not spoof_result or len(spoof_result) == 0:
                os.remove(temp_path)
                return jsonify({'error': 'No face detected'}), 400

            face_data = spoof_result[0]
            is_real = face_data.get('is_real', False)
            spoof_confidence = float(face_data.get('confidence', 0))

            # Continue with face detection if real face
            detection_start = time.time()
            image = cv2.imread(temp_path)
            detector = MTCNN()
            faces = detector.detect_faces(image)
            detection_time = time.time() - detection_start

            if faces:
                # Time face processing
                process_start = time.time()
                x, y, width, height = faces[0]['box']
                margin = 20
                x = max(x - margin, 0)
                y = max(y - margin, 0)
                width += margin * 2
                height += margin * 2
                
                cropped_face = image[y:y+height, x:x+width]
                resized_face = cv2.resize(cropped_face, (250, 250))
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                cv2.imwrite(file_path, resized_face)
                process_time = time.time() - process_start

                # Continue with embedding generation and matching...
                embedding_start = time.time()
                embedding_obj = DeepFace.represent(
                    img_path=file_path,
                    model_name=FACE_RECOGNITION_MODEL,
                    detector_backend=FACE_DETECTION_MODEL,
                    enforce_detection=False,
                    align=True
                )
                embedding_time = time.time() - embedding_start

                # Time matching
                matching_start = time.time()
                embeddings = load_h5_embeddings()
                current_embedding = np.array(embedding_obj[0]['embedding'])
                
                matched_name, similarity = find_matching_face(current_embedding, embeddings)
                matching_time = time.time() - matching_start
                
                os.remove(temp_path)
                total_time = time.time() - start_time

                if matched_name:
                    # Connect to database to verify user
                    connection = pymysql.connect(**db_config)
                    with connection.cursor() as cursor:
                        sql = "SELECT id_karyawan FROM karyawan WHERE id_karyawan = %s"
                        cursor.execute(sql, (current_user,))
                        user = cursor.fetchone()

                        if not user:
                            return jsonify({
                                'status': 'error',
                                'message': 'User not authenticated'
                            }), 401

                        if matched_name == current_user:
                            return jsonify({
                                'status': 'success',
                                'message': 'Face matched with authenticated user',
                                'data': {
                                    'matched_name': matched_name,
                                    'confidence': float(similarity),
                                    'filename': filename,
                                    'is_real': is_real
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
                        else:
                            return jsonify({
                                'status': 'error',
                                'message': 'Face matched with different user',
                                'matched_user': matched_name,
                                'current_user': current_user
                            }), 401

                return jsonify({
                    'status': 'error',
                    'message': 'No face match found'
                }), 404

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        # Clean up temp files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'connection' in locals():
            connection.close()

    return jsonify({'error': 'Invalid file type'}), 400

@face_recognition_bp.route('/api/update_dataset', methods=['POST'])
def update_dataset():
    start_time = time.time()
    
    try:
        existing_embeddings = load_h5_embeddings()
        existing_users = set(existing_embeddings.keys())
        
        current_folders = set([f for f in os.listdir(BASE_FOLDER) if os.path.isdir(os.path.join(BASE_FOLDER, f))])
        
        removed_folders = existing_users - current_folders
        
        new_folders = current_folders - existing_users
        
        processed_users = []
        skipped_users = []
        removed_users = []
        
        # Remove embeddings for deleted folders
        if removed_folders:
            for username in removed_folders:
                if username in existing_embeddings:
                    del existing_embeddings[username]
                    removed_users.append(username)
        
        # Process new folders
        for username in new_folders:
            user_path = os.path.join(BASE_FOLDER, username)
            user_embeddings = []
            
            for img_name in os.listdir(user_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(user_path, img_name)
                    try:
                        embedding_obj = DeepFace.represent(
                            img_path=img_path,
                            model_name=FACE_RECOGNITION_MODEL,
                            detector_backend=FACE_DETECTION_MODEL,
                            enforce_detection=True,
                            align=True
                        )
                        
                        embedding_vector = np.array(embedding_obj[0]['embedding'])
                        user_embeddings.append(embedding_vector)
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue
            
            if user_embeddings:
                existing_embeddings[username] = np.array(user_embeddings)
                processed_users.append(username)
            else:
                skipped_users.append(username)
        
        # Save updated embeddings
        if removed_users or processed_users:
            with h5py.File(EMBEDDINGS_PATH, 'w') as hf:
                for username, embeddings in existing_embeddings.items():
                    hf.create_dataset(f"{username}", data=embeddings)
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset updated successfully',
            'data': {
                'processed_users': processed_users,
                'skipped_users': skipped_users,
                'removed_users': removed_users,
                'total_processed': len(processed_users),
                'total_skipped': len(skipped_users),
                'total_removed': len(removed_users)
            },
            'time_taken': f"{time.time() - start_time:.3f}s"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'time_taken': f"{time.time() - start_time:.3f}s"
        }), 500
        
@face_recognition_bp.route('/api/spoof', methods=['POST'])        
def check_spoofing():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Generate filename and save temp file
            filename = secure_filename(file.filename)
            filename = get_incremental_filename(UPLOAD_FOLDER, filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_spoof_{filename}")
            file.save(temp_path)

            # Extract faces with anti-spoofing
            result = DeepFace.extract_faces(
                img_path=temp_path,
                anti_spoofing=True
            )
            
            spoof_time = time.time() - start_time
            
            # Clean up temp file
            os.remove(temp_path)
            
            if result and len(result) > 0:
                face_data = result[0]
                return jsonify({
                    'status': 'success',
                    'filename': filename,
                    'anti_spoofing': {
                        'is_real': face_data.get('is_real', False),
                        'confidence': float(face_data.get('confidence', 0)),
                        'result': 'Real' if face_data.get('is_real', False) else 'Fake'
                    },
                    'face_detection': {
                        'found': True,
                        'confidence': float(face_data.get('confidence', 0)),
                        'facial_area': face_data.get('facial_area', {})
                    },
                    'timings': {
                        'process_time': f"{spoof_time:.3f}s"
                    }
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'No face detected or spoofing check failed'
                }), 400

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
@face_recognition_bp.route('/api/check_db_connection', methods=['GET'])
def check_db_connection():
    print("Database Configuration:")
    print("Host:", db_config["host"])
    print("User:", db_config["user"])
    print("Password:", db_config["password"])
    print("Database:", db_config["database"])

    try:
        # Periksa apakah konfigurasi tersedia
        if not all(db_config.values()):
            raise ValueError("Database configuration is incomplete. Check environment variables.")

        # Membuka koneksi ke database
        connection = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"]
        )
        
        return jsonify({"status": "success", "message": "Database connected successfully!"}), 200
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except pymysql.MySQLError as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            if connection:
                connection.close()
        except NameError:
            pass

@face_recognition_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print("\n=== Login Request ===")
        print(f"Time: {datetime.now()}")
        print("Request data:", json.dumps(data, indent=2))
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            response = {
                'status': 'error',
                'message': 'Email and password are required'
            }
            print("\n=== Login Response ===")
            print(json.dumps(response, indent=2))
            return jsonify(response), 400
            
        connection = pymysql.connect(**db_config)
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id_karyawan, email, nama
                    FROM karyawan 
                    WHERE email = %s AND password = %s
                """, (email, password))
                
                user = cursor.fetchone()
                
                if user:
                    access_token = create_access_token(
                        identity=user['nama'],
                        expires_delta=False
                        # Token will never expire
                    )
                    
                    response = {
                        'status': 'success',
                        'message': 'Login successful',
                        'data': {
                            'id_karyawan': user['id_karyawan'],
                            'email': user['email'],
                            'nama': user['nama'],
                            'token': access_token,
                            'token_type': 'Bearer',
                            'expires_in': None  # Token does not expire
                        }
                    }
                    print("\n=== Login Response ===")
                    print(json.dumps(response, indent=2))
                    return jsonify(response), 200
                else:
                    response = {
                        'status': 'error',
                        'message': 'Invalid email or password'
                    }
                    print("\n=== Login Response ===")
                    print(json.dumps(response, indent=2))
                    return jsonify(response), 401
                    
        finally:
            connection.close()
            
    except Exception as e:
        response = {
            'status': 'error',
            'message': str(e)
        }
        print("\n=== Login Error ===")
        print(json.dumps(response, indent=2))
        return jsonify(response), 500

@face_recognition_bp.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200