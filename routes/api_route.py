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
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import json
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from functools import wraps
import jwt
from retinaface import RetinaFace

# Load environment variables
load_dotenv()

jwt = JWTManager()

def init_jwt(app):
    jwt.init_app(app)

# Blueprint Initialization
api_route_bp = Blueprint('api_route_bp', __name__)


# Get configurations from environment
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
BASE_FOLDER = os.getenv('BASE_FOLDER')
FACE_DETECTION_MODEL = os.getenv('FACE_DETECTION_MODEL')
FACE_RECOGNITION_MODEL = os.getenv('FACE_RECOGNITION_MODEL')
FACE_IMAGE_SIZE = int(os.getenv('FACE_IMAGE_SIZE'))
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH')
MODEL_FOLDER = os.getenv('MODEL_FOLDER')

db_config = {
    "host": os.getenv("POSTGRES_HOST"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_DB"),
    "port": os.getenv("POSTGRES_PORT")
}

def get_db_connection():
    try:
        connection = psycopg2.connect(**db_config)
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    
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
@api_route_bp.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'status': 'API is running'})

def crop_and_save_face(image_data, output_path, filename):
    try:
        # Initialize MTCNN detector
        detector = RetinaFace
        
        # Convert image data to cv2 format
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            image = image_data

        # Detect faces
        faces = detector.detect_faces(image)

        if isinstance(faces, dict):  # RetinaFace mengembalikan dict jika wajah terdeteksi
            # Ambil wajah pertama yang terdeteksi
            # RetinaFace menggunakan format berbeda untuk koordinat dibanding MTCNN
            face_data = faces['face_1']
            facial_area = face_data['facial_area']
            
            # Format RetinaFace: [x1, y1, x2, y2] (koordinat sudut)
            x = facial_area[0]
            y = facial_area[1]
            width = facial_area[2] - facial_area[0]
            height = facial_area[3] - facial_area[1]
            
            # Tambahkan margin seperti sebelumnya
            margin = 20
            x = max(int(x - margin), 0)
            y = max(int(y - margin), 0)
            width = int(width + margin * 2)
            height = int(height + margin * 2)

            # Pastikan koordinat tidak melebihi ukuran gambar
            h, w = image.shape[:2]
            width = min(width, w - x)
            height = min(height, h - y)

            # Crop dan resize wajah
            cropped_face = image[y:y+height, x:x+width]
            cropped_face_resized = cv2.resize(cropped_face, (250, 250))

            # Simpan hasil crop
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
@api_route_bp.route('/api/save_image', methods=['POST'])
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

@api_route_bp.route('/api/save_user', methods=['POST'])
def save_user():
    data = request.get_json()
    print("Received data:", data)
    username = data.get('name')
    email = data.get('email')
    role = data.get('role')
    password = "123"

    if not username or not email or not role:
        return jsonify({'error': 'All fields are required'}), 400

    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO karyawan (nama, email, role, password) 
                VALUES (%s, %s, %s, %s)
                RETURNING id_karyawan
            """
            cursor.execute(sql, (username, email, role, password))
            connection.commit()
            
        return jsonify({'message': 'User saved successfully', 'password': password}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()
            
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

location_data = {}

@api_route_bp.route('/api/getlocation', methods=['POST'])
def get_location():
    try:
        data = request.get_json()
        global location_data
        
        location_data = {
            'location_type': data.get('location_data', {}).get('location_type'),
            'office_name': data.get('location_data', {}).get('office_name'),
            'timestamp': data.get('location_data', {}).get('timestamp'),
            'latitude': data.get('location_data', {}).get('latitude'),
            'longitude': data.get('location_data', {}).get('longitude')
        }
        print(location_data)
        return jsonify({
            'status': 'success',
            'message': 'Location data received',
            'data': location_data
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 400
        
@api_route_bp.route('/api/upload', methods=['POST'])
@jwt_required()
def upload():
    start_time = time.time()
    temp_path = None
    
    try:
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
                
        # Save to temp only
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
        file.save(temp_path)
        
        # Anti-spoofing check
        spoof_start = time.time()
        spoof_result = DeepFace.extract_faces(
            img_path=temp_path,
            anti_spoofing=True
        )
        spoof_time = time.time() - spoof_start

        if not spoof_result or len(spoof_result) == 0:
            return jsonify({'error': 'No face detected'}), 400

        face_data = spoof_result[0]
        is_real = face_data.get('is_real', False)
        spoof_confidence = float(face_data.get('confidence', 0))

        # Face detection check
        detection_start = time.time()
        image = cv2.imread(temp_path)
        detector = RetinaFace
        faces = detector.detect_faces(image)
        detection_time = time.time() - detection_start

        if not faces:
            return jsonify({
                'status': 'error',
                'message': 'No face detected in image'
            }), 400

        # Generate embedding directly from temp file
        embedding_start = time.time()
        embedding_obj = DeepFace.represent(
            img_path=temp_path,
            model_name=FACE_RECOGNITION_MODEL,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True
        )
        embedding_time = time.time() - embedding_start

        # Face matching
        matching_start = time.time()
        embeddings = load_h5_embeddings()
        current_embedding = np.array(embedding_obj[0]['embedding'])
        matched_name, similarity = find_matching_face(current_embedding, embeddings)
        matching_time = time.time() - matching_start
        
        if not matched_name:
            return jsonify({
                'status': 'error',
                'message': 'No face match found'
            }), 404

        # Database verification
        try:
            connection = get_db_connection()
            with connection.cursor(cursor_factory=DictCursor) as cursor:
                sql = "SELECT id_karyawan FROM karyawan WHERE id_karyawan = %s"
                cursor.execute(sql, (current_user,))
                user = cursor.fetchone()
        finally:
            if connection:
                connection.close()

        # Prepare response
        response_data = {
            'status': 'success' if matched_name == current_user else 'error',
            'message': 'Face matched with authenticated user' if matched_name == current_user 
                      else 'Face matched with different user',
            'data': {
                'matched_name': matched_name,
                'confidence': float(similarity),
                'database_name': current_user,
                'is_real': is_real,
                'spoof_confidence': spoof_confidence
            },
            'timing': {
                'spoofing': f"{spoof_time:.3f}s",
                'detection': f"{detection_time:.3f}s",
                'embedding': f"{embedding_time:.3f}s",
                'matching': f"{matching_time:.3f}s",
                'total': f"{time.time() - start_time:.3f}s"
            },
            'location': location_data,
        }
        print(response_data)
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

@api_route_bp.route('/api/update_dataset', methods=['POST'])
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

@api_route_bp.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print("\n=== Login Request ===")
        print(f"Time: {datetime.now()}")
        print("Request data:", json.dumps(data, indent=2))
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                'status': 'error',
                'message': 'Email and password are required'
            }), 400
            
        connection = get_db_connection()
        try:
            with connection.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT id_karyawan, email, nama, role
                    FROM karyawan 
                    WHERE email = %s AND password = %s
                """, (email, password))
                
                user = cursor.fetchone()
                
                if user:
                    access_token = create_access_token(
                        identity=user['nama'],
                        expires_delta=False
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
                            'role': user['role'],
                            'expires_in': None  # Token does not expire
                        }
                    }
                    print("\n=== Login Response ===")
                    print(json.dumps(response, indent=2))
                    return jsonify(response), 200
                else:
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid email or password'
                    }), 401
        finally:
            connection.close()        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
            
@api_route_bp.route('/api/presence', methods=['POST'])
def save_presence():
    try:
        json_data = request.get_json()
        location_data = json_data.get('location', {})
        data = json_data.get('data', {})
        
        timestamp = location_data.get('timestamp')
        latitude = location_data.get('latitude')
        longitude = location_data.get('longitude')
        office_name = location_data.get('office_name')
        location_type = location_data.get('location_type')
        database_name = data.get('database_name')
        
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = """
                INSERT INTO absensi 
                (id_karyawan, nama, work_type, office, latitude, longitude, absensi_masuk) 
                VALUES (
                    (SELECT id_karyawan FROM karyawan WHERE nama = %s), 
                    %s, %s, %s, %s, %s, %s
                )
            """
            
            cursor.execute(sql, (
                database_name,
                database_name,
                location_type,
                office_name,
                latitude,
                longitude,
                timestamp
            ))
            
            connection.commit()
            
        return jsonify({
            'status': 'success',
            'message': 'Presence recorded successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if connection:
            connection.close()

@api_route_bp.route('/api/presencecheck', methods=['GET'])
def check_presence():
    try:
        user_name = request.args.get('nama')
        user_id = request.args.get('id_karyawan')

        if not user_name and not user_id:
            return jsonify({
                'status': 'error',
                'message': 'Parameter "name" atau "id" diperlukan untuk memeriksa presensi.'
            }), 400

        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=DictCursor)

        if user_id:
            sql = """
                SELECT * FROM absensi 
                WHERE id_karyawan = %s AND DATE(absensi_masuk) = CURRENT_DATE
            """
            cursor.execute(sql, (user_id,))
        else:
            sql = """
                SELECT * FROM absensi 
                WHERE nama = %s AND DATE(absensi_masuk) = CURRENT_DATE
            """
            cursor.execute(sql, (user_name,))

        result = cursor.fetchone()

        if result:
            return jsonify({
                'status': 'success',
                'message': 'User telah melakukan presensi hari ini.',
                'data': dict(result)
            }), 200
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'User belum melakukan presensi hari ini.'
            }), 404

    except psycopg2.Error as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

@api_route_bp.route('/api/history', methods=['GET'])
def history():
    try:
        user_name = request.args.get('nama')

        if not user_name:
            return jsonify({
                'status': 'error',
                'message': 'Parameter "nama" diperlukan untuk melihat history.'
            }), 400

        connection = get_db_connection()
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            sql = """
                SELECT 
                    TO_CHAR(absensi_masuk, 'YYYY-MM-DD') as tanggal,
                    TO_CHAR(absensi_masuk, 'HH24:MI:SS') as jam,
                    nama,
                    work_type
                FROM absensi 
                WHERE nama = %s
                ORDER BY absensi_masuk DESC
            """
            
            cursor.execute(sql, (user_name,))
            records = cursor.fetchall()

            if not records:
                return jsonify({
                    'status': 'error',
                    'message': 'Data absensi tidak ditemukan'
                }), 404

            absensi_data = [{
                'tanggal': record['tanggal'],
                'jam': record['jam'],
                'nama': record['nama'],
                'work_type': record['work_type']
            } for record in records]

            return jsonify({
                'status': 'success',
                'data': absensi_data
            }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if connection:
            connection.close()