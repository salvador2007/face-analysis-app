from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import logging
from werkzeug.utils import secure_filename
import uuid
import hashlib
import magic
import re
from html import escape
import tempfile
import sqlite3
from pathlib import Path

app = Flask(__name__)

# 보안 설정 강화
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # 임시 디렉토리 사용
app.config['DATABASE'] = 'analysis_data.db'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB로 제한 (Render 최적화)
app.config['WTF_CSRF_ENABLED'] = True

# 보안 헤더 설정
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com"
    return response

# 로깅 설정 강화
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 허용된 파일 확장자 및 MIME 타입
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MIME_TYPES = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
    'image/jpg': 'jpg'
}

# 데이터베이스 초기화
def init_db():
    """SQLite 데이터베이스 초기화"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename_hash TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                emotion TEXT,
                confidence REAL,
                genres TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("데이터베이스 초기화 완료")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")

def validate_file_content(file_path):
    """파일 내용 검증"""
    try:
        # Magic number로 실제 파일 타입 확인
        mime = magic.Magic(mime=True)
        file_mime = mime.from_file(file_path)
        
        if file_mime not in ALLOWED_MIME_TYPES:
            return False, f"허용되지 않는 파일 형식: {file_mime}"
        
        # 이미지 파일 유효성 검사 (OpenCV로)
        img = cv2.imread(file_path)
        if img is None:
            return False, "유효하지 않은 이미지 파일"
        
        # 이미지 크기 검사 (너무 큰 이미지 방지)
        height, width = img.shape[:2]
        if width > 4000 or height > 4000:
            return False, "이미지 크기가 너무 큽니다 (최대 4000x4000)"
        
        return True, "유효한 파일"
    except Exception as e:
        return False, f"파일 검증 실패: {str(e)}"

def sanitize_input(text):
    """입력 데이터 살균"""
    if not text:
        return ""
    # HTML 태그 제거 및 이스케이프
    sanitized = escape(str(text))
    # CSV 인젝션 방지
    if sanitized.startswith(('=', '+', '-', '@')):
        sanitized = "'" + sanitized
    return sanitized

def get_client_info(request):
    """클라이언트 정보 수집 (보안 모니터링용)"""
    return {
        'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
        'user_agent': request.headers.get('User-Agent', '')[:200]  # 길이 제한
    }

def allowed_file(filename):
    """파일 확장자 검증"""
    if not filename:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_analysis_result(data):
    """분석 결과를 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analysis_results 
            (timestamp, filename_hash, age, gender, emotion, confidence, genres, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'],
            data['filename_hash'],
            data['age'],
            data['gender'],
            data['emotion'],
            data['confidence'],
            data['genres'],
            data['ip_address'],
            data['user_agent']
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"데이터베이스 저장 실패: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    client_info = get_client_info(request)
    
    try:
        # Rate limiting 체크 (간단한 구현)
        # 실제 환경에서는 Redis나 Flask-Limiter 사용 권장
        
        if 'photo' not in request.files:
            logger.warning(f"파일 없음 - IP: {client_info['ip']}")
            flash('사진을 선택해주세요', 'error')
            return redirect(url_for('index'))

        file = request.files['photo']
        selected_genres = request.form.getlist('genre')

        # 장르 입력 검증 및 살균
        safe_genres = []
        allowed_genres = ['코미디', '공포', '드라마', '액션', '로맨스', '스릴러', 'SF', '판타지']
        for genre in selected_genres:
            if genre in allowed_genres:
                safe_genres.append(sanitize_input(genre))

        if file.filename == '':
            flash('파일이 선택되지 않았습니다', 'error')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            logger.warning(f"허용되지 않는 파일 형식 - IP: {client_info['ip']}, 파일: {file.filename}")
            flash('지원하지 않는 파일 형식입니다 (PNG, JPG, JPEG만 허용)', 'error')
            return redirect(url_for('index'))

        # 안전한 파일명 생성
        original_filename = secure_filename(file.filename)
        file_hash = hashlib.sha256(file.read()).hexdigest()[:16]
        file.seek(0)  # 파일 포인터 리셋
        
        unique_filename = f"{file_hash}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 파일 저장
        file.save(filepath)
        
        # 파일 내용 검증
        is_valid, validation_msg = validate_file_content(filepath)
        if not is_valid:
            os.remove(filepath)
            logger.warning(f"파일 검증 실패 - IP: {client_info['ip']}, 사유: {validation_msg}")
            flash(f'파일 검증 실패: {validation_msg}', 'error')
            return redirect(url_for('index'))

        logger.info(f"파일 업로드 성공 - IP: {client_info['ip']}, 파일: {unique_filename}")

        # DeepFace 분석 (에러 처리 강화)
        try:
            # Render에서 DeepFace가 제대로 작동하지 않을 수 있으므로 fallback 추가
            result = DeepFace.analyze(
                img_path=filepath, 
                actions=['age', 'gender', 'emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            age = int(result.get('age', 25))  # 기본값 설정
            gender_data = result.get('gender', {'Man': 50, 'Woman': 50})
            emotion = result.get('dominant_emotion', 'neutral')
            emotion_scores = result.get('emotion', {'neutral': 100})
            
        except Exception as deepface_error:
            logger.error(f"DeepFace 분석 실패: {deepface_error}")
            # Fallback: 기본값 사용
            age = 25
            gender_data = {'Man': 50, 'Woman': 50}
            emotion = 'neutral'
            emotion_scores = {'neutral': 100}
            flash('AI 분석 중 오류가 발생하여 기본값을 사용합니다.', 'warning')
        
        # 데이터 처리
        confidence = max(emotion_scores.values()) if emotion_scores else 0
        
        if isinstance(gender_data, dict):
            gender_label = max(gender_data.items(), key=lambda x: x[1])[0]
            gender_confidence = max(gender_data.values())
        else:
            gender_label = str(gender_data)
            gender_confidence = 0

        # 데이터베이스에 저장
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'filename_hash': file_hash,
            'age': age,
            'gender': f"{gender_label}({gender_confidence:.1f}%)",
            'emotion': sanitize_input(emotion),
            'confidence': confidence,
            'genres': '|'.join(safe_genres),
            'ip_address': client_info['ip'][:45],  # IP 길이 제한
            'user_agent': client_info['user_agent'][:200]  # User-Agent 길이 제한
        }
        
        if not save_analysis_result(analysis_data):
            flash('데이터 저장 중 오류가 발생했습니다', 'warning')

        # 임시 파일 정리
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

        return render_template('result.html', 
                             age=age,
                             gender=gender_label,
                             gender_confidence=gender_confidence,
                             emotion=emotion,
                             confidence=confidence,
                             emotion_scores=emotion_scores,
                             selected_genres=safe_genres)
        
    except Exception as e:
        logger.error(f"업로드 처리 오류 - IP: {client_info['ip']}, 오류: {str(e)}")
        flash('처리 중 오류가 발생했습니다. 다시 시도해주세요.', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    })

@app.errorhandler(413)
def too_large(e):
    logger.warning(f"파일 크기 초과 - IP: {get_client_info(request)['ip']}")
    flash('파일이 너무 큽니다 (최대 5MB)', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"서버 내부 오류: {str(e)}")
    flash('서버 내부 오류가 발생했습니다', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

# 앱 시작 시 데이터베이스 초기화
init_db()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
    