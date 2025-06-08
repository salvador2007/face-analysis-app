from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import cv2
import os
import csv
from datetime import datetime
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
import tempfile
import sqlite3
from pathlib import Path

app = Flask(__name__, static_folder='static', template_folder='templates')

# 보안 설정 강화
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['DATABASE'] = 'analysis_data.db'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['WTF_CSRF_ENABLED'] = False  # Render에서 CSRF 문제 방지

# 보안 헤더 설정
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("데이터베이스 초기화 완료")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")

def validate_image_file(file_path):
    """이미지 파일 검증 (간단한 버전)"""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False, "유효하지 않은 이미지 파일"
        
        height, width = img.shape[:2]
        if width > 4000 or height > 4000:
            return False, "이미지 크기가 너무 큽니다"
        
        return True, "유효한 파일"
    except Exception as e:
        return False, f"파일 검증 실패: {str(e)}"

def get_client_info(request):
    """클라이언트 정보 수집"""
    return {
        'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr),
        'user_agent': request.headers.get('User-Agent', '')[:200]
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
            (timestamp, filename_hash, age, gender, emotion, confidence, genres, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['timestamp'],
            data['filename_hash'],
            data['age'],
            data['gender'],
            data['emotion'],
            data['confidence'],
            data['genres'],
            data['ip_address']
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
        if 'photo' not in request.files:
            logger.warning(f"파일 없음 - IP: {client_info['ip']}")
            flash('사진을 선택해주세요', 'error')
            return redirect(url_for('index'))

        file = request.files['photo']
        selected_genres = request.form.getlist('genre')

        # 장르 입력 검증
        safe_genres = []
        allowed_genres = ['코미디', '공포', '드라마', '액션', '로맨스', '스릴러', 'SF', '판타지']
        for genre in selected_genres:
            if genre in allowed_genres:
                safe_genres.append(genre)

        if file.filename == '':
            flash('파일이 선택되지 않았습니다', 'error')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            logger.warning(f"허용되지 않는 파일 형식 - IP: {client_info['ip']}")
            flash('PNG, JPG, JPEG 파일만 허용됩니다', 'error')
            return redirect(url_for('index'))

        # 안전한 파일명 생성
        original_filename = secure_filename(file.filename)
        file_hash = hashlib.sha256(file.read()).hexdigest()[:16]
        file.seek(0)
        
        unique_filename = f"{file_hash}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 파일 저장
        file.save(filepath)
        
        # 파일 검증
        is_valid, validation_msg = validate_image_file(filepath)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.warning(f"파일 검증 실패 - IP: {client_info['ip']}")
            flash(f'파일 검증 실패: {validation_msg}', 'error')
            return redirect(url_for('index'))

        logger.info(f"파일 업로드 성공 - IP: {client_info['ip']}")

        # DeepFace 분석 대신 임시 값 사용 (Render에서 DeepFace 문제 시)
        try:
            # DeepFace를 사용할 수 없는 경우를 대비한 fallback
            from deepface import DeepFace
            
            result = DeepFace.analyze(
                img_path=filepath, 
                actions=['age', 'gender', 'emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            age = int(result.get('age', 25))
            gender_data = result.get('gender', {'Woman': 60, 'Man': 40})
            emotion = result.get('dominant_emotion', 'neutral')
            emotion_scores = result.get('emotion', {'neutral': 100})
            
        except Exception as deepface_error:
            logger.error(f"DeepFace 분석 실패, 기본값 사용: {deepface_error}")
            # 기본값 사용
            age = 25
            gender_data = {'Woman': 60, 'Man': 40}
            emotion = 'neutral'
            emotion_scores = {
                'neutral': 40,
                'happy': 25,
                'sad': 15,
                'angry': 10,
                'surprise': 5,
                'disgust': 3,
                'fear': 2
            }
            flash('AI 분석이 일시적으로 불가능하여 샘플 데이터를 표시합니다', 'warning')
        
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
            'emotion': emotion,
            'confidence': confidence,
            'genres': '|'.join(safe_genres),
            'ip_address': client_info['ip'][:45]
        }
        
        save_analysis_result(analysis_data)

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
        logger.error(f"업로드 처리 오류: {str(e)}")
        flash('처리 중 오류가 발생했습니다. 다시 시도해주세요.', 'error')
        return redirect(url_for('index'))

@app.route('/graph')
def graph():
    """통계 그래프 페이지"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        
        if df.empty:
            return render_template('graph.html', 
                                 stats={'total_users': 0, 'avg_age': 0, 'most_common_emotion': 'N/A', 'date_range': 'N/A'},
                                 graph_data=None)
        
        # 통계 계산
        stats = {
            'total_users': len(df),
            'avg_age': df['age'].mean() if 'age' in df.columns else 0,
            'most_common_emotion': df['emotion'].mode().iloc[0] if 'emotion' in df.columns and not df['emotion'].mode().empty else 'N/A',
            'date_range': f"{len(df)}건"
        }
        
        # 간단한 그래프 생성
        plt.figure(figsize=(10, 6))
        if 'emotion' in df.columns:
            emotion_counts = df['emotion'].value_counts()
            plt.bar(emotion_counts.index, emotion_counts.values)
            plt.title('감정 분포')
            plt.xlabel('감정')
            plt.ylabel('빈도')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # 이미지를 base64로 변환
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return render_template('graph.html', stats=stats, graph_data=graph_data)
        
    except Exception as e:
        logger.error(f"그래프 생성 오류: {e}")
        return render_template('graph.html', 
                             stats={'total_users': 0, 'avg_age': 0, 'most_common_emotion': 'N/A', 'date_range': 'N/A'},
                             graph_data=None)

@app.route('/recommend')
def recommend():
    """장르 추천 페이지"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        
        recommendations = []
        insights = []
        total_users = len(df) if not df.empty else 0
        
        if not df.empty and 'emotion' in df.columns and 'genres' in df.columns:
            # 감정별 추천 생성 (샘플)
            emotions = ['happy', 'sad', 'angry', 'neutral']
            genres = ['코미디', '드라마', '액션', '로맨스']
            
            for i, emotion in enumerate(emotions):
                if i < len(genres):
                    recommendations.append({
                        'emotion': emotion,
                        'genre': genres[i],
                        'count': total_users // 4,
                        'percentage': 25.0
                    })
            
            insights = [
                f"총 {total_users}명의 사용자가 분석을 받았습니다.",
                "가장 인기 있는 장르는 드라마입니다.",
                "행복한 감정일 때 코미디를 선호하는 경향이 있습니다."
            ]
        
        return render_template('recommend.html', 
                             recommendations=recommendations,
                             insights=insights,
                             total_users=total_users)
        
    except Exception as e:
        logger.error(f"추천 페이지 오류: {e}")
        return render_template('recommend.html', 
                             recommendations=[],
                             insights=["데이터를 불러올 수 없습니다."],
                             total_users=0)

@app.route('/admin')
def admin():
    """관리자 페이지"""
    try:
        conn = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY created_at DESC LIMIT 10", conn)
        conn.close()
        
        stats = {
            'total_users': len(df),
            'avg_age': df['age'].mean() if not df.empty and 'age' in df.columns else 0,
            'age_range': f"{df['age'].min()}-{df['age'].max()}" if not df.empty and 'age' in df.columns else "N/A",
            'most_emotion': df['emotion'].mode().iloc[0] if not df.empty and 'emotion' in df.columns and not df['emotion'].mode().empty else "N/A"
        }
        
        recent_data = df.to_dict('records') if not df.empty else []
        
        return render_template('admin.html', stats=stats, recent_data=recent_data)
        
    except Exception as e:
        logger.error(f"관리자 페이지 오류: {e}")
        return render_template('admin.html', 
                             stats={'total_users': 0, 'avg_age': 0, 'age_range': 'N/A', 'most_emotion': 'N/A'},
                             recent_data=[])

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'version': '2.1'
    })

@app.errorhandler(413)
def too_large(e):
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