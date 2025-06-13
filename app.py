from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import hashlib
from datetime import datetime
import tempfile
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
# --- 얼굴형 분류 모델 준비 (app.py 상단에) ---
import json
from tensorflow import keras
import numpy as np
from PIL import Image

# 모델 경로
FACE_MODEL_JSON = 'face_shape_optimized_model_architecture.json'
FACE_MODEL_WEIGHTS = 'best_face_shape_optimized_model_01_0.1990.weights.h5'  # 최신/best로 수정
CLASS_INDICES = 'class_indices.json'

# 전역 모델, 클래스명
face_shape_model = None
face_shape_classes = []

def load_face_shape_model():
    global face_shape_model, face_shape_classes
    if face_shape_model is not None and face_shape_classes:
        return  # 이미 로드됨
    with open(FACE_MODEL_JSON, encoding='utf-8') as f:
        face_shape_model = keras.models.model_from_json(f.read())
    face_shape_model.load_weights(FACE_MODEL_WEIGHTS)
    with open(CLASS_INDICES, encoding='utf-8') as f:
        index_map = json.load(f)
        face_shape_classes.clear()
        for k, v in sorted(index_map.items(), key=lambda x: x[1]):
            face_shape_classes.append(k)
load_face_shape_model()

def predict_face_shape(img_path):
    try:
        # 이미지 불러오기 & 전처리 (224x224, RGB, 정규화)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = face_shape_model.predict(arr)
        idx = np.argmax(preds)
        prob = float(np.max(preds))
        label = face_shape_classes[idx]
        return label, prob
    except Exception as e:
        logger.error(f"얼굴형 예측 오류: {e}")
        return "Unknown", 0.0

# DeepFace 임포트를 try-catch로 감싸서 에러 처리
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Face analysis will use mock data.")

app = Flask(__name__)

# 보안 강화된 설정
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# 설정
DB_PATH = os.environ.get('DB_PATH', 'analysis_data.db')
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get('ADMIN_PASSWORD', 'admin1234'))
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 업로드 폴더 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o755)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (에러 방지)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except Exception:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def init_db():
    """데이터베이스 초기화 및 보안 설정"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                age INTEGER CHECK(age >= 0 AND age <= 150),
                gender TEXT CHECK(gender IN ('Man', 'Woman', 'Unknown')),
                gender_confidence REAL CHECK(gender_confidence >= 0 AND gender_confidence <= 100),
                emotion TEXT,
                emotion_confidence REAL CHECK(emotion_confidence >= 0 AND emotion_confidence <= 100),
                emotion_scores TEXT,
                genres TEXT,
                filename_hash TEXT,
                face_shape TEXT DEFAULT 'Unknown',
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        # 인덱스 생성 (성능 향상)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_results(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion ON analysis_results(emotion)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename_hash ON analysis_results(filename_hash)')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def validate_mime_type(file_content):
    """파일 내용 기반 MIME 타입 검증"""
    try:
        if file_content.startswith(b'\xff\xd8\xff'):  # JPEG
            return True
        elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return True
        return False
    except Exception:
        return False

def get_file_hash(file_content):
    """파일 해시 생성 (보안 강화)"""
    try:
        return hashlib.sha256(file_content).hexdigest()[:16]
    except Exception as e:
        logger.error(f"Hash generation error: {e}")
        return "unknown"

def get_client_ip():
    """클라이언트 IP 주소 안전하게 가져오기"""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        ip = request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    else:
        ip = request.environ.get('REMOTE_ADDR', 'unknown')
    return ip[:45]  # IPv6 최대 길이 제한

def sanitize_user_agent():
    """User Agent 문자열 정리"""
    user_agent = request.headers.get('User-Agent', 'unknown')
    return user_agent[:500]  # 길이 제한

def analyze_face(image_path):
    """얼굴 분석 함수 (DeepFace 사용 또는 모의 데이터)"""
    if not DEEPFACE_AVAILABLE:
        import random
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        genders = ['Man', 'Woman']
        dominant_emotion = random.choice(emotions)
        emotion_scores = {emotion: random.uniform(0, 100) for emotion in emotions}
        emotion_scores[dominant_emotion] = max(emotion_scores[dominant_emotion], 70)
        return {
            'age': random.randint(18, 65),
            'gender': random.choice(genders),
            'gender_confidence': random.uniform(70, 95),
            'emotion': dominant_emotion,
            'emotion_confidence': emotion_scores[dominant_emotion],
            'emotion_scores': emotion_scores
        }
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            silent=True
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        age = max(0, min(150, int(analysis.get('age', 25))))
        gender = analysis.get('gender', 'Unknown')
        emotion = analysis.get('dominant_emotion', 'neutral')
        emotion_scores = analysis.get('emotion', {'neutral': 100})
        gender_confidence = 85.0
        if isinstance(gender, dict):
            gender_confidence = max(0, min(100, max(gender.values())))
            gender = max(gender, key=gender.get)
        if gender not in ['Man', 'Woman']:
            gender = 'Unknown'
        emotion_confidence = max(0, min(100, emotion_scores.get(emotion, 0)))
        return {
            'age': age,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'emotion': emotion,
            'emotion_confidence': emotion_confidence,
            'emotion_scores': emotion_scores
        }
    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        return {
            'age': 25,
            'gender': 'Unknown',
            'gender_confidence': 0,
            'emotion': 'neutral',
            'emotion_confidence': 0,
            'emotion_scores': {'neutral': 100}
        }

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error for {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 error: {e}")
    return render_template('500.html'), 500

@app.errorhandler(413)
def file_too_large(e):
    flash('파일 크기가 너무 큽니다. 최대 16MB까지 업로드 가능합니다.', 'error')
    return redirect(url_for('index'))

@app.route('/health')
def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('SELECT 1')
        conn.close()
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "deepface_available": DEEPFACE_AVAILABLE
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'photo' not in request.files:
            flash('파일이 선택되지 않았습니다.', 'error')
            return redirect(url_for('index'))
        file = request.files['photo']
        if file.filename == '':
            flash('파일이 선택되지 않았습니다.', 'error')
            return redirect(url_for('index'))
        if not allowed_file(file.filename):
            flash('PNG, JPG, JPEG 파일만 업로드 가능합니다.', 'error')
            return redirect(url_for('index'))
        file_content = file.read()
        if len(file_content) == 0:
            flash('빈 파일은 업로드할 수 없습니다.', 'error')
            return redirect(url_for('index'))
        if not validate_mime_type(file_content):
            flash('유효하지 않은 이미지 파일입니다.', 'error')
            return redirect(url_for('index'))
        file_hash = get_file_hash(file_content)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM analysis_results 
            WHERE filename_hash = ? AND timestamp > datetime('now', '-1 hour')
        ''', (file_hash,))
        recent_count = cursor.fetchone()[0]
        conn.close()
        if recent_count > 5:
            flash('같은 파일을 너무 자주 업로드했습니다. 잠시 후 다시 시도해주세요.', 'error')
            return redirect(url_for('index'))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=UPLOAD_FOLDER) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        try:
            analysis_result = analyze_face(temp_path)
            selected_genres = request.form.getlist('genre')
            allowed_genres = ['액션', '코미디', '드라마', '공포', '로맨스', 'SF', '다큐멘터리', '애니메이션']
            selected_genres = [g for g in selected_genres if g in allowed_genres]
            genres_str = ', '.join(selected_genres[:10])
            client_ip = get_client_ip()
            user_agent = sanitize_user_agent()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_results 
                (timestamp, age, gender, gender_confidence, emotion, emotion_confidence, 
                emotion_scores, genres, filename_hash, face_shape, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                analysis_result['age'],
                analysis_result['gender'],
                analysis_result['gender_confidence'],
                analysis_result['emotion'],
                analysis_result['emotion_confidence'],
                str(analysis_result['emotion_scores']),
                genres_str,
                file_hash,
                'Unknown',
                client_ip,
                user_agent
            ))
            conn.commit()
            conn.close()
            logger.info(f"Analysis completed for hash: {file_hash}")
            return render_template('result.html',
                age=analysis_result['age'],
                gender=analysis_result['gender'],
                gender_confidence=analysis_result['gender_confidence'],
                emotion=analysis_result['emotion'],
                confidence=analysis_result['emotion_confidence'],
                emotion_scores=analysis_result['emotion_scores'],
                selected_genres=selected_genres)
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
    except Exception as e:
        logger.error(f'파일 처리 중 오류: {str(e)}')
        flash('파일 처리 중 오류가 발생했습니다. 다시 시도해주세요.', 'error')
        return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if len(password) > 200:
            flash('입력이 너무 깁니다.', 'error')
            return render_template('login.html')
        if check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['logged_in'] = True
            session['login_time'] = datetime.now().isoformat()
            logger.info(f"Admin login successful from {get_client_ip()}")
            return redirect(url_for('admin'))
        else:
            logger.warning(f"Failed admin login attempt from {get_client_ip()}")
            flash('비밀번호가 올바르지 않습니다.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logger.info(f"Admin logout from {get_client_ip()}")
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, age, gender, emotion, genres, filename_hash, ip_address
            FROM analysis_results 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (50,))
        rows = cursor.fetchall()
        conn.close()
        table_html = '''
        <table class="data-table">
            <thead>
                <tr>
                    <th>시간</th>
                    <th>나이</th>
                    <th>성별</th>
                    <th>감정</th>
                    <th>선호 장르</th>
                    <th>파일 해시</th>
                    <th>IP 주소</th>
                </tr>
            </thead>
            <tbody>
        '''
        for row in rows:
            escaped_row = [str(cell).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') for cell in row]
            table_html += f'''
                <tr>
                    <td>{escaped_row[0]}</td>
                    <td>{escaped_row[1]}</td>
                    <td>{escaped_row[2]}</td>
                    <td>{escaped_row[3]}</td>
                    <td>{escaped_row[4]}</td>
                    <td>{escaped_row[5]}</td>
                    <td>{escaped_row[6]}</td>
                </tr>
            '''
        table_html += '</tbody></table>'
        return render_template('admin.html', table_html=table_html)
    except Exception as e:
        logger.error(f'관리자 페이지 데이터 로드 중 오류: {str(e)}')
        flash('데이터를 불러오는 중 오류가 발생했습니다.', 'error')
        return render_template('admin.html', table_html='<p>데이터를 불러올 수 없습니다.</p>')

@app.route('/graph')
def graph():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        if df.empty:
            return render_template('graph.html', 
                                genre_plot="", 
                                emotion_plot="", 
                                face_plot="")
        genre_plot = create_genre_plot(df)
        emotion_plot = create_emotion_plot(df)
        face_plot = create_face_plot(df)
        return render_template('graph.html',
                            genre_plot=genre_plot,
                            emotion_plot=emotion_plot,
                            face_plot=face_plot)
    except Exception as e:
        logger.error(f"Graph generation error: {e}")
        return render_template('graph.html', 
                            genre_plot="", 
                            emotion_plot="", 
                            face_plot="")
    
    
@app.route('/correlation') 
def correlation():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        cols = ["age", "emotion", "face_shape", "genres"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df_clean = df[cols].dropna()
        if df_clean.empty:
            return render_template("correlation.html", correlation_plot="")
        # 범주형 변수 원-핫 인코딩
        df_encoded = pd.get_dummies(df_clean, columns=["emotion", "face_shape", "genres"])
        corr = df_encoded.corr().round(2)
        import matplotlib
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title("얼굴형/감정/장르 상관관계 히트맵", fontsize=14)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return render_template("correlation.html", correlation_plot=plot_data)
    except Exception as e:
        print(f"Correlation error: {e}")
        return render_template("correlation.html", correlation_plot="")



def create_genre_plot(df):
    try:
        genre_data = []
        for genres_str in df['genres'].dropna():
            if genres_str and len(str(genres_str)) < 1000:
                genre_data.extend([g.strip() for g in str(genres_str).split(',') if len(g.strip()) < 100])
        if not genre_data:
            return ""
        genre_counts = pd.Series(genre_data).value_counts().head(20)
        plt.figure(figsize=(10, 6))
        genre_counts.plot(kind='bar', color='skyblue')
        plt.title('장르별 선호도', fontsize=14)
        plt.xlabel('장르')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Genre plot error: {e}")
        return ""

def create_emotion_plot(df):
    try:
        emotion_genre_data = []
        for _, row in df.iterrows():
            if (pd.notna(row['emotion']) and pd.notna(row['genres']) and 
                len(str(row['genres'])) < 1000):
                genres = [g.strip() for g in str(row['genres']).split(',') if len(g.strip()) < 100]
                for genre in genres[:10]:
                    emotion_genre_data.append({
                        'emotion': str(row['emotion'])[:50],
                        'genre': genre
                    })
        if not emotion_genre_data:
            return ""
        emotion_df = pd.DataFrame(emotion_genre_data)
        crosstab = pd.crosstab(emotion_df['emotion'], emotion_df['genre'])
        if crosstab.shape[0] > 10:
            crosstab = crosstab.head(10)
        if crosstab.shape[1] > 15:
            crosstab = crosstab.iloc[:, :15]
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('감정별 장르 선호도', fontsize=14)
        plt.xlabel('감정')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Emotion plot error: {e}")
        return ""

def create_face_plot(df):
    try:
        face_data = df[df['face_shape'] != 'Unknown']
        if face_data.empty:
            return ""
        face_genre_data = []
        for _, row in face_data.iterrows():
            if (pd.notna(row['face_shape']) and pd.notna(row['genres']) and
                len(str(row['genres'])) < 1000):
                genres = [g.strip() for g in str(row['genres']).split(',') if len(g.strip()) < 100]
                for genre in genres[:10]:
                    face_genre_data.append({
                        'face_shape': str(row['face_shape'])[:50],
                        'genre': genre
                    })
        if not face_genre_data:
            return ""
        face_df = pd.DataFrame(face_genre_data)
        crosstab = pd.crosstab(face_df['face_shape'], face_df['genre'])
        plt.figure(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=True)
        plt.title('얼굴형별 장르 선호도', fontsize=14)
        plt.xlabel('얼굴형')
        plt.ylabel('선택 횟수')
        plt.xticks(rotation=45)
        plt.legend(title='장르', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        return plot_data
    except Exception as e:
        logger.error(f"Face plot error: {e}")
        return ""

@app.route('/recommend')
def recommend():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()
        if df.empty:
            return render_template('recommend.html', 
                                recommendations=[], 
                                insights=[], 
                                total_users=0)
        recommendations = generate_recommendations(df)
        insights = generate_insights(df)
        return render_template('recommend.html',
                            recommendations=recommendations,
                            insights=insights,
                            total_users=len(df))
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return render_template('recommend.html', 
                            recommendations=[], 
                            insights=[], 
                            total_users=0)

def generate_recommendations(df):
    try:
        recommendations = []
        for emotion in df['emotion'].unique():
            if pd.isna(emotion) or len(str(emotion)) > 50:
                continue
            emotion_data = df[df['emotion'] == emotion]
            genre_counts = {}
            for genres_str in emotion_data['genres'].dropna():
                if genres_str and len(str(genres_str)) < 1000:
                    genres = [g.strip() for g in str(genres_str).split(',') if len(g.strip()) < 100]
                    for genre in genres[:10]:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if genre_counts:
                top_genre = max(genre_counts, key=genre_counts.get)
                count = genre_counts[top_genre]
                percentage = (count / len(emotion_data)) * 100
                recommendations.append({
                    'emotion': str(emotion),
                    'genre': str(top_genre),
                    'count': count,
                    'percentage': round(percentage, 1)
                })
        return sorted(recommendations, key=lambda x: x['percentage'], reverse=True)[:20]
    except Exception as e:
        logger.error(f"Recommendation generation error: {e}")
        return []

def generate_insights(df):
    try:
        insights = []
        total_users = len(df)
        if total_users == 0:
            return ["아직 분석된 데이터가 없습니다."]
        insights.append(f"총 {total_users:,}명의 사용자가 분석을 완료했습니다.")
        if not df['emotion'].isna().all():
            emotion_counts = df['emotion'].value_counts()
            if len(emotion_counts) > 0:
                most_common_emotion = emotion_counts.index[0]
                emotion_count = emotion_counts.iloc[0]
                insights.append(f"가장 많이 감지된 감정은 '{most_common_emotion}'입니다. ({emotion_count:,}명)")
        if not df['age'].isna().all():
            valid_ages = df['age'][(df['age'] >= 0) & (df['age'] <= 150)]
            if len(valid_ages) > 0:
                avg_age = valid_ages.mean()
                insights.append(f"사용자 평균 나이는 {avg_age:.1f}세입니다.")
        if not df['gender'].isna().all():
            gender_counts = df['gender'].value_counts()
            if len(gender_counts) > 0:
                top_gender = gender_counts.index[0]
                gender_percentage = (gender_counts.iloc[0] / total_users) * 100
                insights.append(f"사용자의 {gender_percentage:.1f}%가 {top_gender}으로 분석되었습니다.")
        # 얼굴형 인사이트 추가
        if 'face_shape' in df.columns and not df['face_shape'].isna().all():
            face_shape_counts = df['face_shape'].value_counts()
            if len(face_shape_counts) > 0 and face_shape_counts.index[0] != "Unknown":
                most_common_face_shape = face_shape_counts.index[0]
                face_shape_count = face_shape_counts.iloc[0]
                insights.append(
                    f"가장 많이 감지된 얼굴형은 '{most_common_face_shape}'입니다. ({face_shape_count:,}명)"
                )
        # 장르 인사이트 추가
        genre_series = df['genres'].dropna().apply(
            lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
        )
        all_genres = [g for genres in genre_series for g in genres]
        if all_genres:
            top_genre = pd.Series(all_genres).value_counts().index[0]
            top_genre_count = pd.Series(all_genres).value_counts().iloc[0]
            insights.append(f"가장 인기 있는 장르는 '{top_genre}'입니다. ({top_genre_count:,}회 선택)")
        return insights
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return ["데이터 인사이트를 생성하는 중 오류가 발생했습니다."]

if __name__ == '__main__':
    # 최초 실행 시 DB 자동 생성
    init_db()
    app.run(debug=True)
