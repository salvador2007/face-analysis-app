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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FILE'] = 'analysis_results.csv'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# CSV 파일 초기화
if not os.path.exists(app.config['DATA_FILE']):
    with open(app.config['DATA_FILE'], mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'filename', 'age', 'gender', 'emotion', 'confidence', 'genres'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'photo' not in request.files:
            flash('사진을 선택해주세요', 'error')
            return redirect(url_for('index'))

        file = request.files['photo']
        selected_genres = request.form.getlist('genre')

        if file.filename == '':
            flash('파일이 선택되지 않았습니다', 'error')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            flash('지원하지 않는 파일 형식입니다', 'error')
            return redirect(url_for('index'))

        # 안전한 파일명 생성
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        logger.info(f"파일 저장됨: {filepath}")

        # DeepFace 분석
        result = DeepFace.analyze(
            img_path=filepath, 
            actions=['age', 'gender', 'emotion'], 
            enforce_detection=False
        )[0]
        
        age = result['age']
        gender = result['gender']
        emotion = result['dominant_emotion']
        
        # 신뢰도 계산
        emotion_scores = result['emotion']
        confidence = max(emotion_scores.values())
        
        # 성별 처리
        if isinstance(gender, dict):
            gender_label = max(gender.items(), key=lambda x: x[1])[0]
            gender_confidence = max(gender.values())
        else:
            gender_label = str(gender)
            gender_confidence = 0

        # CSV에 저장
        with open(app.config['DATA_FILE'], mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), 
                unique_filename, 
                age, 
                f"{gender_label}({gender_confidence:.1f}%)",
                emotion,
                f"{confidence:.1f}%",
                '|'.join(selected_genres)
            ])

        # 파일 정리
        if os.path.exists(filepath):
            os.remove(filepath)

        return render_template('result.html', 
                             age=age,
                             gender=gender_label,
                             gender_confidence=gender_confidence,
                             emotion=emotion,
                             confidence=confidence,
                             emotion_scores=emotion_scores,
                             selected_genres=selected_genres)
        
    except Exception as e:
        logger.error(f"분석 오류: {str(e)}")
        flash(f'분석 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/graph')
def graph():
    try:
        if not os.path.exists(app.config['DATA_FILE']):
            flash('분석된 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))
            
        df = pd.read_csv(app.config['DATA_FILE'])
        
        if df.empty:
            flash('분석된 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))

        # 그래프 생성
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 사용자 분석 대시보드', fontsize=16, fontweight='bold')
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 1. 감정 분포
        if 'emotion' in df.columns and not df['emotion'].empty:
            emotion_counts = df['emotion'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
            emotion_counts.plot(kind='bar', ax=axes[0,0], color=colors[:len(emotion_counts)])
            axes[0,0].set_title('Emotion Distribution', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Count')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].grid(axis='y', alpha=0.3)

        # 2. 나이 분포
        if 'age' in df.columns and not df['age'].empty:
            ages = pd.to_numeric(df['age'], errors='coerce').dropna()
            axes[0,1].hist(ages, bins=15, color='#74B9FF', alpha=0.8, edgecolor='black')
            axes[0,1].set_title('Age Distribution', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Age')
            axes[0,1].set_ylabel('Count')
            axes[0,1].grid(axis='y', alpha=0.3)
            axes[0,1].axvline(ages.mean(), color='red', linestyle='--', 
                            label=f'Average: {ages.mean():.1f}')
            axes[0,1].legend()

        # 3. 장르 선호도
        if 'genres' in df.columns:
            all_genres = []
            for genres in df['genres'].dropna():
                if genres and genres != 'nan':
                    all_genres.extend(genres.split('|'))
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                colors = ['#FD79A8', '#FDCB6E', '#6C5CE7', '#A29BFE', '#00B894']
                wedges, texts, autotexts = axes[1,0].pie(genre_counts.values, 
                                                        labels=genre_counts.index,
                                                        autopct='%1.1f%%',
                                                        colors=colors[:len(genre_counts)],
                                                        startangle=90)
                axes[1,0].set_title('Genre Preferences', fontsize=14, fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

        # 4. 성별 및 감정 교차분석
        if 'gender' in df.columns and 'emotion' in df.columns:
            gender_clean = []
            for gender in df['gender']:
                if isinstance(gender, str):
                    if 'Woman' in gender:
                        gender_clean.append('Female')
                    elif 'Man' in gender:
                        gender_clean.append('Male')
                    else:
                        gender_clean.append('Unknown')
                else:
                    gender_clean.append('Unknown')
            
            df_temp = pd.DataFrame({
                'gender': gender_clean,
                'emotion': df['emotion']
            })
            
            crosstab = pd.crosstab(df_temp['gender'], df_temp['emotion'])
            if not crosstab.empty:
                crosstab.plot(kind='bar', ax=axes[1,1], stacked=True, 
                            colormap='Set3', alpha=0.8)
                axes[1,1].set_title('Gender-Emotion Distribution', fontsize=14, fontweight='bold')
                axes[1,1].set_ylabel('Count')
                axes[1,1].tick_params(axis='x', rotation=0)
                axes[1,1].legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1,1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        
        # 이미지로 변환
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)

        # 통계 정보 계산
        stats = {
            'total_users': len(df),
            'avg_age': df['age'].mean() if 'age' in df.columns else 0,
            'most_common_emotion': df['emotion'].mode().iloc[0] if 'emotion' in df.columns and not df['emotion'].empty else 'N/A',
            'date_range': f"{df['timestamp'].min()[:10]} ~ {df['timestamp'].max()[:10]}" if 'timestamp' in df.columns else 'N/A'
        }

        return render_template('graph.html', 
                             graph_data=img_base64,
                             stats=stats)
        
    except Exception as e:
        logger.error(f"그래프 생성 오류: {str(e)}")
        flash(f'그래프 생성 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/recommend')
def recommend():
    try:
        if not os.path.exists(app.config['DATA_FILE']):
            flash('추천을 위한 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))
            
        df = pd.read_csv(app.config['DATA_FILE'])
        
        if df.empty:
            flash('추천을 위한 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))

        recommendations = []
        insights = []
        
        # 감정별 장르 선호도 분석
        emotion_genre_map = {}
        for _, row in df.iterrows():
            emotion = row['emotion']
            genres = str(row['genres']).split('|') if pd.notna(row['genres']) else []
            
            if emotion not in emotion_genre_map:
                emotion_genre_map[emotion] = {}
            
            for genre in genres:
                if genre and genre.strip():
                    emotion_genre_map[emotion][genre] = emotion_genre_map[emotion].get(genre, 0) + 1

        # 추천 생성
        for emotion, genres in emotion_genre_map.items():
            if genres:
                sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
                top_genre = sorted_genres[0]
                recommendations.append({
                    'emotion': emotion,
                    'genre': top_genre[0],
                    'count': top_genre[1],
                    'percentage': (top_genre[1] / sum(genres.values())) * 100
                })

        # 인사이트 생성
        if len(df) > 0:
            avg_age = df['age'].mean() if 'age' in df.columns else 0
            most_emotion = df['emotion'].mode().iloc[0] if 'emotion' in df.columns and not df['emotion'].empty else 'unknown'
            
            insights.append(f"평균 연령대는 {avg_age:.1f}세입니다")
            insights.append(f"가장 많이 감지된 감정은 '{most_emotion}'입니다")
            
            # 장르 분석
            all_genres = []
            for genres in df['genres'].dropna():
                if genres and genres != 'nan':
                    all_genres.extend(genres.split('|'))
            
            if all_genres:
                top_genre = pd.Series(all_genres).mode().iloc[0]
                insights.append(f"가장 인기 있는 장르는 '{top_genre}'입니다")

        return render_template('recommend.html', 
                             recommendations=recommendations,
                             insights=insights,
                             total_users=len(df))
        
    except Exception as e:
        logger.error(f"추천 생성 오류: {str(e)}")
        flash(f'추천 생성 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/admin')
def admin():
    try:
        if not os.path.exists(app.config['DATA_FILE']):
            flash('관리할 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))
            
        df = pd.read_csv(app.config['DATA_FILE'])
        
        if df.empty:
            flash('관리할 데이터가 없습니다', 'warning')
            return redirect(url_for('index'))

        # 최근 데이터 (최대 10개)
        recent_data = df.tail(10).to_dict('records')
        
        # 전체 통계
        stats = {
            'total_users': len(df),
            'avg_age': f"{df['age'].mean():.1f}세" if 'age' in df.columns else 'N/A',
            'age_range': f"{df['age'].min():.0f}~{df['age'].max():.0f}세" if 'age' in df.columns else 'N/A',
            'most_emotion': df['emotion'].mode().iloc[0] if 'emotion' in df.columns and not df['emotion'].empty else 'N/A'
        }

        return render_template('admin.html', 
                             recent_data=recent_data,
                             stats=stats)
        
    except Exception as e:
        logger.error(f"관리자 페이지 오류: {str(e)}")
        flash(f'관리자 페이지 로드 중 오류가 발생했습니다: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/stats')
def api_stats():
    """API 엔드포인트 - 통계 데이터 JSON으로 반환"""
    try:
        if not os.path.exists(app.config['DATA_FILE']):
            return jsonify({'error': 'No data available'}), 404
            
        df = pd.read_csv(app.config['DATA_FILE'])
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404

        stats = {
            'total_users': len(df),
            'avg_age': float(df['age'].mean()) if 'age' in df.columns else 0,
            'emotions': df['emotion'].value_counts().to_dict() if 'emotion' in df.columns else {},
            'last_updated': datetime.now().isoformat()
        }

        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(413)
def too_large(e):
    flash('파일이 너무 큽니다 (최대 16MB)', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    flash('서버 내부 오류가 발생했습니다', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)