from flask import Flask, request
import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import dlib
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FILE'] = 'analysis_results.csv'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if not os.path.exists(app.config['DATA_FILE']):
    with open(app.config['DATA_FILE'], mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'filename', 'age', 'gender', 'emotion', 'face_shape', 'eye_distance', 'jaw_ratio', 'eye_size', 'nose_size', 'mouth_size', 'eye_angle', 'genres'])

def get_facial_features(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return (None,) * 8

    shape = predictor(gray, faces[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()])

    eye_distance = np.linalg.norm(coords[36] - coords[45])
    jaw_width = np.linalg.norm(coords[3] - coords[13])
    face_height = np.linalg.norm(coords[8] - coords[27])
    jaw_ratio = jaw_width / face_height if face_height else 0

    left_eye = coords[36:42]
    right_eye = coords[42:48]
    eye_size = (np.mean(np.linalg.norm(left_eye - np.roll(left_eye, -1, axis=0), axis=1)) +
                 np.mean(np.linalg.norm(right_eye - np.roll(right_eye, -1, axis=0), axis=1))) / 2

    nose_width = np.linalg.norm(coords[31] - coords[35])
    nose_height = np.linalg.norm(coords[27] - coords[33])
    nose_size = (nose_width + nose_height) / 2

    mouth_width = np.linalg.norm(coords[48] - coords[54])
    mouth_height = np.linalg.norm(coords[51] - coords[57])
    mouth_size = (mouth_width + mouth_height) / 2

    eye_angle = np.degrees(np.arctan2(coords[45][1] - coords[36][1], coords[45][0] - coords[36][0]))

    if jaw_ratio < 1.2:
        face_shape = "계란형"
    elif jaw_ratio < 1.5:
        face_shape = "둥근형"
    else:
        face_shape = "각진형"

    return face_shape, round(eye_distance, 2), round(jaw_ratio, 2), round(eye_size, 2), round(nose_size, 2), round(mouth_size, 2), round(eye_angle, 2)

@app.route('/')
def index():
    return '''
        <h2>DeepFace 얼굴 분석기 + 장르 설문</h2>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <label>사진 업로드:</label><br>
            <input type="file" name="photo"><br><br>

            <label>좋아하는 장르를 선택하세요:</label><br>
            <input type="checkbox" name="genre" value="코미디"> 코미디<br>
            <input type="checkbox" name="genre" value="공포"> 공포<br>
            <input type="checkbox" name="genre" value="드라마"> 드라마<br>
            <input type="checkbox" name="genre" value="액션"> 액션<br>
            <input type="checkbox" name="genre" value="다큐멘터리"> 다큐멘터리<br><br>

            <input type="submit" value="제출">
        </form>
        <br><a href="/graph">📊 시각화된 결과 보기</a>
        <br><a href="/recommend">🤖 AI 추천 보기</a>
        <br><a href="/admin">📈 관리자용 분석 페이지</a>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    photo = request.files['photo']
    selected_genres = request.form.getlist('genre')

    if photo:
        filename = photo.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)

        try:
            result = DeepFace.analyze(img_path=filepath, actions=['age', 'gender', 'emotion'], enforce_detection=False)[0]
            age = result['age']
            gender = result['gender']
            emotion = result['dominant_emotion']

            face_shape, eye_distance, jaw_ratio, eye_size, nose_size, mouth_size, eye_angle = get_facial_features(filepath)

            with open(app.config['DATA_FILE'], mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), filename, age, gender, emotion,
                    face_shape, eye_distance, jaw_ratio, eye_size, nose_size, mouth_size, eye_angle,
                    '|'.join(selected_genres)
                ])

            return f"""
                <h3>분석 결과:</h3>
                <ul>
                    <li>나이 추정: {age}</li>
                    <li>성별: {gender}</li>
                    <li>감정: {emotion}</li>
                    <li>얼굴형: {face_shape}</li>
                    <li>눈 사이 거리: {eye_distance}</li>
                    <li>턱 비율: {jaw_ratio}</li>
                    <li>눈 크기: {eye_size}</li>
                    <li>코 크기: {nose_size}</li>
                    <li>입 크기: {mouth_size}</li>
                    <li>눈꼬리 각도: {eye_angle}°</li>
                </ul>
                <a href='/'>다시하기</a> | <a href='/graph'>📊 시각화 보기</a> | <a href='/recommend'>🤖 AI 추천 보기</a> | <a href='/admin'>📈 관리자 분석 보기</a>
            """
        except Exception as e:
            return f"<h3>분석 오류: {str(e)}</h3><a href='/'>돌아가기</a>"

    return "<h3>사진 업로드 실패</h3>"

@app.route('/admin')
def admin():
    try:
        df = pd.read_csv(app.config['DATA_FILE'])
        fig, ax = plt.subplots(figsize=(10, 5))
        df['face_shape'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("얼굴형 분포")
        ax.set_ylabel("명 수")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)

        return f'<h3>관리자용 얼굴형 분석 결과</h3><img src="data:image/png;base64,{img_base64}"><br><a href="/">돌아가기</a>'
    except Exception as e:
        return f"<h3>분석 실패: {str(e)}</h3><a href='/'>돌아가기</a>"

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render에서 사용하는 환경변수
    app.run(debug=True, host='0.0.0.0', port=port)


