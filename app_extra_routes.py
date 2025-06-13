
@app.route('/correlation')
def correlation():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        conn.close()

        # 필요한 열만 남기고, 결측치 제거
        cols = ["age", "emotion", "face_shape", "genres"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df_clean = df[cols].dropna()

        if df_clean.empty:
            return render_template("correlation.html", correlation_plot="")

        # 범주형 변수 원-핫 인코딩
        df_encoded = pd.get_dummies(df_clean, columns=["emotion", "face_shape", "genres"])

        # 상관계수 계산
        corr = df_encoded.corr().round(2)

        # 한글 폰트 설정
        import matplotlib
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 히트맵 그리기
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
