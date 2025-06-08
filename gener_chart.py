import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('analysis_results.csv')

all_genres = []
for g in df['genres'].dropna():
    all_genres.extend(g.split('|'))

genre_counts = Counter(all_genres)

plt.figure(figsize=(8,5))
plt.bar(genre_counts.keys(), genre_counts.values(), color='skyblue')
plt.title('사용자들이 선택한 장르 분포')
plt.xlabel('장르')
plt.ylabel('선택 횟수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
