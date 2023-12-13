import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./ratings.csv', header=None, names=['movie_id', 'rating', 'movie_date'])

df['rating'] = pd.to_numeric(df['rating'], errors='coerce') 


df = df.dropna(subset=['rating'])


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
title_matrix = tfidf_vectorizer.fit_transform(df['movie_id'])


title_distances = pairwise_distances(title_matrix, metric='cosine')


scaler = StandardScaler()
ratings = scaler.fit_transform(df[['rating']])


df['release_year'] = pd.to_datetime(df['movie_date'], dayfirst=True, errors='coerce').dt.year


features = pd.DataFrame(index=df.index)
features['title_distance'] = title_distances[:, 0] 


features['ratings'] = ratings
features['release_year'] = df['release_year']


k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(features)


print(df[['movie_id', 'rating', 'movie_date', 'release_year', 'cluster']].head())


plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='rating', data=df)
plt.title('Distribution of Ratings within Clusters')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df)
plt.title('Number of Movies in Each Cluster')
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(x='release_year', hue='cluster', data=df, multiple="stack", bins=20)
plt.title('Release Year Trends in Clusters')
plt.show()


df.to_csv('clustered_movies.csv', index=False)
print(df.columns)
