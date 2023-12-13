import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Netflix Prize dataset (assuming it has no column titles)
df = pd.read_csv('./ratings.csv', header=None, names=['movie_id', 'rating', 'movie_date'])

# Clean 'rating' column
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # convert to numeric, coerce errors to NaN

# Drop rows with NaN values in the 'rating' column
df = df.dropna(subset=['rating'])

# Feature Engineering
# TF-IDF for movie titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
title_matrix = tfidf_vectorizer.fit_transform(df['movie_id'])

# Edit distance for movie titles
title_distances = pairwise_distances(title_matrix, metric='cosine')

# Standardize ratings
scaler = StandardScaler()
ratings = scaler.fit_transform(df[['rating']])

# Release date as a feature (detecting date format)
df['release_year'] = pd.to_datetime(df['movie_date'], dayfirst=True, errors='coerce').dt.year

# Combine features
features = pd.DataFrame(index=df.index)
features['title_distance'] = title_distances[:, 0]  # Assuming you want to use the first column of title_distances
features['ratings'] = ratings
features['release_year'] = df['release_year']

# Model Training
# Choose the number of clusters (K) based on your analysis
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(features)
# Display the first few rows of the DataFrame with cluster results
print(df[['movie_id', 'rating', 'movie_date', 'release_year', 'cluster']].head())

# Analysis and Visualization
# Visualize the clusters or perform further analysis as needed
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='rating', data=df)
plt.title('Distribution of Ratings within Clusters')
plt.show()

# Movie Count in Each Cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=df)
plt.title('Number of Movies in Each Cluster')
plt.show()

# Release Year Trends
plt.figure(figsize=(12, 6))
sns.histplot(x='release_year', hue='cluster', data=df, multiple="stack", bins=20)
plt.title('Release Year Trends in Clusters')
plt.show() 

# Save the results to a new CSV file
df.to_csv('clustered_movies.csv', index=False)
print(df.columns)
