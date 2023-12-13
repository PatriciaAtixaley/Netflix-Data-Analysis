import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load data from CSV
#file_path = './movie_titles.csv'  # Replace with the actual path
file_path = './training_set/movie_titles.txt'  # Replace with the actual path
movies_df = pd.read_csv(file_path)

# Impute missing values (replace NaN with the mean of the column)
imputer = SimpleImputer(strategy='mean')
movies_df['Year'] = imputer.fit_transform(movies_df[['Year']])

# Feature extraction
movies_df['Title_Length'] = movies_df['Title'].apply(len)

# TF-IDF Vectorization for movie titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Title'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine features
features = ['Year', 'Title_Length']
movies_features = pd.concat([movies_df[features], tfidf_df], axis=1)

# Normalize features
scaler = StandardScaler()
movies_scaled = scaler.fit_transform(movies_features)

# K-means clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
movies_df['Cluster'] = kmeans.fit_predict(movies_scaled)

# View the resulting clusters
print(movies_df[['Title', 'Cluster']])

# Specify the path where you want to save the new CSV file
output_path = './movie_clustering.csv'  # Replace with the desired path

# Save the DataFrame with clustering results to the new CSV file
movies_df.to_csv(output_path, index=False)

print(f"Clustering results saved to {output_path}")

# Count the number of movies in each cluster
cluster_counts = movies_df['Cluster'].value_counts()

# Display the cluster distribution
print(cluster_counts)

# Display mean values for each feature within each cluster
cluster_means = movies_df.groupby('Cluster').mean()
print(cluster_means)

# Visualize feature distributions within each cluster
for cluster in range(num_clusters):
    plt.figure(figsize=(10, 6))
    plt.title(f'Cluster {cluster} Feature Distributions')

    for feature in features:
        plt.hist(movies_df[movies_df['Cluster'] == cluster][feature], alpha=0.5, label=feature)

    plt.legend()
    plt.show()
