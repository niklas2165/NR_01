import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

#Load the dataset
file_path = '/Users/Niklas/Assignment 1 Streamlit/02/Assignment 2/micro_world_139countries.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

sample_df = df[['economy', 'pop_adult', 'age']].sample(n=100, random_state=42)
sample_df = sample_df.dropna(subset=['pop_adult', 'age'])       
X = sample_df[['pop_adult', 'age']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

economy_names = sample_df['economy'].values    

# Step 1: Initialization - Randomly initialize k centroids
k = 4 #number of clusters
random_indices = np.random.choice(len(X), k, replace = False)
centroids_random = X[random_indices]    

euclidean_distances(X, centroids_random) 

# Step 2: Assignment - Assign each of the data points to the nearest centroid using euclidean distance
euclidean_distances_matrix = euclidean_distances(X, centroids_random)
labels_iter1 = np.argmin(euclidean_distances_matrix, axis=1)
#labels_iter1

iteration_visualizations = []
# Capture current state for visualization
iteration_visualizations.append((centroids_random.copy(), labels_iter1.copy()))
#iteration_visualizations[0]

# Step 3: Update Centroids - Recalculate centroids
#Initialise empty list to hold new centroids
new_centroids = []
# Loop through each cluster label from 0 to k-1
for cluster_labels in range(k):
  cluster_points = X[labels_iter1 == cluster_labels]
  new_centroid = np.mean(cluster_points, axis=0)
  new_centroids.append(new_centroid)  # Append each new centroid to the list
# Convert the list of centroids into a NumPy array
new_centroids = np.array(new_centroids)

#Plot 
def plot_clusters(data, centroids, labels, iteration):
    point_indices = data
    """Helper function to plot data points, centroids, and indices."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusting the figure size as needed.
    # Scatter plot for data points
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
    # Scatter plot for centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
    # Annotating each point with its index
    for index, point in enumerate(data):
        ax.text(point[0], point[1], str(index), color='black', fontsize=8)
    # Set title for the subplot
    ax.set_title(f"Iteration {iteration + 1}")


# Capture current state for visualization
iteration_visualizations = []
iteration_visualizations.append((centroids_random.copy(), labels_iter1.copy()))

#Plot Each Iteration 
plot_clusters(X, centroids_random, labels_iter1, 0)
for i in range (5):
  new_centroids = []
  for j in range(k):
    cluster_points = X[labels_iter1 == j]
    centroid = np.mean(cluster_points, axis=0)
    new_centroids.append(centroid)
    
# Step 2
  distances_iter = euclidean_distances(X, np.array(new_centroids))
  labels_iter = np.argmin(distances_iter, axis=1)
  iteration_visualizations.append((new_centroids.copy(), labels_iter.copy()))
  plt.xlabel('Pop Adult')
  plt.ylabel('Age')
  plt.title('KMeans Clustering of Economies')
  plot_clusters(X, np.array(new_centroids), labels_iter, i+1)
  # Check for convergence (might need adjustment if using other than L2 norm

  if np.allclose(iteration_visualizations[i][0], np.array(new_centroids)):
      break


  labels_iter1 = labels_iter.copy()

  #ELBOW METHOD --> OPTIMAL NUMBERS OF CLUSTERS
  # Initialize an empty list to store the within-cluster sum of squares (WCSS)
wcss = []

# Define a range of K values to test
k_values1 = range(1, 11)  # You can adjust the range as needed

# Calculate WCSS for each K value
for k in k_values1:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values
plt.plot(k_values1, wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.show()