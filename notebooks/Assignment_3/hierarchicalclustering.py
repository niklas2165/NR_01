import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

# Function to load the dataset  
file_path = '/Users/Niklas/Assignment 1 Streamlit/02/Assignment 2/micro_world_139countries.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

sample_df = df[['economy', 'pop_adult', 'age']].sample(n=15, random_state=42)
sample_df = sample_df.dropna(subset=['pop_adult', 'age']) 

# Performs Hierarchical Clustering 
clustering = AgglomerativeClustering().fit(sample_df[['pop_adult', 'age']].values)

# Calculate the linkage matrix using Average Linkage
linkage_matrix = linkage(sample_df[['pop_adult', 'age']], method='average')

# Create the dendrogram
dendrogram(linkage_matrix, labels=sample_df['economy'].tolist(), orientation='right')

# Add labels and show the plot
plt.xlabel('Average Linkage Distance')
plt.ylabel('Economies')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

#RECOMMENDER SYSTEM
#Step 1: Label Encoding and Matrix Creation
# # A. Label Encoding

le_country_economy = LabelEncoder() #Label Encoder converts categorical variables into numerical representations
le_pop_adult = LabelEncoder()
df['economy'] = le_country_economy.fit_transform(df['economy']) #each unique economy gets a unique integer assigned to it
df['pop_adult'] = le_pop_adult.fit_transform(df['pop_adult'])

# B. Matrix Creation
matrix_country_economy = df.groupby(['economy','pop_adult'])['age'].agg('mean').unstack().fillna(0)
#Rows represent unique economies, Columns unique population amounts
#fillna to fill missing values with 0
#values of matrix represent mean age 

#Step 2: Perform Dimensionality Reduction
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42) #Reduce Dimensionality of the matrix
matrix_country_economy_dr = svd.fit_transform(matrix_country_economy)

#Step 3: Calculate the Similarity Matrix
#Cosine distance used to measure similarity between different economies based on population and mean age.
cosine_distances_matrix_country_economy_dr = cosine_distances(matrix_country_economy)
df.economy.nunique()