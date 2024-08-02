import pandas as pd
# Determining work location
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('human_mobility_data.csv')

# Assuming home and work locations have more data points
kmeans = KMeans(n_clusters=2, random_state=42)
df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Save the clustered data
df.to_csv('human_mobility_clustered.csv', index=False)