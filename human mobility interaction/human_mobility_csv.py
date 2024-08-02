import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Coordinates
home_locations = {
    1: {'interest': 'Gym', 'latitude': 21.0978421648822, 'longitude': 79.1654240021961},  # Bahadura
    2: {'interest': 'Coding', 'latitude': 21.116632988644838, 'longitude': 79.04375565378935},  # Sambhaji Nagar
    3: {'interest': 'Badminton', 'latitude': 21.202096780577005, 'longitude': 79.06365853487708},  # Zhingabai Takli
    4: {'interest': 'Singing', 'latitude': 21.122951179695665, 'longitude': 79.06388126700782},  # Laxmi Nagar
    5: {'interest': 'Guitar', 'latitude': 21.118096097430136, 'longitude': 79.04441307790263},  # Trimurti Nagar
}
work_location = {'latitude': 21.0596546918643, 'longitude': 79.01719338348013}  # Mihan

# Major spots (malls, cafes)
major_spots = {
    'Empress Mall': {'latitude': 21.1381, 'longitude': 79.0747},
    'Eternity Mall': {'latitude': 21.1450, 'longitude': 79.0831},
    'Café Coffee Day': {'latitude': 21.1210, 'longitude': 79.0480},
    'Famous Place 1': {'latitude': 21.1100, 'longitude': 79.0600},
    'Famous Place 2': {'latitude': 21.1150, 'longitude': 79.0850},
}

# Generate mock data
np.random.seed(42)
num_entries = 1000
user_ids = np.random.choice(list(home_locations.keys()), num_entries)
timestamps = pd.date_range(start='2024-01-01', periods=num_entries, freq='h')

# Assign home and work locations, and random movement
latitude = []
longitude = []
interests = []

for user_id in user_ids:
    if np.random.rand() < 0.5:
        # 50% chance to be at work
        latitude.append(work_location['latitude'])
        longitude.append(work_location['longitude'])
    else:
        # 50% chance to be at home or random place
        if np.random.rand() < 0.7:
            # 70% chance to be at home
            latitude.append(home_locations[user_id]['latitude'])
            longitude.append(home_locations[user_id]['longitude'])
        else:
            # 30% chance to be at a random location (e.g., malls, cafes)
            latitude.append(np.random.uniform(21.10, 21.20))  # Approximate Nagpur lat range
            longitude.append(np.random.uniform(79.00, 79.10))  # Approximate Nagpur long range

    interests.append(home_locations[user_id]['interest'])

data = {
    'user_id': user_ids,
    'timestamp': timestamps,
    'latitude': latitude,
    'longitude': longitude,
    'interest': interests
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('human_mobility_data.csv', index=False)

print("CSV file 'human_mobility_data.csv' generated successfully.")

# Clustering using DBSCAN
dbscan = DBSCAN(eps=0.01, min_samples=5)
df['location_cluster'] = dbscan.fit_predict(df[['latitude', 'longitude']])

# Save the clustered data
df.to_csv('human_mobility_clustered.csv', index=False)

# Dynamic color assignment for clusters
unique_clusters = df['location_cluster'].unique()
colors = {cluster: f'#{random.randint(0, 0xFFFFFF):06x}' for cluster in unique_clusters}

# Visualization with Folium
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# Color for each user
user_colors = {user_id: f'#{random.randint(0, 0xFFFFFF):06x}' for user_id in home_locations.keys()}

# Add routes to the map
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id]
    locations = list(zip(user_data['latitude'], user_data['longitude']))
    folium.PolyLine(
        locations,
        color=user_colors[user_id],
        weight=2.5,
        opacity=0.7,
        popup=f"User: {user_id} | Route"
    ).add_to(m)

# Add points to the map
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=colors[row['location_cluster']],
        fill=True,
        fill_color=colors[row['location_cluster']],
        popup=f"User: {row['user_id']}<br>Interest: {row['interest']}<br>Cluster: {row['location_cluster']}"
    ).add_to(marker_cluster)

# Add major spots to the map
for spot, coords in major_spots.items():
    folium.Marker(
        location=[coords['latitude'], coords['longitude']],
        popup=spot,
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

# Add legend to map
legend_html = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 300px; height: 200px;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;">
     &nbsp; <b>Legend</b> <br>
'''
for cluster, color in colors.items():
    legend_html += f'&nbsp; <i class="fa fa-circle" style="color:{color}"></i> Cluster {cluster} <br>'
for user_id, color in user_colors.items():
    legend_html += f'&nbsp; <i class="fa fa-circle" style="color:{color}"></i> User {user_id} <br>'
legend_html += '</div>'

m.get_root().html.add_child(folium.Element(legend_html))

# Add a potential metro line
metro_line_coords = [
    [21.0978421648822, 79.1654240021961],  # Bahadura
    [21.0596546918643, 79.01719338348013]   # Mihan
]
folium.PolyLine(
    metro_line_coords,
    color='blue',
    weight=5,
    opacity=0.8,
    tooltip='Potential Metro Line from Bahadura to MIHAN'
).add_to(m)

# Save the map to an HTML file
m.save('human_mobility_map_with_routes.html')

print("Interactive map 'human_mobility_map_with_routes.html' generated successfully.")

# Plotting footfalls at interest locations
plt.figure(figsize=(12, 7))
sns.countplot(data=df, x='interest', palette='viridis')
plt.title('Footfalls at Interest Locations')
plt.xlabel('Interest')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('footfalls_interest_locations.png')
plt.show()

# Plotting user demographics
plt.figure(figsize=(12, 7))
sns.countplot(data=df, x='user_id', palette='viridis')
plt.title('User Demographics')
plt.xlabel('User ID')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('user_demographics.png')
plt.show()

# Plotting user activities over time
plt.figure(figsize=(14, 8))
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id]
    user_data.set_index('timestamp').resample('W').count()['user_id'].plot(label=f'User {user_id}', linewidth=2.5)
plt.title('User Activities Over Time')
plt.xlabel('Date')
plt.ylabel('Activity Count')
plt.legend()
plt.tight_layout()
plt.savefig('user_activities_over_time.png')
plt.show()

# Plotting DBSCAN clusters
plt.figure(figsize=(12, 7))
plt.scatter(df['latitude'], df['longitude'], c=df['location_cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='DBSCAN Cluster ID')
plt.title('DBSCAN Clustering of Human Mobility Data')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.tight_layout()
plt.savefig('dbscan_clusters.png')
plt.show()

print("Graphs generated successfully.")
