import folium
import numpy as np
import pandas as pd
import os

# Load the datasets (assuming they are in the same directory as the script)
OS_PATH = os.path.dirname(os.path.realpath('__file__'))

weather_sensors_df = pd.read_csv(os.path.join(OS_PATH, 'data/metr-la/sensors/metr_la_sensors_weather.csv'))
traffic_sensors_df = pd.read_csv(os.path.join(OS_PATH, 'data/metr-la/sensors/metr_la_sensors_traffic.csv'))

traffic_speed_df = pd.read_csv(os.path.join(OS_PATH, 'data/metr-la/traffic/speed.csv'))
air_temp_df = pd.read_csv(os.path.join(OS_PATH, 'data/metr-la/weather/air_temp_set_1_fahrenheit.csv'))

# Haversine formula to calculate the distance between two geographical points
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of Earth in kilometers
    return c * r

def find_nearest_weather_sensor(traffic_lat, traffic_lon, weather_df):
    distances = weather_df.apply(lambda row: haversine(traffic_lat, traffic_lon, row['lat'], row['long']), axis=1)
    return weather_df.iloc[distances.idxmin()]['detid']

# Map each traffic sensor to its nearest weather sensor
traffic_sensors_df['nearest_weather_sensor'] = traffic_sensors_df.apply(
    lambda row: find_nearest_weather_sensor(row['lat'], row['long'], weather_sensors_df),
    axis=1
)

# Dictionary mapping of traffic sensor to its nearest weather sensor
sensor_to_weather_mapping = dict(zip(traffic_sensors_df['detid'], traffic_sensors_df['nearest_weather_sensor']))

def calculate_centroid(df):
    """
    Calculate the centroid (geometric center) of a set of points.
    
    Parameters:
    - df: DataFrame containing lat and long columns
    
    Returns:
    - A tuple (centroid_lat, centroid_lon).
    """
    centroid_lat = df['lat'].mean()
    centroid_lon = df['long'].mean()
    
    return centroid_lat, centroid_lon

def generate_sensor_map(traffic_sensors_df, weather_sensors_df, sensor_to_weather_mapping):
    """
    Generate a map showing the traffic sensors and weather sensors.
    
    Parameters:
    - traffic_sensors_df: DataFrame containing traffic sensor locations.
    - weather_sensors_df: DataFrame containing weather sensor locations.
    - sensor_to_weather_mapping: Dictionary mapping traffic sensors to their nearest weather sensor.
    
    Returns:
    - A folium map object.
    """

    centroid_lat, centroid_lon = calculate_centroid(traffic_sensors_df)
    
    # Create a base map
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=12)  # Centered around San Francisco

    # Define distinct colors for weather sensors
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray'
    ]

    # Create a mapping from weather sensor to color
    weather_sensor_to_color = {detid: colors[i % len(colors)] for i, detid in enumerate(weather_sensors_df['detid'].unique())}

    # Plot the weather sensors on the map
    for idx, row in weather_sensors_df.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['long']),
            radius=8,
            color=weather_sensor_to_color[row['detid']],
            fill=True,
            fill_color=weather_sensor_to_color[row['detid']]
        ).add_to(m)

    # Plot the traffic sensors on the map
    for idx, row in traffic_sensors_df.iterrows():
        nearest_weather_sensor = sensor_to_weather_mapping[row['detid']]
        folium.CircleMarker(
            location=(row['lat'], row['long']),
            radius=4,
            color=weather_sensor_to_color[nearest_weather_sensor],
            fill=True,
            fill_color=weather_sensor_to_color[nearest_weather_sensor]
        ).add_to(m)

    return m

# Generate the map using the function
# Save the merged dataframe to a CSV file
output_folder = os.path.join(OS_PATH, 'output/metr-la')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

map_output = generate_sensor_map(traffic_sensors_df, weather_sensors_df, sensor_to_weather_mapping)
map_output.save(os.path.join(output_folder, 'sensor_map.html'))

