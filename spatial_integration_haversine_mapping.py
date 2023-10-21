import numpy as np
import pandas as pd
import os

# Load the datasets (assuming they are in the same directory as the script)
OS_PATH = os.path.dirname(os.path.realpath('__file__'))
weather_sensors_df = os.path.join(OS_PATH, 'data/pems-bay/sensors/pems_bay_sensors_weather.csv')
traffic_sensors_df = os.path.join(OS_PATH, 'data/pems-bay/sensors/pems_bay_sensors_traffic.csv')


traffic_speed_df = os.path.join(OS_PATH, 'data/pems-bay/traffic/speed.csv')
air_temp_df = os.path.join(OS_PATH, 'data/pems-bay/weather/air_temp_set_1_fahrenheit.csv')


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

# Merge weather data with traffic data in chunks
chunk_size = 5000

def merge_weather_with_traffic(traffic_data, weather_data, sensor_to_weather_mapping):
    merged_data = traffic_data[['DATETIMESTAMP']].copy()
    for sensor in traffic_data.columns:
        if sensor != "DATETIMESTAMP":
            nearest_weather_sensor = sensor_to_weather_mapping.get(sensor, None)
            if nearest_weather_sensor:
                weather_chunk = weather_data[['Date_Time', nearest_weather_sensor]].rename(columns={nearest_weather_sensor: sensor})
                merged_data = merged_data.merge(weather_chunk, left_on="DATETIMESTAMP", right_on="Date_Time", how="left").drop(columns="Date_Time")
    return merged_data

# Example of merging air temperature data with traffic data in chunks
merged_chunks = []
for start_row in range(0, traffic_speed_df.shape[0], chunk_size):
    end_row = start_row + chunk_size
    traffic_chunk = traffic_speed_df.iloc[start_row:end_row]
    merged_chunk = merge_weather_with_traffic(traffic_chunk, air_temp_df, sensor_to_weather_mapping)
    merged_chunks.append(merged_chunk)


final_merged_df = pd.concat(merged_chunks, axis=0)

# Define output path and save the final merged dataframe
output_folder = os.path.join(OS_PATH, 'output')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
output_file_path = os.path.join(output_folder, 'merged_traffic_weather_data.csv')
final_merged_df.to_csv(output_file_path, index=False)