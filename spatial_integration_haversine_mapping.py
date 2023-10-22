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

# Function to merge data based on timestamps
def merge_data_on_timestamps(traffic_speed_df, air_temp_df, sensor_to_weather_mapping):
    # Initialize merged dataframe with DATETIMESTAMP column
    merged_df = pd.DataFrame()
    merged_df["DATETIMESTAMP"] = traffic_speed_df["DATETIMESTAMP"]
    
    # Iterate through each sensor in the traffic_speed_df
    for sensor in traffic_speed_df.columns[1:]:
        # Copy the speed data
        merged_df[sensor] = traffic_speed_df[sensor]
        
        # Find corresponding weather sensor
        weather_sensor = sensor_to_weather_mapping[int(sensor)]
        
        # Find corresponding temperature data
        if f"{weather_sensor}" in air_temp_df.columns:
            merged_df[f"{sensor}_temp"] = air_temp_df[f"{weather_sensor}"].reindex_like(traffic_speed_df)
    
    return merged_df

# Reorder columns function
def reorder_columns(merged_df):
    # Create a list for the reordered columns
    speed_cols = [col for col in merged_df.columns if '_temp' not in col]
    temp_cols = [col for col in merged_df.columns if '_temp' in col]
    # Reorder the columns
    final_order = speed_cols + temp_cols
    merged_df = merged_df[final_order]
    return merged_df

# Rename speed columns with "_speed" suffix
def rename_speed_columns(merged_df):
    speed_cols = [col for col in merged_df.columns if '_temp' not in col and col != "DATETIMESTAMP"]
    speed_columns_renamed = {col: f"{col}_speed" for col in speed_cols}
    merged_df.rename(columns=speed_columns_renamed, inplace=True)
    return merged_df

# Merge, reorder, rename and save the dataframe
merged_df = merge_data_on_timestamps(traffic_speed_df, air_temp_df, sensor_to_weather_mapping)
merged_df = reorder_columns(merged_df)
merged_df = rename_speed_columns(merged_df)

# Forward fill and then backward fill NaN values
merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

# Save the merged dataframe to a CSV file
output_folder = os.path.join(OS_PATH, 'output/metr-la')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
merged_file_path = os.path.join(output_folder, 'merged_speed_traffic_and_air_temperature_data.csv')
merged_df.to_csv(merged_file_path, index=False)
