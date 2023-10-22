# Haversine Mapping for Spatial Integration in Graph Convolutional Networks
Calculating the nearest weather sensor for each traffic sensor and then merging the weather sensors' temporal data with the traffic sensors' using the Haversine Formula.

### Citations

1. For the Los Angeles metr-la and Santa Clara pems-bay datasets cite: Kwak, Semin. (2020). PEMS-BAY and METR-LA in csv [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5146275

2. Weather sensors are provided by University of Utah Department of Atmospheric Sciences https://mesowest.utah.edu/

3. Bicycle sensors from the City of Munich Opendata Portal: https://opendata.muenchen.de/dataset/raddauerzaehlstellen-muenchen/resource/211e882d-fadd-468a-bf8a-0014ae65a393?view_id=11a47d6c-0bc1-4bfa-93ea-126089b59c3d

4. OpenStreetMap https://www.openstreetmap.org/ must also be referenced because the matrices where calculated using OpenStreetMap.
5. If you use any of the Maps you must reference both OpenStreetMap and Mapbox https://www.mapbox.com/.

### Introduction
Alright so you have traffic data already for your temporal graph convolutional network but would like to incorporate additional data such as weather measurements collected by additional weather sensors. Using the Haversine Formula we can find the nearest weather sensor for each traffic sensor. This data is then appended and matched via the timestamps to and exported in merged_speed_traffic_and_air_temperature_data.csv file.

merged_speed_traffic_and_air_temperature_data.csv
```
DATETIMESTAMP,773869_speed,767541_speed,767542_speed, ... ,773869_temp,767541_temp,767542_temp
2012-03-01 00:00:00,64.375,67.625,67.125,61.5,66.875, ... ,nan, 51.8, 51.8
```
Now you just need to forwards fill the nan values and normalize the data for T-GCN model and feed it in.

<img src="https://github.com/ThomasAFink/haversine_mapping_for_spatial_integration_in_graph_convolutional_networks/assets/53316058/5b846c74-7bdb-4962-b00b-1451dccab64c" width="48%">
<img src="https://github.com/ThomasAFink/haversine_mapping_for_spatial_integration_in_graph_convolutional_networks/assets/53316058/4b71cec2-e95e-43d4-b798-3969dfc8956d" width="48%">
<br />
<br />
<br />
Traffic sensors are computed to the nearest weather sensor for the pems-bay and metr-la datasets.

### Repo Structure

A brief file structure overview of the repository is provided.
```
/
map_spatial_integration_haversine_mapping.py
spatial_integration_haversine_mapping.py

- / data / metr-la /
         - /sensors
            metr_la_sensors_traffic.csv
            metr_la_sensors_weather.csv 
         - /traffic
            speed.csv
            ...
         - /weather
            air_temp_set_1_fahrenheit.csv
            ...
  
- / output / metr-la /
            merged_speed_traffic_and_air_temperature_data.csv
            sensor_map.html
```

### Prerequisites
Before jumping into the code the following requirements and packages are needed to run the code:

```
Python 3.10.6
pip3 install pandas
pip3 install numpy
pip3 install folium

```

First the packages that were just installed are imported into our file adjacency_matrix.py

```
import numpy as np
import pandas as pd
import os

```

### Code

```
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

```
