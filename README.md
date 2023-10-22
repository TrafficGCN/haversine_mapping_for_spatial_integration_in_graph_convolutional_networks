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

The Haversine formula calculates the shortest distance between two points on the surface of a sphere, given their longitudes and latitudes.

Given two points:

![formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7DP_1(%5Ctext%7Blat%7D_1,%5Ctext%7Blon%7D_1)%5Ctext%7B)

and

![formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7DP_1(%5Ctext%7Blat%7D_2,%5Ctext%7Blon%7D_2)%5Ctext%7B)

The Haversine formula is:

<img src="https://latex.codecogs.com/png.latex?\dpi{150}\begin{align*}a&=\sin^2\left(\frac{\Delta\text{lat}}{2}\right)+\cos(\text{lat}_1)\cdot\cos(\text{lat}_2)\cdot\sin^2\left(\frac{\Delta\text{lon}}{2}\right)\\c&=2\cdot\text{arcsin}\left(\sqrt{a}\right)\\d&=r\cdot%20c\end{align*}" />

<i>r</i> is the radius of the Earth (approximately 6371 kilometers).

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
```
To find the nearest weather sensor for a given traffic sensor, we calculate the distance between the traffic sensor and each weather sensor using the Haversine formula. The weather sensor with the smallest distance is considered the nearest.

Given a traffic sensor 

![formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7DT(%5Ctext%7Blat%7D_T,%5Ctext%7Blon%7D_T))

and a set of weather sensors
![formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7DW=%7BW_1,W_2,%5Cdots,W_n%7D)

, the nearest weather sensor w_k is

![formula](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7DW_k=%5Cunderset%7Bi%7D%7B%5Ctext%7Bargmin%7D%7D%7B%5Ctext%7Bhaversine%7D(%5Ctext%7Blat%7D_T,%5Ctext%7Blon%7D_T,%5Ctext%7Blat%7D_%7BW_i%7D,%5Ctext%7Blon%7D_%7BW_i%7D)%7D)

```
def find_nearest_weather_sensor(traffic_lat, traffic_lon, weather_df):
    distances = weather_df.apply(lambda row: haversine(traffic_lat, traffic_lon, row['lat'], row['long']), axis=1)
    return weather_df.iloc[distances.idxmin()]['detid']

# Map each traffic sensor to its nearest weather sensor
traffic_sensors_df['nearest_weather_sensor'] = traffic_sensors_df.apply(
    lambda row: find_nearest_weather_sensor(row['lat'], row['long'], weather_sensors_df),
    axis=1
)
```
Using the above method, each traffic sensor is mapped to its nearest weather sensor. This results in a dictionary where the keys are traffic sensor IDs and the values are the corresponding nearest weather sensor IDs.

![formula](https://latex.codecogs.com/png.latex?\dpi{150}M:\text{Traffic%20Sensor%20ID}\rightarrow\text{Weather%20Sensor%20ID})

Where <i>M</i> is the mapping function.

```
# Dictionary mapping of traffic sensor to its nearest weather sensor
sensor_to_weather_mapping = dict(zip(traffic_sensors_df['detid'], traffic_sensors_df['nearest_weather_sensor']))
```

The traffic and tempurature are matched via the date timestamps and saved into one csv file. `spatial_integration_haversine_mapping.py`

```
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
...

merged_df.to_csv(merged_file_path, index=False)
```
