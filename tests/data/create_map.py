"""
Creae a sample map of assignments using test data.

Requires:
    - pandas
    - folium
"""

import pandas as pd
import folium

if __name__ == "__main__":
    center = (15.0794, 120.6200)
    map = folium.Map(location=center, zoom_start=11)

    enums = pd.read_csv("./test_enum_data.csv")
    targets = pd.read_csv("./test_target_data.csv")

    enum_group = folium.FeatureGroup(name="Enumerators")
    target_group = folium.FeatureGroup(name="Targets")

    for i, row in targets.iterrows():
        folium.Marker(
            [row.gps_lat, row.gps_lon],
            icon=folium.Icon(color="blue"),
            popup=f"Target ID: {row.id}",
        ).add_to(target_group)

    for i, row in enums.iterrows():
        folium.Marker(
            [row.gps_lat, row.gps_lon],
            icon=folium.Icon(color="pink"),
            popup=f"Enumerator ID: {row.id}",
        ).add_to(enum_group)

    map.add_child(target_group)
    map.add_child(enum_group)
    map.add_child(folium.LayerControl())
    map.save("./map.html")
