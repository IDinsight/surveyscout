import folium
from typing import Tuple
import pandas as pd

from surveyscout.utils import LocationDataset


def compute_center(
    enum_locations: LocationDataset, target_locations: LocationDataset
) -> Tuple[float, float]:
    """Compute center coordinate amongst enumerators and targets"""
    all_lats = (
        enum_locations.get_gps_coords()[:, 0].tolist()
        + target_locations.get_gps_coords()[:, 0].tolist()
    )
    all_lons = (
        enum_locations.get_gps_coords()[:, 1].tolist()
        + target_locations.get_gps_coords()[:, 1].tolist()
    )

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    center = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
    return center


def plot_enum_targets(
    enum_locations: LocationDataset, target_locations: LocationDataset
) -> folium.Map:
    """Plot enumerator and target locations on the map"""
    map = folium.Map(location=compute_center(enum_locations, target_locations))

    enum_group = folium.FeatureGroup(name="Enumerators")
    target_group = folium.FeatureGroup(name="Targets")

    for target_id, target_coords in zip(
        target_locations.get_ids(), target_locations.get_gps_coords()
    ):
        folium.CircleMarker(
            target_coords,
            radius=3,
            color="#1D53A6",
            fill=True,
            fill_color="#1D53A6",
            fill_opacity=1.0,
            stroke=False,
            popup=f"Target ID: {target_id}",
        ).add_to(target_group)

    for enum_id, enum_coords in zip(
        enum_locations.get_ids(), enum_locations.get_gps_coords()
    ):
        folium.Marker(
            enum_coords,
            icon=folium.Icon(color="orange", icon="user"),
            popup=f"Enumerator ID: {enum_id}",
        ).add_to(enum_group)

    map.add_child(target_group)
    map.add_child(enum_group)
    map.add_child(folium.LayerControl())
    map.fit_bounds(map.get_bounds())
    return map


def plot_assignments(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    assignments: pd.DataFrame,
) -> folium.Map:
    """Plot assignments of targets to enumerators on the map"""
    # create a map
    map = folium.Map(location=compute_center(enum_locations, target_locations))

    # create a color map
    # TODO: automatically generate colors
    colors = [
        # available Folium Icon colors
        "darkgreen",
        "pink",
        "red",
        "darkblue",
        "orange",
        "lightgray",
        "darkpurple",
        "cadetblue",
        "blue",
        "black",
        "lightgreen",
        "beige",
        "purple",
        "lightred",
        "darkred",
        "green",
        "white",
        "lightblue",
        "gray",
    ]
    if len(enum_locations) > len(colors):
        raise ValueError(
            f"Too many enumerators to plot: {len(enum_locations)}. Currently, "
            f"we do not support plotting more than {len(colors)} enumerators."
        )

    enum_ids = enum_locations.get_ids()
    color_map = dict(zip(enum_ids, [color for color in colors]))

    # Create map group for each enumerator
    groups = {enum_id: folium.FeatureGroup(name=enum_id) for enum_id in enum_ids}

    # Merge target locations to assignments table
    result_with_locs = assignments.merge(
        target_locations.get_df(),
        left_on="target_id",
        right_on=target_locations.get_id_column(),
    )

    # Plot enumerators
    for enum_id, enum_coords in zip(
        enum_locations.get_ids(),
        enum_locations.get_gps_coords(),
    ):
        color = color_map[enum_id]
        folium.Marker(
            enum_coords,
            icon=folium.Icon(color=color, icon="user"),
            popup=f"Enumerator ID: {enum_id}",
        ).add_to(groups[enum_id])

    # Plot assigned targets for each enumerator
    for enum_id, df in result_with_locs.groupby("enum_id"):
        color = color_map[enum_id]

        for _, row in df.iterrows():
            target_coords = row[list(target_locations.get_gps_columns())].values
            target_id = row[target_locations.get_id_column()]
            folium.CircleMarker(
                target_coords,
                radius=3,
                color="#4B4B4B",
                stroke=True,
                weight=1.0,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"Target ID: {target_id}\nEnumerator ID: {enum_id}",
            ).add_to(groups[enum_id])

    # Format map
    for group in groups.values():
        map.add_child(group)
    map.add_child(folium.LayerControl())
    map.fit_bounds(map.get_bounds())
    return map
