import pandas as pd
import folium

from branca.element import Template, MacroElement


class Legend(MacroElement):
    def __init__(self, title, colors, labels):
        super(Legend, self).__init__()
        self._name = "Legend"
        self._title = title
        self._colors = colors
        self._labels = labels

        self._template = Template("""
{% macro header(this, kwargs) %}
<div style='position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; 
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding:5px; border-radius: 10px;'>
<h4 style='text-align: center;'>{{this.title}}</h4>
{% for color, label in zip(this.colors, this.labels) %}
<div style='text-align: left; margin-left: 20px;'><i class='fa fa-circle fa-1x' style='color:{{color}};'></i> {{label}}</div>
{% endfor %}
</div>
{% endmacro %}
""")


def load_data():
    enums = pd.read_csv("./test_enum_data.csv")
    targets = pd.read_csv("./test_target_data.csv")
    assignment = pd.read_csv("./test_assignment_data.csv")
    return enums, targets, assignment


def merge_data(enums, targets, assignment):
    # Merge to link enums and targets based on assignment
    assigned_targets = assignment.merge(
        targets, left_on="target_id", right_on="id", suffixes=("", "_target")
    )
    assigned_enums = assigned_targets.merge(
        enums, left_on="enum_id", right_on="id", suffixes=("_target", "_enum")
    )
    return assigned_enums


def assign_colors(data):
    # Supported colors by folium for icons
    supported_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "white",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]
    unique_groups = data["enum_id"].unique()
    if len(unique_groups) > len(supported_colors):
        raise ValueError(
            "There are more groups than supported colors. Some groups will have the same color."
        )
    group_colors = {
        enum_id: supported_colors[i % len(supported_colors)]
        for i, enum_id in enumerate(unique_groups)
    }
    return group_colors


def create_map(assigned_enums, group_colors):
    center = (15.0794, 120.6200)
    map = folium.Map(location=center, zoom_start=11)

    # Marker for each target and enumerator
    for i, row in assigned_enums.iterrows():
        # Targets
        folium.Marker(
            [row["gps_lat_target"], row["gps_lon_target"]],
            icon=folium.Icon(
                color=group_colors[row["enum_id"]],
                icon_color="white",
                icon="circle",
                prefix="fa",
            ),
            popup=f"Target ID: {row['target_id']}",
        ).add_to(map)

        # Enumerators
        folium.Marker(
            [row["gps_lat_enum"], row["gps_lon_enum"]],
            icon=folium.Icon(color=group_colors[row["enum_id"]], icon="user"),
            popup=f"Enumerator ID: {row['enum_id']}",
        ).add_to(map)

    map.add_child(folium.LayerControl())

    return map


if __name__ == "__main__":
    enums, targets, assignment = load_data()
    assigned_enums = merge_data(enums, targets, assignment)
    group_colors = assign_colors(assigned_enums)
    map = create_map(assigned_enums, group_colors)
    map.save("./map_group_colors.html")
