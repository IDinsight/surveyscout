import os
import pandas as pd
import aiohttp
from typing import Optional
import asyncio

import surveyscout
from surveyscout.utils import LocationDataset


def _convert_coords_to_osrm_query(locations: LocationDataset) -> str:
    """Turn cooordinates in LocationDataset into OSRM query string"""
    lng_col, lat_col = locations.gps_lng_column, locations.gps_lat_column

    return ";".join(
        locations.df.apply(lambda row: f"{row[lng_col]},{row[lat_col]}", axis=1)
    )


def add_visit_order(
    assignment_df: pd.DataFrame,
    target_locations: LocationDataset,
    osrm_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Task to add suggested visit order to the assignment dataframe.

    Suggests the route for the enumerators to visit the targets using OSRM's greedy
    travelling salesman algorithm.

    Parameters
    ----------
    assignment_matrix : numpy.ndarray or list of lists
        The raw results matrix from the optimization algorithm, with `1` indicating
        an assigned target and `0` otherwise.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets, with a
        similar structure to `enum_locations`.

    osrm_url: Optional[str]
        URL for OSRM API. If None, looks for a non-null value in this order:
        1. library config surveyscout.OSRM_URL, 2. environment variable OSRM_URL, 3.
        default value "http://localhost:5001".

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with suggest order of visiting targets.
        Has columns
            - "target_id": inherited from assignment_df
            - "cost": inherited from assignment_df
            - "target_visit_order": suggested order of visiting
            - "distance_to_next_in_km": distance to the next target in kilometers
    """
    osrm_url = (
        osrm_url
        or surveyscout.OSRM_URL
        or os.getenv("OSRM_URL")
        or "http://localhost:5001"
    )
    url = osrm_url + "/trip/v1/car/"

    visit_orders = asyncio.run(_get_visit_order(assignment_df, target_locations, url))

    assignment_df = assignment_df.merge(
        visit_orders, on=["target_id", "enum_id"], how="left"
    ).sort_values(by=["enum_id", "target_visit_order"])

    return assignment_df


async def _get_visit_order(
    assignment_df: pd.DataFrame,
    target_locations: LocationDataset,
    url: str,
) -> pd.DataFrame:
    """
    Suggests the route for the enumerators to visit the targets using OSRM's greedy
    travelling salesman algorithm.
    """
    visit_ranks = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        for enum_id, subdf in assignment_df.groupby("enum_id"):
            assigned_target_locations = target_locations.create_subset(
                subdf["target_id"]
            )

            coord_query_string = _convert_coords_to_osrm_query(
                assigned_target_locations
            )
            data = subdf.copy()

            task = _get_visit_order_for_enum(data, url, coord_query_string, session)
            tasks.append(task)

        visit_ranks = await asyncio.gather(*tasks)

    return pd.concat(visit_ranks, axis=0)


async def _fetch_osrm_trip_data(
    url: str, coordinates_string: str, session: aiohttp.ClientSession
) -> dict:
    """Makes OSRM trip request"""
    params = dict(
        steps="false", geometries="polyline", overview="simplified", annotations="false"
    )
    async with session.get(
        url + coordinates_string,
        params=params,
    ) as response:
        response.raise_for_status()
        return await response.json()


async def _get_visit_order_for_enum(
    df: pd.DataFrame, url: str, coordinates_string: str, session: aiohttp.ClientSession
) -> pd.DataFrame:
    """Get visit order for a given enumerator based on OSRM trip API"""
    response_json = await _fetch_osrm_trip_data(
        url=url, coordinates_string=coordinates_string, session=session
    )

    waypoints = response_json["waypoints"]
    trips = response_json["trips"]

    target_visit_order = [waypoint["waypoint_index"] for waypoint in waypoints]
    distance_km_to_next = [leg["distance"] / 1000 for leg in trips[0]["legs"]]

    df["target_visit_order"] = target_visit_order
    df["distance_to_next_in_km"] = distance_km_to_next

    df = _rerank(df)

    return df


def _rerank(df: pd.DataFrame) -> pd.DataFrame:
    """Rerank the visit order of targets for each enumerator so that the first target is
    the closest target."""
    closest_target_id = df.target_id[df.cost.idxmin()]

    start_rank = df.loc[df.target_id == closest_target_id, "target_visit_order"].iloc[0]

    cycled_ranks = (df.target_visit_order - start_rank) % len(df)
    df["target_visit_order"] = cycled_ranks

    df = df.sort_values(by="target_visit_order")

    return df
