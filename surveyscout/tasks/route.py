from typing import Tuple, List
import pandas as pd
import aiohttp
import asyncio

from surveyscout.config import OSRM_URL
from surveyscout.utils import LocationDataset


def convert_coords_to_osrm_query(locations: LocationDataset) -> str:
    """Turn cooordinates in LocationDataset into OSRM query string"""
    lng_col, lat_col = locations.gps_lng_column, locations.gps_lat_column

    return ";".join(
        locations.df.apply(lambda row: f"{row[lng_col]},{row[lat_col]}", axis=1)
    )


async def suggest_route(
    assignment_df: pd.DataFrame,
    target_locations: LocationDataset,
    enum_target_cost_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Suggests the route for the enumerators to visit the targets using OSRM's greedy
    travelling salesman algorithm.

    Parameters
    ----------
    assignment_matrix : numpy.ndarray or list of lists
        The raw results matrix from the optimization algorithm, with `1` indicating
        an assigned target and `0` otherwise.

    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets, with a
        similar structure to `enum_locations`.

    enum_target_cost_matrix : pd.DataFrame, optional
        distance matrix between enumerators and targets.
        Columns are enumerator IDs, rows are target IDs.

    **kwargs : dict, optional
        Additional keyword arguments that can be used in a custom postprocessing step.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame where each row represents an assignment with columns corresponding
        to target IDs, enumerator IDs, and assigned values (all 1's).
    """
    # df.columns = [
    #     "target_id",
    #     "enum_id",
    #     "cost",
    # ]
    visit_ranks = []

    async with aiohttp.ClientSession() as session:
        tasks = []

        for enum_id, subdf in assignment_df.groupby("enum_id"):
            # Get cyclic order of targets
            # TODO: look at the heuristic to see if it always chooses the closest one first?
            # TODO: is it beneficial to order targets by cost?
            assigned_target_locations = target_locations.create_subset(
                subdf["target_id"]
            )

            coord_query_string = convert_coords_to_osrm_query(assigned_target_locations)
            data = subdf.copy()

            task = get_visit_order_for_enum(data, coord_query_string, session)
            tasks.append(task)

        visit_ranks = await asyncio.gather(*tasks)

    return pd.concat(visit_ranks, axis=0)


async def get_visit_order_for_enum(
    df: pd.DataFrame, coordinates_string: str, session: aiohttp.ClientSession
) -> Tuple[List[int], List[float]]:
    """Get visit order for a given enumerator based on OSRM trip API"""
    trip_endpoint = "/trip/v1/car/{coordinates_string}"

    params = dict(
        steps="false", geometries="polyline", overview="simplified", annotations="false"
    )

    async with session.get(
        OSRM_URL + trip_endpoint.format(coordinates_string=coordinates_string),
        params=params,
    ) as response:
        response.raise_for_status()
        response_json = await response.json()

        waypoints = response_json["waypoints"]
        trips = response_json["trips"]

        # TODO: cycle so that the first target is the closest to the enumerator
        target_visit_rank = [waypoint["waypoint_index"] for waypoint in waypoints]
        distance_km_to_next = [leg["distance"] / 1000 for leg in trips[0]["legs"]]

        df["target_visit_rank"] = target_visit_rank
        df["distance_to_next_in_km"] = distance_km_to_next

        return df
