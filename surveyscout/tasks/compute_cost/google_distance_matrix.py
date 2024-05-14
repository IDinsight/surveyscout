"""Computing distances using Google Distance Matrix API."""

from typing import List
import logging
import os
import urllib.parse
from collections import defaultdict
from itertools import product

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import requests

from surveyscout.utils import LocationDataset

logger = logging.getLogger(__name__)


# Request limits as per https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
MAX_DEST = 25
MAX_ELEMENTS_PER_REQUEST = 100
MAX_ORIG = MAX_ELEMENTS_PER_REQUEST // MAX_DEST
# Since there are more targets (destinations) than enumerators (origins), we use
# MAX_DEST = 25 and MAX_ORIG = 4 to arrive at MAX_ELEMENTS_PER_REQUEST = 100

MAX_ELEMENTS_PER_SECOND = 1000  # 60000 EPM / 60 seconds
WAIT = MAX_ELEMENTS_PER_REQUEST / MAX_ELEMENTS_PER_SECOND  # TODO: use in logic


def get_enum_target_google_distance_matrix(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    value: str = "duration",
) -> pd.DataFrame:
    """
    Create a enumerator-target distance matrix.

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets.

    value : str, optional
        "duration" or "distance". Value to use for the distance matrix, by default "duration"

    Returns
    -------
    pd.DataFrame
        distance matrix between enumerators and targets.
        Columns are enumerator IDs, rows are target IDs.
    """
    cost_table = get_enum_target_google_distance_table(enum_locations, target_locations)

    cost_matrix = cost_table.pivot_table(
        values=value, index="dest_id", columns="orig_id"
    )

    cost_matrix.columns = enum_locations.get_ids()
    cost_matrix.index = target_locations.get_ids()

    return cost_matrix


def get_enum_target_google_distance_table(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
) -> pd.DataFrame:
    """Creates a enumerator-target distance/duration table

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets.


    Returns
    -------
    pd.DataFrame
        Table of distances/durations between every enumerator and target
        with columns `orig_id` corresponding to enumerator IDs and `dest_id`
        corresponding to target IDs.
    """
    enum_ids = enum_locations.get_ids()
    enum_gps_coords = enum_locations.get_gps_coords()

    target_ids = target_locations.get_ids()
    target_gps_coords = target_locations.get_gps_coords()

    logger.debug(f"Num origins: {len(enum_ids)}")
    logger.debug(f"Num destinations: {len(target_ids)}")

    request_idx_pairs_generator = _generate_google_enum_target_request_pairs(
        len(enum_ids), len(target_ids)
    )

    parsed_dfs = []

    for orig_idx, dest_idx in request_idx_pairs_generator:
        response = _get_google_distance_matrix(
            enum_gps_coords[orig_idx], target_gps_coords[dest_idx]
        )

        parsed = _parse_google_distance_matrix_response(
            response, enum_ids[orig_idx].tolist(), target_ids[dest_idx].tolist()
        )
        parsed_dfs.append(parsed)

    distance_df = pd.concat(parsed_dfs)

    return distance_df


def _format_coords_into_string(coords: NDArray) -> str:
    """Formats GPS coordinates into string as required by Google Distance Matrix API

    Parameters
    ----------
    coords : np.array
        numpy array of shape (2, n) representing latitude longitude pairs

    Returns
    -------
    str
        formatted string of GPS coordinates
    """
    return "|".join(map(lambda x: str(x[0]) + "," + str(x[1]), coords))


def _get_google_distance_matrix(
    origin_coords: NDArray, destination_coords: NDArray
) -> dict:
    """Get Google Distance Matrix API response for a single request

    Parameters
    ----------
    origin_coords : np.array
        numpy array of shape (2, n) representing latitude longitude pairs of origins
    destination_coords : np.array
        numpy array of shape (2, n) representing latitude longitude pairs of destinations

    Returns
    -------
    dict
        Response JSON
    """
    assert len(origin_coords) <= MAX_ORIG
    assert len(destination_coords) <= MAX_DEST
    assert len(origin_coords) * len(destination_coords) <= MAX_ELEMENTS_PER_REQUEST

    formatted_orig_locations = _format_coords_into_string(origin_coords)
    formatted_dest_locations = _format_coords_into_string(destination_coords)

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    payload: dict = {}
    headers: dict = {}

    params = {
        "key": os.getenv("GOOGLE_MAPS_PLATFORM_API_KEY"),
        "origins": formatted_orig_locations,
        "destinations": formatted_dest_locations,
        "region": "in",
    }

    query_str = urllib.parse.urlencode(params)
    assert len(f"{url}?{query_str}") <= 8192

    response = requests.request(
        "GET", url, headers=headers, data=payload, params=params
    )

    return response.json()


def _parse_google_distance_matrix_response(
    response_json: dict, orig_ids: List, dest_ids: List
) -> pd.DataFrame:
    """Parse Google Distance Matrix API response into a dictionary of lists (columns)

    Parameters
    ----------
    response : Response
        A response object returned by Google Distance Matrix API
    orig_ids : list-like
        List of origin IDs
    dest_ids : list-like
        List of destination IDs

    Returns
    -------
    pandas.DataFrame
        A dataframe of origin and destination IDs and the travel distance/duration
        between them
    """
    columns = ["distance", "duration", "distance_text", "duration_text"]

    parsed = defaultdict(list)

    if response_json["status"] != "OK":
        logger.warning(
            f"Response status not OK for origins {orig_ids} and destinations {dest_ids}.\n Response status={response_json['status']}"
        )

        for col in columns:
            parsed[col].extend([np.nan] * len(orig_ids) * len(dest_ids))

        return pd.DataFrame(parsed)

    assert len(response_json["rows"]) == len(orig_ids)  # Each row is an origin

    for i, row in enumerate(response_json["rows"]):
        assert len(row["elements"]) == len(dest_ids)  # Each element is a destination
        logger.debug(f"{i}th row['elements']={len(row['elements'])}=len(dest_ids)")

        for j, el in enumerate(row["elements"]):
            if el["status"] != "OK":
                logger.warning(
                    f"{i}th row {j}th element status not ok: {row['status']}"
                )

                for col in columns:
                    parsed[col].append(np.nan)
            else:
                parsed["distance"].append(el["distance"]["value"])
                parsed["duration"].append(el["duration"]["value"])
                parsed["distance_text"].append(el["distance"]["text"])
                parsed["duration_text"].append(el["duration"]["text"])

        logger.debug(f"{i}th row: orig_id={[orig_ids[i]] * len(dest_ids)}")
        logger.debug(f"{i}th row: dest_id={dest_ids}")
        parsed["orig_id"].extend([orig_ids[i]] * len(dest_ids))
        parsed["dest_id"].extend(dest_ids)

    return pd.DataFrame(parsed)


def _generate_google_enum_target_request_pairs(n_orig: int, n_dest: int):
    """
    Generate enum-target pairs such that the request limit is not exceeded.

    Params
    ------
    n_orig: int
        Number of origins

    n_dest: int
        Number of destinations

    Returns
    -------
    Generator[orig_idx_slice, dest_idx_slice]
        where each (orig_idx_slice, dest_idx_slice) pair is formed such that the request limit is not exceeded (https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing)
    """
    n_orig_groups = int(np.ceil(n_orig / MAX_ORIG))
    n_dest_groups = int(np.ceil(n_dest / MAX_DEST))

    orig_indices = np.arange(n_orig)
    dest_indices = np.arange(n_dest)

    orig_generator = (
        orig_indices[i * MAX_ORIG : (i + 1) * MAX_ORIG] for i in range(n_orig_groups)
    )
    dest_generator = (
        dest_indices[j * MAX_DEST : (j + 1) * MAX_DEST] for j in range(n_dest_groups)
    )

    return product(orig_generator, dest_generator)
