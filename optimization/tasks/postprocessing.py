import pandas as pd


def postprocess_results(results, enum_locations, target_locations, **kwargs):
    """Postprocess optimization results."""
    # Postprocess results
    df_results = pd.DataFrame(
        results,
        index=target_locations.get_ids,
        columns=enum_locations.get_ids(),
    )
    stacked = df_results.stack()
    filtered = stacked[stacked == 1]
    df = filtered.reset_index(level=[0, 1])
    df.columns = [
        target_locations.get_id_column(),
        enum_locations.get_id_column(),
        "value",
    ]

    return results
