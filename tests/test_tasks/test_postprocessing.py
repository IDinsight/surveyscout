import numpy as np
import pandas as pd
import pytest

from surveyscout.tasks.postprocessing import postprocess_results


@pytest.fixture(scope="module")
def assignment_df(enum_df, target_df):
    a = np.zeros((len(target_df), len(enum_df)))
    a[[3, 4, 7], 0] = 1.0
    a[[1, 2, 8, 9], 1] = 1.0
    a[[0, 5, 6], 2] = 1.0
    return pd.DataFrame(a, columns=enum_df.get_ids(), index=target_df.get_ids())


def test_postprocess_results(assignment_df, enum_df, target_df):
    df = postprocess_results(assignment_df.values, enum_df, target_df)

    for idx, row in df.iterrows():
        assert assignment_df.loc[row.target_id, row.enum_id] == 1
