from otter.test_files import test_case

OK_FORMAT = False

name = "q4fi"
points = 0.5

@test_case(points=None, hidden=False)
def test_q4fi_type(least_confident, test_df):
    import pandas as pd
    assert isinstance(least_confident, pd.DataFrame), 'least_confident should be a pandas DataFrame'
    assert len(least_confident) == len(test_df), 'least_confident should contain all rows from test_df'
    assert set(least_confident.columns) == set(test_df.columns), 'least_confident should have the same columns as test_df'

@test_case(points=None, hidden=False)
def test_q4fi_sorted(least_confident, test_df):
    import numpy as np
    assert (least_confident['confidence'].diff()[1:] >= -1e-09).all(), 'least_confident should be sorted by confidence in ascending order'
    assert np.isclose(least_confident.iloc[0]['confidence'], test_df['confidence'].min(), atol=1e-06), 'First row should have lowest confidence'

