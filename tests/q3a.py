from otter.test_files import test_case

OK_FORMAT = False

name = "q3a"
points = 2.0

@test_case(points=None, hidden=False)
def test_q3a_length(kmeans_df, meta_sample):
    assert len(kmeans_df) == len(meta_sample), f'kmeans_df and meta_sample have different lengths {len(kmeans_df)} != {len(meta_sample)}'

@test_case(points=None, hidden=False)
def test_q3a_columns(kmeans_df):
    assert set(kmeans_df.columns) == set(['cluster', 'label', 'fmnist_idx']), f'kmeans_df has unexpected columns {set(kmeans_df.columns)}'

