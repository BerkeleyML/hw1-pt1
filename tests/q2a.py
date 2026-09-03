from otter.test_files import test_case

OK_FORMAT = False

name = "q2a"
points = 1.0

@test_case(points=None, hidden=False)
def test_q2a_type(label_distribution):
    import pandas as pd
    assert isinstance(label_distribution, pd.Series), 'label_distribution should be a pandas Series'

@test_case(points=None, hidden=False)
def test_q2a_sum_and_classes(label_distribution, meta, class_names):
    assert label_distribution.sum() == len(meta), 'label_distribution should sum to the number of samples'
    assert set(label_distribution.index) == set(class_names), 'label_distribution should have the same classes as class_names'

@test_case(points=None, hidden=False)
def test_q2a_balanced_type(is_balanced):
    import numpy as np
    assert isinstance(is_balanced, (bool, np.bool_)), 'is_balanced should be a boolean'

@test_case(points=None, hidden=False)
def test_q2a_balanced_value(is_balanced):
    assert is_balanced, 'Dataset should be balanced'

