from otter.test_files import test_case

OK_FORMAT = False

name = "q2ci"
points = 0.5

@test_case(points=None, hidden=False)
def test_q2ci_label_count(sample_labels):
    assert len(sample_labels) == 20, f'Expected 20 labels, got {len(sample_labels)}'

@test_case(points=None, hidden=False)
def test_q2ci_two_per_class(sample_labels):
    import pandas as pd
    counts = pd.Series(sample_labels).value_counts()
    assert all(counts == 2), f'Expected 2 samples per class, got {counts.to_dict()}'

