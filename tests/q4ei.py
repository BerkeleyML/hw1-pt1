from otter.test_files import test_case

OK_FORMAT = False

name = "q4ei"
points = 1.5

@test_case(points=None, hidden=False)
def test_q4ei_accuracy(accuracy_from_matrix, test_df):
    import numpy as np
    assert isinstance(accuracy_from_matrix, float), 'accuracy_from_matrix should be a float'
    assert np.isclose(accuracy_from_matrix, test_df['correct'].mean(), atol=1e-06), f"Accuracy from confusion matrix {accuracy_from_matrix} does not match accuracy from DataFrame {test_df['correct'].mean()}"

@test_case(points=None, hidden=False)
def test_q4ei_metrics_list(per_class_metrics, class_names):
    assert per_class_metrics is not None, 'per_class_metrics variable should exist'
    assert isinstance(per_class_metrics, list), 'per_class_metrics should be a list'
    assert len(per_class_metrics) == len(class_names), 'per_class_metrics should have the same length as class_names'
    assert all((isinstance(metric, dict) for metric in per_class_metrics)), 'all elements of per_class_metrics should be dictionaries'
    dict_keys = ['class', 'precision', 'recall']
    assert all((set(metric.keys()) == set(dict_keys) for metric in per_class_metrics)), f'all dictionaries in per_class_metrics should have the keys {dict_keys}'

