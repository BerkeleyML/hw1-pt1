from otter.test_files import test_case

OK_FORMAT = False

name = "q2b"
points = 2.0

@test_case(points=None, hidden=False)
def test_q2b_type(mean_brightness):
    import numpy as np
    assert isinstance(mean_brightness, dict), 'mean_brightness should be a dict'
    assert all((isinstance(v, (int, float, np.floating)) for v in mean_brightness.values())), 'mean_brightness values should be numbers'

@test_case(points=None, hidden=False)
def test_q2b_keys(mean_brightness, class_names):
    assert set(mean_brightness.keys()) == set(class_names), 'mean_brightness keys should be the class names'

