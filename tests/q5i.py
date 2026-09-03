from otter.test_files import test_case

OK_FORMAT = False

name = "q5i"
points = 2.0

@test_case(points=None, hidden=False)
def test_q5i_defined(env):
    assert 'aug_performance' in env, 'aug_performance DataFrame not found. Did you create it using groupby and agg operations?'

