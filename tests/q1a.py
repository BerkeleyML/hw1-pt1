from otter.test_files import test_case

OK_FORMAT = False

name = "q1a"
points = 2.0

@test_case(points=None, hidden=False)
def test_q1a_scaled_triangular(matrix_power_to_zero):
    import numpy as np
    A = np.array([[0.3, 0.2], [0.0, 0.8]])
    got = matrix_power_to_zero(A)
    assert isinstance(got, (bool, np.bool_)), 'matrix_power_to_zero should return a bool'
    assert bool(got) is True, '[[0.3, 0.2], [0, 0.8]] should go to 0'

@test_case(points=None, hidden=False)
def test_q1a_scaled_rotation(matrix_power_to_zero):
    import numpy as np
    A = np.array([[0.0, -0.5], [0.5, 0.0]])
    got = matrix_power_to_zero(A)
    assert isinstance(got, (bool, np.bool_)), 'matrix_power_to_zero should return a bool'
    assert bool(got) is True, '[[0.0, -0.5], [0.5, 0.0]] should go to 0'

@test_case(points=None, hidden=False)
def test_q1a_jordan_block(matrix_power_to_zero):
    import numpy as np
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    got = matrix_power_to_zero(A)
    assert isinstance(got, (bool, np.bool_)), 'matrix_power_to_zero should return a bool'
    assert bool(got) is False, '[[1.0, 1.0], [0.0, 1.0]] should not go to 0'

@test_case(points=None, hidden=False)
def test_q1a_unit_rotation(matrix_power_to_zero):
    import numpy as np
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    got = matrix_power_to_zero(A)
    assert isinstance(got, (bool, np.bool_)), 'matrix_power_to_zero should return a bool'
    assert bool(got) is False, '[[0.0, -1.0], [1.0, 0.0]] should not go to 0'

