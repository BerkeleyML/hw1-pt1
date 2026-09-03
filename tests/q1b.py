from otter.test_files import test_case

OK_FORMAT = False

name = "q1b"
points = 2.0

@test_case(points=None, hidden=False)
def test_q1b_shape(maximize_Ax):
    import numpy as np
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    x_star = maximize_Ax(A)
    assert x_star.shape == (2,), 'for [[3, 0], [0, 1]], x_star should have shape (2,)'

@test_case(points=None, hidden=False)
def test_q1b_unit(maximize_Ax):
    import numpy as np
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    x_star = maximize_Ax(A)
    assert np.isclose(np.linalg.norm(x_star), 1.0), 'for [[3, 0], [0, 1]], x_star should be a unit vector'

@test_case(points=None, hidden=False)
def test_q1b_direction(maximize_Ax):
    import numpy as np
    A = np.array([[3.0, 0.0], [0.0, 1.0]])
    x_star = maximize_Ax(A)
    assert np.isclose(abs(x_star[0]), 1.0) and np.isclose(x_star[1], 0.0), 'for [[3, 0], [0, 1]], x_star should be [±1, 0]'

