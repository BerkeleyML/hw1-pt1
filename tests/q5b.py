from otter.test_files import test_case

OK_FORMAT = False

name = "q5b"
points = 2.0

@test_case(points=None, hidden=False)
def test_q5b_matrix_shape(create_shift_matrix):
    assert create_shift_matrix(5, 0).shape == (784, 784), 'Shift matrix should be 784x784'

@test_case(points=None, hidden=False)
def test_q5b_shift_right(shift_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/shift_right_transform.npy')
    assert np.array_equal(shift_image(test_image, 5, 0), gt), 'Shift right image does not match solution'

@test_case(points=None, hidden=False)
def test_q5b_shift_left(shift_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/shift_left_transform.npy')
    assert np.array_equal(shift_image(test_image, -5, 0), gt), 'Shift left image does not match solution'

@test_case(points=None, hidden=False)
def test_q5b_shift_up(shift_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/shift_up_transform.npy')
    assert np.array_equal(shift_image(test_image, 0, 5), gt), 'Shift up image does not match solution'

@test_case(points=None, hidden=False)
def test_q5b_shift_down(shift_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/shift_down_transform.npy')
    assert np.array_equal(shift_image(test_image, 0, -5), gt), 'Shift down image does not match solution'

