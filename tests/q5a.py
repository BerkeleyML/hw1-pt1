from otter.test_files import test_case

OK_FORMAT = False

name = "q5a"
points = 1.0

@test_case(points=None, hidden=False)
def test_q5a_matrix_shape(create_horizontal_flip_matrix):
    assert create_horizontal_flip_matrix().shape == (784, 784), 'Horizontal flip matrix should be 784x784'

@test_case(points=None, hidden=False)
def test_q5a_flipped_image(horizontal_flip, test_image):
    import numpy as np
    gt_horizontal_flip_image = np.load('test_data/public/horizontal_flip_image.npy')
    assert np.array_equal(horizontal_flip(test_image), gt_horizontal_flip_image), 'Horizontal flip image does not match solution'

