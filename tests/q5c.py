from otter.test_files import test_case

OK_FORMAT = False

name = "q5c"
points = 2.0

@test_case(points=None, hidden=False)
def test_q5c_matrix_shape(create_blur_matrix):
    assert create_blur_matrix(2).shape == (784, 784), 'Blur matrix should be 784x784'

@test_case(points=None, hidden=False)
def test_q5c_blur_3x3(blur_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/blur_3x3_transform.npy')
    gt_updated = np.load('test_data/public/blur_3x3_transform_updated.npy')
    got = blur_image(test_image, 3)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Blur 3x3 image does not match solution'

@test_case(points=None, hidden=False)
def test_q5c_blur_5x5(blur_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/blur_5x5_transform.npy')
    gt_updated = np.load('test_data/public/blur_5x5_transform_updated.npy')
    got = blur_image(test_image, 5)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Blur 5x5 image does not match solution'

