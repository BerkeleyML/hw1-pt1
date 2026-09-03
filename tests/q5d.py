from otter.test_files import test_case

OK_FORMAT = False

name = "q5d"
points = 2.0

@test_case(points=None, hidden=False)
def test_q5d_matrix_shape(create_rotation_matrix):
    assert create_rotation_matrix(15).shape == (784, 784), 'Rotation matrix should be 784x784'

@test_case(points=None, hidden=False)
def test_q5d_rotate_45(rotate_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_45_transform.npy')
    gt_updated = np.load('test_data/public/rotate_45_transform_updated.npy')
    got = rotate_image(test_image, 45)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Rotate 45 image does not match solution'

@test_case(points=None, hidden=False)
def test_q5d_rotate_90(rotate_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_90_transform.npy')
    gt_updated = np.load('test_data/public/rotate_90_transform_updated.npy')
    got = rotate_image(test_image, 90)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Rotate 90 image does not match solution'

@test_case(points=None, hidden=False)
def test_q5d_rotate_200(rotate_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_200_transform.npy')
    gt_updated = np.load('test_data/public/rotate_200_transform_updated.npy')
    got = rotate_image(test_image, 200)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Rotate 200 image does not match solution'

@test_case(points=None, hidden=False)
def test_q5d_rotate_270(rotate_image, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_270_transform.npy')
    gt_updated = np.load('test_data/public/rotate_270_transform_updated.npy')
    got = rotate_image(test_image, 270)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Rotate 270 image does not match solution'

