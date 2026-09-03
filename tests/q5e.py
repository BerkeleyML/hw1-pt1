from otter.test_files import test_case

OK_FORMAT = False

name = "q5e"
points = 2.0

@test_case(points=None, hidden=False)
def test_q5e_matrix_shape(create_bilinear_rotation_matrix):
    assert create_bilinear_rotation_matrix(15).shape == (784, 784), 'Rotation matrix should be 784x784'

@test_case(points=None, hidden=False)
def test_q5e_rotate_45(rotate_image_bilinear, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_45_bilinear_transform.npy')
    gt_updated = np.load('test_data/public/rotate_45_bilinear_transform_updated.npy')
    got = rotate_image_bilinear(test_image, 45)
    assert np.allclose(got, gt, rtol=0.0001, atol=0.0001) or np.allclose(got, gt_updated, rtol=0.0001, atol=0.0001), 'Rotate bilinear 45 image does not match solution'

