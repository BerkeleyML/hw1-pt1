from otter.test_files import test_case

OK_FORMAT = False

name = "q5f"
points = 1.0

@test_case(points=None, hidden=False)
def test_q5f_compose_shape(compose_transforms, create_rotation_matrix, create_blur_matrix):
    assert compose_transforms(create_rotation_matrix(45), create_blur_matrix(2)).shape == (784, 784), 'Compose transforms should return a 784x784 matrix'

@test_case(points=None, hidden=False)
def test_q5f_rotate_then_blur(rotate_then_blur, test_image):
    import numpy as np
    gt = np.load('test_data/public/rotate_then_blur_transform.npy')
    gt_updated = np.load('test_data/public/rotate_then_blur_transform_updated.npy')
    assert np.allclose(rotate_then_blur(test_image, 45, 2), gt, rtol=0.0001, atol=0.0001) or np.allclose(rotate_then_blur(test_image, 45, 3), gt_updated, rtol=0.0001, atol=0.0001), 'Rotate then blur image does not match solution'

@test_case(points=None, hidden=False)
def test_q5f_shift_then_rotate_then_blur(shift_then_rotate_then_blur, test_image):
    import numpy as np
    gt = np.load('test_data/public/shift_then_rotate_then_blur_transform.npy')
    gt_updated = np.load('test_data/public/shift_then_rotate_then_blur_transform_updated.npy')
    assert np.allclose(shift_then_rotate_then_blur(test_image, 1, -4, 200, 3), gt, rtol=0.0001, atol=0.0001) or np.allclose(shift_then_rotate_then_blur(test_image, 1, -4, 200, 5), gt_updated, rtol=0.0001, atol=0.0001), 'Shift then rotate then blur image does not match solution'

