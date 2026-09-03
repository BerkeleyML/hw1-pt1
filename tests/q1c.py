from otter.test_files import test_case

OK_FORMAT = False

name = "q1c"
points = 2.0

@test_case(points=None, hidden=False)
def test_q1c_point(autograd_f_grads):
    gx2, gy2 = autograd_f_grads(1.0, 0.0)
    assert abs(gx2 - 0.0) < 1e-05 and abs(gy2 - 2.0) < 1e-05, 'Check grads at (1, 0): expected (0, 2)'

