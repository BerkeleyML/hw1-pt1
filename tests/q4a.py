from otter.test_files import test_case

OK_FORMAT = False

name = "q4a"
points = 1.0

@test_case(points=None, hidden=False)
def test_q4a_test_accuracy(model, X_test_sc, y_test):
    acc = model.score(X_test_sc, y_test)
    assert acc < 0.91 and acc > 0.86, f'Your accuracy ({int(acc * 100)}%) does not match expected'

