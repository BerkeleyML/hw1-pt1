from otter.test_files import test_case

OK_FORMAT = False

name = "q4ci"
points = 0.5

@test_case(points=None, hidden=False)
def test_q4ci_columns_and_splits(class_accuracy):
    assert set(class_accuracy.columns) == {'split', 'label', 'correct'}, f'class_accuracy has incorrect columns: {set(class_accuracy.columns)}'
    assert set(class_accuracy['split']) == {'train', 'test'}, f"class_accuracy has incorrect splits: {set(class_accuracy['split'])}"

@test_case(points=None, hidden=False)
def test_q4ci_labels_and_values(class_accuracy, train_df, test_df):
    all_labels = set(train_df['label']).union(set(test_df['label']))
    assert set(class_accuracy['label']) == all_labels, f"class_accuracy has incorrect labels: {set(class_accuracy['label'])}"
    assert class_accuracy['correct'].between(0, 1).all(), "class_accuracy has incorrect 'correct' values (should be between 0 and 1)"

