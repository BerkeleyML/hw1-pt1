from otter.test_files import test_case

OK_FORMAT = False

name = "q4b"
points = 2.0

@test_case(points=None, hidden=False)
def test_q4b_columns(train_df, test_df):
    columns = ['fmnist_idx', 'label', 'predicted_label', 'correct', 'probs', 'confidence']
    assert all((col in train_df.columns for col in columns)), f'train_df missing columns: {set(columns) - set(train_df.columns)}'
    assert all((col in test_df.columns for col in columns)), f'test_df missing columns: {set(columns) - set(test_df.columns)}'

@test_case(points=None, hidden=False)
def test_q4b_types(train_df):
    import numpy as np
    first = train_df.iloc[0]
    assert isinstance(first['label'], str), 'label should be a string'
    assert isinstance(first['predicted_label'], str), 'predicted_label should be a string'
    assert isinstance(first['correct'], (bool, np.bool_)), 'correct should be a boolean'
    assert isinstance(first['probs'], list), 'probs should be a list'
    assert isinstance(first['confidence'], (float, np.floating)), 'confidence should be a float'

@test_case(points=None, hidden=False)
def test_q4b_probs_length(train_df, test_df):
    assert all((isinstance(p, list) and len(p) == 10 for p in train_df['probs'])), "'probs' in train_df not correct shape (should be length 10)"
    assert all((isinstance(p, list) and len(p) == 10 for p in test_df['probs'])), "'probs' in test_df not correct shape (should be length 10)"

@test_case(points=None, hidden=False)
def test_q4b_confidence(train_df, test_df):
    assert all((abs(conf - max(probs)) < 1e-06 for conf, probs in zip(train_df['confidence'], train_df['probs']))), "'confidence' in train_df not max of 'probs'"
    assert all((abs(conf - max(probs)) < 1e-06 for conf, probs in zip(test_df['confidence'], test_df['probs']))), "'confidence' in test_df not max of 'probs'"

