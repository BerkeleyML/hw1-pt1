from otter.test_files import test_case

OK_FORMAT = False

name = "q4gi"
points = 0.5

@test_case(points=None, hidden=False)
def test_q4gi_trouser_filter(test_df_trouser):
    import pandas as pd
    assert isinstance(test_df_trouser, pd.DataFrame), 'test_df_trouser is not a pandas DataFrame'
    assert all(test_df_trouser['label'] == 'Trouser'), "test_df_trouser should only contain rows where label is 'Trouser'"

@test_case(points=None, hidden=False)
def test_q4gi_high_conf_incorrect(high_conf_incorrect):
    import pandas as pd
    assert isinstance(high_conf_incorrect, pd.DataFrame), 'high_conf_incorrect is not a pandas DataFrame'
    assert all(high_conf_incorrect['correct'] == False), 'high_conf_incorrect should only contain incorrect predictions'

@test_case(points=None, hidden=False)
def test_q4gi_high_conf_sorted(high_conf_incorrect):
    assert all(high_conf_incorrect['confidence'].diff().fillna(0) <= 0), 'high_conf_incorrect should be sorted by confidence in descending order'

