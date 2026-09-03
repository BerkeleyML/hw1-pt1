from otter.test_files import test_case

OK_FORMAT = False

name = "q5hi"
points = 0.75

@test_case(points=None, hidden=False)
def test_q5hi_record_type(augmented_data):
    assert isinstance(augmented_data[0], dict), 'Each item in augmented_data should be a dictionary'
    assert all((key in augmented_data[0] for key in ['original_idx', 'fmnist_idx', 'augmentation', 'label'])), 'Augmented data dictionaries missing required keys'

@test_case(points=None, hidden=False)
def test_q5hi_df_vs_images(aug_df, augmented_images):
    assert 'image' not in aug_df.columns, 'aug_df should not contain an image column; store pixels in augmented_images'
    assert set(['original_idx', 'fmnist_idx', 'augmentation', 'label']).issubset(aug_df.columns), 'aug_df is missing required columns'
    assert augmented_images.shape[0] == len(aug_df), 'augmented_images should have one row per aug_df record'
    assert augmented_images.shape[1] == 784, 'augmented_images should have 784 features'

@test_case(points=None, hidden=False)
def test_q5hi_has_shift(aug_df):
    aug_types = aug_df['augmentation'].unique()
    assert any(('shift' in aug for aug in aug_types)), 'No shift augmentations found'

@test_case(points=None, hidden=False)
def test_q5hi_has_rotate(aug_df):
    aug_types = aug_df['augmentation'].unique()
    assert any(('rotate' in aug for aug in aug_types)), 'No rotation augmentations found'

@test_case(points=None, hidden=False)
def test_q5hi_has_blur(aug_df):
    aug_types = aug_df['augmentation'].unique()
    assert any(('blur' in aug for aug in aug_types)), 'No blur augmentations found'

