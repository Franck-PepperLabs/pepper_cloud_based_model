from pepper.env import _get_created_subdir, get_data_im_dir, get_prep_im_dir


def get_data_im_sample_300_dir() -> str:
    """Returns the project's dataset images sample (300) directory path."""
    return _get_created_subdir(get_data_im_dir(), "sample_300")


def get_prep_im_sample_300_dir() -> str:
    """Returns the project's preprocessed images sample (300) directory path."""
    return _get_created_subdir(get_prep_im_dir(), "sample_300")
