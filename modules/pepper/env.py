from typing import Optional

import os
import re


def show_env(pattern: Optional[str] = None):
    """Displays the values of environment variables, optionally filtered by a
    regex pattern.

    Parameters
    ----------
    pattern : str, optional
        Regex pattern to filter the environment variables (default is None).

    Examples
    --------
    >>> # Display all environment variables
    >>> show_env()
    >>> # Display environment variables filtered by regex pattern
    >>> show_env("^PYSPARK_.*")
    """
    # Retrieve all environment variables
    env_vars = os.environ

    # Filter the environment variables based on the regex pattern
    filtered_vars = {}
    if pattern:
        regex = re.compile(pattern)
        filtered_vars = {
            var: value
            for var, value in env_vars.items()
            if regex.search(var)
        }

    # Display the values of the filtered or non-filtered environment variables
    for var, value in filtered_vars.items() if filtered_vars else env_vars.items():
        print(f'{var}: {value}')


def _create_if_not_exist(dir: str) -> None:
    """Creates a directory if it doesn't already exist.

    Parameters
    ----------
    dir : str
        The directory path to create.

    Returns
    -------
    None
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_project_dir() -> str:
    """Returns the project's base directory path.

    Raises
    ------
    RuntimeError
        If the `PROJECT_DIR` environment variable is not set.

    Returns
    -------
    str
        The project's base directory path.
    """
    if project_path := os.getenv("PROJECT_DIR"):
        return project_path
    else:
        raise RuntimeError("The `PROJECT_DIR` environment variable is not set.")


def _get_created_subdir(parent_fullpath, subdir_name: str) -> str:
    dir = os.path.join(parent_fullpath, subdir_name)
    _create_if_not_exist(dir)
    return dir

def get_modules_dir() -> str:
    """Returns the project's modules directory path."""
    return _get_created_subdir(get_project_dir(), "modules")


def get_data_dir() -> str:
    """Returns the project's dataset directory path."""
    return _get_created_subdir(get_project_dir(), "data")


def get_data_csv_dir() -> str:
    """Returns the project's dataset CSV directory path."""
    return _get_created_subdir(get_data_dir(), "csv")


def get_data_pqt_dir() -> str:
    """Returns the project's dataset Parquet directory path."""
    return _get_created_subdir(get_data_dir(), "pqt")


def get_data_im_dir() -> str:
    """Returns the project's dataset images directory path."""
    return _get_created_subdir(get_data_dir(), "im")


def get_tmp_dir() -> str:
    """Returns the project's tmp directory path."""
    return _get_created_subdir(get_project_dir(), "tmp")


def get_tmp_im_dir() -> str:
    """Returns the project's output images directory path."""
    return _get_created_subdir(get_tmp_dir(), "im")


def get_prep_dir() -> str:
    """Returns the project's output preprocessed data directory path."""
    return _get_created_subdir(get_tmp_dir(), "prep")


def get_prep_im_dir() -> str:
    """Returns the project's output preprocessed images data directory path."""
    return _get_created_subdir(get_prep_dir(), "im")