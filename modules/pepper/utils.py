from typing import Optional, List, Any

import locale
import calendar

import os
import glob

from IPython.display import clear_output

import matplotlib.pyplot as plt

# Set the style of plots
plt.style.use('dark_background')


""" Locale
"""

# Ex. f"{123456789:n}" : 123 456 789
def get_default_locale():
    return "fr_FR.UTF-8"


locale.setlocale(locale.LC_ALL, get_default_locale())


def get_weekdays(target_locale=None):
    if target_locale is None:
        target_locale = "en_US.UTF-8"
    locale.setlocale(locale.LC_ALL, target_locale)
    weekdays = [day.upper() for day in calendar.day_name]
    locale.setlocale(locale.LC_ALL, get_default_locale())
    return weekdays



def cls():
    """Clears the output of the current cell receiving output."""
    clear_output(wait=True)


"""Simple functions for pretty printing text.

Functions
---------
bold(s):
    Returns a string `s` wrapped in ANSI escape codes for bold text.
italic(s):
    Returns a string `s` wrapped in ANSI escape codes for italic text.
cyan(s):
    Returns a string `s` wrapped in ANSI escape codes for cyan text.
magenta(s):
    Returns a string `s` wrapped in ANSI escape codes for magenta text.
red(s):
    Returns a string `s` wrapped in ANSI escape codes for red text.
green(s):
    Returns a string `s` wrapped in ANSI escape codes for green text.
print_title(txt):
    Prints a magenta title with the text `txt` in bold font.
print_subtitle(txt):
    Prints a cyan subtitle with the text `txt` in bold font.
"""


def bold(s: Any) -> str:
    return "\033[1m" + str(s) + "\033[0m"


def italic(s: Any) -> str:
    return "\033[3m" + str(s) + "\033[0m"


def cyan(s: Any) -> str:
    return "\033[36m" + str(s) + "\033[0m"


def magenta(s: Any) -> str:
    return "\033[35m" + str(s) + "\033[0m"


def red(s: Any) -> str:
    return "\033[31m" + str(s) + "\033[0m"


def green(s: Any) -> str:
    return "\033[32m" + str(s) + "\033[0m"


def print_title(txt: str) -> None:
    print(bold(magenta('\n' + txt.upper())))


def print_subtitle(txt: str) -> None:
    print(bold(cyan('\n' + txt)))


def print_subsubtitle(txt: str) -> None:
    print(italic(green('\n' + txt)))

""" Generics
"""


def create_if_not_exist(dir: str) -> None:
    r"""Create a directory if it doesn't already exist.

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


def _get_filenames_glob(
    root_dir: str,
    ext: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """Returns a list of filenames in the specified directory
    using glob pattern matching.

    Parameters
    ----------
        root_dir : str
            The root directory to search for filenames in.
        ext : str, optional
            The extension to filter the filenames by.
            Defaults to None, which returns all files.
        recursive : bool, optional
            Whether or not to search for filenames recursively.
            Defaults to False.

    Returns
    -------
        List[str]
            A list of filenames found in the directory.
    """
    ext = ext if ext else "*"
    if recursive:
        filenames = glob.glob(f"**/*.{ext}", root_dir=root_dir, recursive=True)
    else:
        filenames = glob.glob(f"*.{ext}", root_dir=root_dir)
    filenames = [filename.replace("\\", "/") for filename in filenames]
    return filenames
