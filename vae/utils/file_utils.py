import os
from pathlib import Path


def get_files_in_path(path, matching_pattern="*.csv"):
    files = list(Path(path).rglob(matching_pattern))
    return files
