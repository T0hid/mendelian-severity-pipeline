"""Locate bundled data locally and in a standard Code Ocean capsule."""
import os
from pathlib import Path


def reference_directory(source_directory=None):
    source=Path(source_directory or Path(__file__).resolve().parent)
    # Explicit temporary/legacy package roots remain usable by preservation tests.
    legacy=source/'reference'
    if legacy.is_dir():
        return legacy
    configured=os.environ.get('SEVERITY_DATA_DIR')
    candidates=([Path(configured)/'reference'] if configured else [])+[
        source.parent/'data/reference',source.parent.parent/'data/reference']
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError('Reference data not found. Keep the supplied data folder with the code.')
