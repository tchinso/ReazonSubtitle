from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


APP_NAME = "ReazonSubtitle"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def application_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", application_dir())).resolve()
    return application_dir()


def assets_dir() -> Path:
    """Return the external, portable assets directory.

    The multi-gigabyte models deliberately live beside the executable. Keeping
    them out of PyInstaller's internal directory avoids unpacking or copying
    them whenever the program starts.
    """

    external = application_dir() / "assets"
    if external.is_dir():
        return external
    return resource_dir() / "assets"


def work_root() -> Path:
    local = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local) if local else Path(tempfile.gettempdir())
    path = base / APP_NAME / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chrome_profile_root() -> Path:
    local = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local) if local else Path(tempfile.gettempdir())
    path = base / APP_NAME / "ChromeProfile"
    path.mkdir(parents=True, exist_ok=True)
    return path
