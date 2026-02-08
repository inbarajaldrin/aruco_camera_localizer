"""
Path Finder Utility
Recursively searches for aruco-grasp-annotator data directory in Documents folder.
"""
import json
from pathlib import Path
from typing import Optional, Set


def find_aruco_data_dir() -> Optional[Path]:
    """
    Recursively search Documents folder for aruco-grasp-annotator/data directory.

    Returns:
        Path to data directory if found, None otherwise
    """
    documents_dir = Path.home() / "Documents"
    if not documents_dir.exists():
        return None

    target_name = "aruco-grasp-annotator"
    for path in documents_dir.rglob(target_name):
        if path.is_dir():
            data_dir = path / "data"
            if data_dir.exists() and (data_dir / "aruco").exists():
                return data_dir

    return None


def get_models_by_type(data_dir: Optional[Path] = None) -> dict:
    """
    Read assembly JSON files and return models grouped by type.

    Returns:
        {'board': {'base1', ...}, 'object': {'u_orange', ...}}
    """
    if data_dir is None:
        data_dir = find_aruco_data_dir()
    if data_dir is None:
        return {'board': set(), 'object': set()}

    result = {'board': set(), 'object': set()}
    for f in data_dir.glob("*assembly*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for component in data.get("components", []):
                model_type = component.get("type", "object")
                result.setdefault(model_type, set()).add(component["name"])
        except (json.JSONDecodeError, KeyError):
            continue
    return result
