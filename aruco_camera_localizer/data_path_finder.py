"""
Path Finder Utility
Recursively searches for aruco-grasp-annotator data directory in Documents folder.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Set


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


def get_symmetry_dir(data_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the symmetry subdirectory inside the data directory."""
    if data_dir is None:
        data_dir = find_aruco_data_dir()
    if data_dir is None:
        return None
    sym_dir = data_dir / "symmetry"
    return sym_dir if sym_dir.exists() else None


def get_model_subtypes(data_dir: Optional[Path] = None) -> Dict[str, str]:
    """Read assembly JSONs and return {model_name: subtype}.

    Subtypes come from the 'subtype' field in assembly components
    (e.g. 'block', 'peg', 'socket'). Board components are skipped.
    """
    if data_dir is None:
        data_dir = find_aruco_data_dir()
    if data_dir is None:
        return {}

    subtypes = {}
    for f in data_dir.glob("*assembly*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for component in data.get("components", []):
                if component.get("type") == "board":
                    continue
                name = component.get("name")
                subtype = component.get("subtype")
                if name and subtype:
                    subtypes[name] = subtype
        except (json.JSONDecodeError, KeyError):
            continue
    return subtypes


def load_symmetry_data(data_dir: Optional[Path] = None) -> Dict[str, Dict[str, int]]:
    """Load fold symmetry data for all objects.

    Returns:
        {model_name: {'x': fold_int, 'y': fold_int, 'z': fold_int}}
    """
    sym_dir = get_symmetry_dir(data_dir)
    if sym_dir is None:
        return {}

    result = {}
    for f in sym_dir.glob("*_symmetry.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            obj_name = data.get("object_name", f.stem.replace("_symmetry", ""))
            fold_axes = data.get("fold_axes", {})
            result[obj_name] = {
                axis: fold_axes[axis].get("fold", 1)
                for axis in ("x", "y", "z")
                if axis in fold_axes
            }
        except (json.JSONDecodeError, KeyError):
            continue
    return result
