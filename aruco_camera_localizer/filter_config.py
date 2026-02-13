import re
import yaml
from pathlib import Path


# Default config path: try source tree first, then relative to __file__
def _find_config():
    # Try relative to __file__ (works in source tree)
    p = Path(__file__).resolve().parent.parent / "config" / "filter_config.yaml"
    if p.exists():
        return p
    # Walk up from __file__ to find the workspace src directory
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "aruco_camera_localizer" / "config" / "filter_config.yaml"
        if candidate.exists():
            return candidate
    return p  # return default even if missing (will use defaults)

_DEFAULT_CONFIG_PATH = _find_config()


class FilterConfig:
    """Configuration for the ArUco localizer pipeline. Loads from YAML if available."""

    # Defaults (used if YAML is missing or a key is absent)
    _DEFAULTS = {
        'z_range_min': 0.05,
        'z_range_max': 2.0,
        'enable_ema_smoothing': True,
        'ema_alpha': 0.5,
        'board_pose_mode': 'combined',
        'object_pose_mode': 'combined',
        'board_yaw_only': True,
        'board_snap_z': False,
        'object_snap_z': False,
        'object_snap_z_tolerance': 0.03,
        'enable_active_marker_tracking': True,
        'active_marker_timeout': 60,
        'enable_ghost_tracking': True,
        'ghost_timeout': 5,
        'enable_fold_snap': True,
        'fold_snap_subtypes': ['block', 'peg'],
        'snap_tilt_threshold': 30.0,
        'enable_motion_pause': True,
        'motion_speed_threshold': 50.0,
        'motion_pause_holdover': 30,
        'max_reproj_error': 1.0,
        'enable_ippe_disambiguation': True,
        'ippe_reproj_ratio': 2.0,  # min worst/best reproj ratio to trust disambiguation
        'wireframe_raw_pose_boards': False,
        'wireframe_raw_pose_objects': False,
        'wireframe_prefer_side_markers': True,
        'debug_show_rejected': False,
        'euler_convention': 'extrinsic',
        # Detector parameters (OpenCV defaults)
        'detector_adaptive_thresh_win_max': 23,
        'detector_min_perim_rate': 0.03,
        'detector_max_perim_rate': 4.0,
        'detector_poly_approx_rate': 0.03,
        'detector_corner_refine_method': 0,  # 0=None, 1=Subpix, 2=Contour, 3=AprilTag
        'enable_clahe': True,
    }

    def __init__(self, config_path=None):
        # Start with defaults
        for key, value in self._DEFAULTS.items():
            setattr(self, key, value)

        # Load YAML overrides
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
            for key, value in yaml_data.items():
                if key in self._DEFAULTS:
                    setattr(self, key, value)
                else:
                    print(f"[FilterConfig] Warning: unknown key '{key}' in {path}")
            print(f"[FilterConfig] Loaded from {path}")
        else:
            print(f"[FilterConfig] No config file at {path}, using defaults")

    def to_dict(self):
        """Return current config values as a dict (only keys in _DEFAULTS)."""
        return {key: getattr(self, key) for key in self._DEFAULTS}

    @staticmethod
    def _format_value(value):
        """Format a Python value as a YAML scalar."""
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, float):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Short lists inline: ['block', 'peg']
            items = ', '.join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
            return f"[{items}]"
        return str(value)

    def save(self, path=None):
        """Save current config, preserving comments and structure of the original file."""
        p = Path(path) if path else _DEFAULT_CONFIG_PATH
        data = self.to_dict()

        # Pattern matches "key: value" lines (with optional trailing comment)
        key_re = re.compile(r'^(\s*)([\w]+)\s*:\s*(.*?)(\s*#.*)?$')

        lines = []
        updated_keys = set()

        if p.exists():
            with open(p) as f:
                lines = f.readlines()

        new_lines = []
        for line in lines:
            m = key_re.match(line)
            if m:
                indent, key, old_val, comment = m.groups()
                if key in data:
                    new_val = self._format_value(data[key])
                    comment = comment or ''
                    new_lines.append(f"{indent}{key}: {new_val}{comment}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Append any new keys not in the original file
        missing = [k for k in data if k not in updated_keys]
        if missing:
            new_lines.append('\n')
            for key in missing:
                new_lines.append(f"{key}: {self._format_value(data[key])}\n")

        with open(p, 'w') as f:
            f.writelines(new_lines)
        print(f"[FilterConfig] Saved to {p}")
