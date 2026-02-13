import re
import yaml
from pathlib import Path


def _find_config():
    p = Path(__file__).resolve().parent.parent / "config" / "robot_config.yaml"
    if p.exists():
        return p
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src" / "aruco_camera_localizer" / "config" / "robot_config.yaml"
        if candidate.exists():
            return candidate
    return p

_DEFAULT_CONFIG_PATH = _find_config()


class RobotConfig:
    """Robot & camera configuration. Loads from YAML with defaults fallback."""

    _DEFAULTS = {
        'cam_offset_position': [0.0, -0.07, 0.0825],
        'cam_offset_quat': [0.0, 0.0, 0.0, 1.0],
        'camera_width': 1280,
        'camera_height': 720,
        'camera_hfov': 69.4,
        'camera_vfov': 42.5,
        'camera_matrix': None,
        'distortion_coeffs': [0, 0, 0, 0, 0],
        'exposure_mode': 'auto',        # 'auto' = lock on startup, 'manual' = use values below
        'white_balance_mode': 'auto',   # 'auto' = lock on startup, 'manual' = use values below
        'white_balance_temp': 4500,
        'exposure': -7.0,
        'ee_pose_topic': '/tcp_pose_broadcaster/pose',
        'table_z': -0.0825,
    }

    def __init__(self, config_path=None):
        for key, value in self._DEFAULTS.items():
            setattr(self, key, value)

        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
            for key, value in yaml_data.items():
                if key in self._DEFAULTS:
                    setattr(self, key, value)
                else:
                    print(f"[RobotConfig] Warning: unknown key '{key}' in {path}")
            print(f"[RobotConfig] Loaded from {path}")
        else:
            print(f"[RobotConfig] No config file at {path}, using defaults")

    def to_dict(self):
        return {key: getattr(self, key) for key in self._DEFAULTS}

    @staticmethod
    def _format_value(value):
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, float):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            items = ', '.join(str(v) for v in value)
            return f"[{items}]"
        if value is None:
            return 'null'
        return str(value)

    def save(self, path=None):
        p = Path(path) if path else _DEFAULT_CONFIG_PATH
        data = self.to_dict()
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
        missing = [k for k in data if k not in updated_keys]
        if missing:
            new_lines.append('\n')
            for key in missing:
                new_lines.append(f"{key}: {self._format_value(data[key])}\n")
        with open(p, 'w') as f:
            f.writelines(new_lines)
        print(f"[RobotConfig] Saved to {p}")
