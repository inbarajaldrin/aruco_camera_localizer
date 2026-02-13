"""Interactive tkinter tuning panels for real-time config adjustment.

All panels share a single tkinter thread to avoid segfaults.
"""

import tkinter as tk
from tkinter import ttk
import threading
import cv2
import numpy as np


# ── Filter config groups ─────────────────────────────────────────────────────
PANEL_GROUPS = [
    ("Detection Thresholds", [
        ("Max Reproj Error",        "max_reproj_error",        "scale",  {"from_": 0, "to": 10.0, "resolution": 0.1}),
        ("Z Range Min (m)",         "z_range_min",             "scale",  {"from_": 0, "to": 1.0,  "resolution": 0.01}),
        ("Z Range Max (m)",         "z_range_max",             "scale",  {"from_": 0.5, "to": 5.0, "resolution": 0.1}),
    ]),
    ("Smoothing", [
        ("Enable EMA Smoothing",    "enable_ema_smoothing",    "check",  {}),
        ("EMA Alpha",               "ema_alpha",               "scale",  {"from_": 0, "to": 1.0, "resolution": 0.05}),
    ]),
    ("Orientation Snapping", [
        ("Board Yaw Only",          "board_yaw_only",          "check",  {}),
        ("Board Snap Z",            "board_snap_z",            "check",  {}),
        ("Object Snap Z",           "object_snap_z",           "check",  {}),
        ("Fold Snap",               "enable_fold_snap",        "check",  {}),
        ("Snap Tilt Threshold (deg)", "snap_tilt_threshold",   "scale",  {"from_": 0, "to": 90, "resolution": 1}),
        ("Z Snap Tolerance (m)",    "object_snap_z_tolerance", "scale",  {"from_": 0, "to": 0.1, "resolution": 0.005}),
    ]),
    ("Tracking", [
        ("Ghost Timeout (frames)",  "ghost_timeout",           "int_scale", {"from_": 0, "to": 60}),
        ("Active Timeout (frames)", "active_marker_timeout",   "int_scale", {"from_": 0, "to": 120}),
    ]),
    ("Visualization", [
        ("IPPE Disambiguation",     "enable_ippe_disambiguation", "check", {}),
        ("IPPE Reproj Ratio",       "ippe_reproj_ratio",         "scale", {"from_": 1.0, "to": 10.0, "resolution": 0.1}),
        ("Show Rejected",           "debug_show_rejected",     "check",  {}),
        ("Raw Wireframe (Boards)",  "wireframe_raw_pose_boards",  "check",  {}),
        ("Raw Wireframe (Objects)", "wireframe_raw_pose_objects", "check",  {}),
        ("Side Markers",            "wireframe_prefer_side_markers", "check", {}),
    ]),
    ("Detector Parameters", [
        ("Enable CLAHE",            "enable_clahe",                     "check",     {}),
        ("Adaptive Thresh Win Max", "detector_adaptive_thresh_win_max", "int_scale", {"from_": 3, "to": 100}),
        ("Min Perimeter Rate",      "detector_min_perim_rate",          "scale", {"from_": 0.001, "to": 0.1, "resolution": 0.001}),
        ("Max Perimeter Rate",      "detector_max_perim_rate",          "scale", {"from_": 0.1, "to": 5.0, "resolution": 0.1}),
        ("Poly Approx Rate",        "detector_poly_approx_rate",        "scale", {"from_": 0.01, "to": 0.1, "resolution": 0.005}),
        ("Corner Refine (0=None 1=Subpix)", "detector_corner_refine_method", "int_scale", {"from_": 0, "to": 3}),
    ]),
]

_DETECTOR_ATTR_MAP = {
    'detector_adaptive_thresh_win_max': 'adaptiveThreshWinSizeMax',
    'detector_min_perim_rate': 'minMarkerPerimeterRate',
    'detector_max_perim_rate': 'maxMarkerPerimeterRate',
    'detector_poly_approx_rate': 'polygonalApproxAccuracyRate',
    'detector_corner_refine_method': 'cornerRefinementMethod',
}

# ── Robot config groups ───────────────────────────────────────────────────────
ROBOT_PANEL_GROUPS = [
    ("Camera Offset", [
        ("Cam Offset X (m)", "cam_offset_position", "list_entry", {"index": 0}),
        ("Cam Offset Y (m)", "cam_offset_position", "list_entry", {"index": 1}),
        ("Cam Offset Z (m)", "cam_offset_position", "list_entry", {"index": 2}),
    ]),
    ("Camera Intrinsics", [
        ("H-FOV (deg)",      "camera_hfov",         "entry",  {}),
        ("V-FOV (deg)",      "camera_vfov",         "entry",  {}),
    ]),
    ("Camera Hardware", [
        ("Exposure Mode",    "exposure_mode",       "combo",  {"values": ["auto", "manual"]}),
        ("Exposure",         "exposure",            "entry",  {}),
        ("WB Mode",          "white_balance_mode",  "combo",  {"values": ["auto", "manual"]}),
        ("WB Temp",          "white_balance_temp",  "entry",  {}),
        ("Re-Auto Exposure & WB", "_auto_settle",   "button", {}),
    ]),
    ("Environment", [
        ("Table Z (m)",      "table_z",             "entry",  {}),
    ]),
]


# ── Shared tkinter thread ────────────────────────────────────────────────────

_tk_root = None
_tk_thread = None
_tk_ready = threading.Event()
_tk_lock = threading.Lock()


def _start_tk_thread():
    """Start the shared tkinter thread if not already running."""
    global _tk_thread
    with _tk_lock:
        if _tk_thread is not None and _tk_thread.is_alive():
            return
        _tk_ready.clear()
        _tk_thread = threading.Thread(target=_tk_mainloop, daemon=True)
        _tk_thread.start()
        _tk_ready.wait()


def _tk_mainloop():
    """Tkinter main loop running in a dedicated thread."""
    global _tk_root
    _tk_root = tk.Tk()
    _tk_root.withdraw()  # hidden root — panels are Toplevel windows
    _tk_ready.set()
    _tk_root.mainloop()


def _run_in_tk(func):
    """Schedule func on the tkinter thread and wait for result."""
    result = [None]
    done = threading.Event()

    def _wrapper():
        result[0] = func()
        done.set()

    _tk_root.after(0, _wrapper)
    done.wait()
    return result[0]


# ── Helper: build widgets ────────────────────────────────────────────────────

def _build_filter_group(parent, group_label, items, config_obj, vars_dict):
    """Build a LabelFrame with sliders/checkboxes for filter config."""
    frame = ttk.LabelFrame(parent, text=group_label, padding=5)
    for display_label, attr, wtype, kwargs in items:
        current_val = getattr(config_obj, attr, 0)
        if wtype == "check":
            var = tk.BooleanVar(value=bool(current_val))
            ttk.Checkbutton(frame, text=display_label, variable=var).pack(anchor="w", pady=1)
        elif wtype == "scale":
            var = tk.DoubleVar(value=float(current_val))
            ttk.Label(frame, text=display_label).pack(anchor="w")
            tk.Scale(frame, variable=var, orient="horizontal",
                     from_=kwargs["from_"], to=kwargs["to"],
                     resolution=kwargs.get("resolution", 0.01),
                     length=260).pack(fill="x", padx=10)
        elif wtype == "int_scale":
            var = tk.IntVar(value=int(current_val))
            ttk.Label(frame, text=display_label).pack(anchor="w")
            tk.Scale(frame, variable=var, orient="horizontal",
                     from_=kwargs["from_"], to=kwargs["to"],
                     resolution=1, length=260).pack(fill="x", padx=10)
        else:
            continue
        vars_dict[attr] = var
    return frame


def _build_robot_group(parent, group_label, items, config_obj, vars_dict, list_vars_dict,
                       button_callbacks=None):
    """Build a LabelFrame with entries/checkboxes for robot config."""
    frame = ttk.LabelFrame(parent, text=group_label, padding=5)
    for display_label, attr, wtype, kwargs in items:
        if wtype == "button":
            cb = button_callbacks.get(attr) if button_callbacks else None
            if cb:
                ttk.Button(frame, text=display_label, command=cb).pack(fill="x", padx=10, pady=4)
            continue
        elif wtype == "list_entry":
            idx = kwargs["index"]
            current_list = getattr(config_obj, attr, [0, 0, 0])
            current_val = current_list[idx] if idx < len(current_list) else 0
            var = tk.StringVar(value=str(current_val))
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=display_label, width=20).pack(side="left")
            ttk.Entry(row, textvariable=var, width=14).pack(side="left", padx=5)
            list_vars_dict[(attr, idx)] = var
        elif wtype == "check":
            current_val = getattr(config_obj, attr, False)
            var = tk.BooleanVar(value=bool(current_val))
            ttk.Checkbutton(frame, text=display_label, variable=var).pack(anchor="w", pady=1)
            vars_dict[attr] = var
        elif wtype == "combo":
            current_val = getattr(config_obj, attr, kwargs["values"][0])
            var = tk.StringVar(value=str(current_val))
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=display_label, width=20).pack(side="left")
            ttk.Combobox(row, textvariable=var, values=kwargs["values"],
                         state="readonly", width=12).pack(side="left", padx=5)
            vars_dict[attr] = var
        elif wtype == "entry":
            current_val = getattr(config_obj, attr, 0)
            var = tk.StringVar(value=str(current_val))
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=display_label, width=20).pack(side="left")
            ttk.Entry(row, textvariable=var, width=14).pack(side="left", padx=5)
            vars_dict[attr] = var
        else:
            continue
    return frame


# ── TuningPanel (filter config) ──────────────────────────────────────────────

class TuningPanel:
    def __init__(self, filter_config, detector_params=None):
        self.filter_config = filter_config
        self.detector_params = detector_params
        self._vars = {}
        self._window = None
        self._alive = True
        _start_tk_thread()
        _run_in_tk(self._create_window)

    def _create_window(self):
        self._window = tk.Toplevel(_tk_root)
        self._window.title("Filter Tuning")
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        left_col = ttk.Frame(self._window)
        right_col = ttk.Frame(self._window)
        left_col.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        right_col.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)
        self._window.rowconfigure(0, weight=1)

        mid = (len(PANEL_GROUPS) + 1) // 2
        for group_label, items in PANEL_GROUPS[:mid]:
            f = _build_filter_group(left_col, group_label, items, self.filter_config, self._vars)
            f.pack(fill="x", padx=3, pady=3)
        for group_label, items in PANEL_GROUPS[mid:]:
            f = _build_filter_group(right_col, group_label, items, self.filter_config, self._vars)
            f.pack(fill="x", padx=3, pady=3)

        # Rejection log
        log_frame = ttk.LabelFrame(self._window, text="Rejection Log", padding=5)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=3, pady=3)
        self._window.rowconfigure(1, weight=1)
        self._log_text = tk.Text(log_frame, height=8, width=60, font=("monospace", 9),
                                 bg="#1e1e1e", fg="#cccccc", wrap="word", state="disabled")
        self._log_text.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(self._window, padding=5)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Button(btn_frame, text="Save as Default", command=self._save_as_default).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self._reset_defaults).pack(side="left", padx=5)
        self._status_label = ttk.Label(btn_frame, text="", foreground="green")
        self._status_label.pack(side="left", padx=10)

        self._window.update_idletasks()

    def _flash_status(self, msg):
        self._status_label.config(text=msg)
        self._window.after(2000, lambda: self._status_label.config(text=""))

    def _on_close(self):
        self._alive = False
        self._window.destroy()

    def _save_as_default(self):
        self.sync_to_config()
        self.filter_config.save()
        self._flash_status("Saved as default!")

    def _reset_defaults(self):
        from aruco_camera_localizer.filter_config import FilterConfig
        saved = FilterConfig()  # reload from YAML
        for attr, var in self._vars.items():
            val = getattr(saved, attr, None)
            if val is not None:
                var.set(val)
        self.sync_to_config()
        self._flash_status("Reset to saved defaults")

    @property
    def alive(self):
        return self._alive

    def update_rejection_log(self, marker_poses, rejected_markers, all_raw_markers,
                             detected_objects=None):
        """Update the rejection log with current frame info."""
        if not self._alive or self._window is None:
            return
        lines = []
        accepted_keys = set(marker_poses.keys())
        rejected_keys = set(rejected_markers.keys())
        all_keys = set(all_raw_markers.keys())

        # Which objects actually got a final pose?
        solved_objects = set()
        if detected_objects:
            for obj in detected_objects:
                if not obj.get('ghost_tracked', False):
                    solved_objects.add(obj['name'])

        lines.append(f"Markers — Detected: {len(all_keys)}  Accepted: {len(accepted_keys)}  Rejected: {len(rejected_keys)}")
        lines.append(f"Objects — Solved: {len(solved_objects)}")
        lines.append("")
        for key in sorted(all_keys, key=lambda k: (k[1], k[0])):
            mid, dname = key
            raw = all_raw_markers[key]
            rp = raw.get('reproj_error', 0)
            z = raw.get('tvec', [0, 0, 0])[2]
            if key in accepted_keys:
                lines.append(f"  OK       id={mid} ({dname})  z={z:.3f}m  reproj={rp:.2f}px")
            elif key in rejected_keys:
                reason = rejected_markers[key].get('reject_reason', 'unknown')
                lines.append(f"  REJECTED id={mid} ({dname})  z={z:.3f}m  reproj={rp:.2f}px")
                lines.append(f"    reason: {reason}")
            else:
                lines.append(f"  RAW      id={mid} ({dname})  z={z:.3f}m  reproj={rp:.2f}px")
        text = "\n".join(lines)
        def _update():
            self._log_text.config(state="normal")
            self._log_text.delete("1.0", "end")
            self._log_text.insert("1.0", text)
            self._log_text.config(state="disabled")
        try:
            _run_in_tk(_update)
        except Exception:
            pass

    def sync_to_config(self):
        if not self._alive or self._window is None:
            return
        for attr, var in self._vars.items():
            try:
                value = var.get()
            except tk.TclError:
                continue
            setattr(self.filter_config, attr, value)
            if self.detector_params is not None and attr in _DETECTOR_ATTR_MAP:
                det_attr = _DETECTOR_ATTR_MAP[attr]
                if det_attr in ('adaptiveThreshWinSizeMax', 'cornerRefinementMethod'):
                    setattr(self.detector_params, det_attr, int(value))
                else:
                    setattr(self.detector_params, det_attr, float(value))


# ── RobotTuningPanel (robot config) ──────────────────────────────────────────

class RobotTuningPanel:
    def __init__(self, robot_config):
        self.robot_config = robot_config
        self._vars = {}
        self._list_vars = {}
        self._window = None
        self._alive = True
        _start_tk_thread()
        _run_in_tk(self._create_window)

    def _create_window(self):
        self._window = tk.Toplevel(_tk_root)
        self._window.title("Robot Config Tuning")
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        left_col = ttk.Frame(self._window)
        right_col = ttk.Frame(self._window)
        left_col.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        right_col.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)
        self._window.rowconfigure(0, weight=1)

        btn_cbs = {'_auto_settle': self._request_auto_settle}
        mid = (len(ROBOT_PANEL_GROUPS) + 1) // 2
        for group_label, items in ROBOT_PANEL_GROUPS[:mid]:
            f = _build_robot_group(left_col, group_label, items,
                                   self.robot_config, self._vars, self._list_vars,
                                   button_callbacks=btn_cbs)
            f.pack(fill="x", padx=3, pady=3)
        for group_label, items in ROBOT_PANEL_GROUPS[mid:]:
            f = _build_robot_group(right_col, group_label, items,
                                   self.robot_config, self._vars, self._list_vars,
                                   button_callbacks=btn_cbs)
            f.pack(fill="x", padx=3, pady=3)

        btn_frame = ttk.Frame(self._window, padding=5)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save as Default", command=self._save_as_default).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset to Default", command=self._reset_defaults).pack(side="left", padx=5)
        self._status_label = ttk.Label(btn_frame, text="", foreground="green")
        self._status_label.pack(side="left", padx=10)

        self._window.update_idletasks()

    def _flash_status(self, msg):
        self._status_label.config(text=msg)
        self._window.after(2000, lambda: self._status_label.config(text=""))

    def _on_close(self):
        self._alive = False
        self._window.destroy()

    def _apply(self):
        self.sync_to_config()
        self._flash_status("Applied!")

    def _request_auto_settle(self):
        """Set flag so the main loop re-triggers auto-exposure/WB settle."""
        self._auto_settle_requested = True
        self._flash_status("Auto-settle requested...")

    @property
    def auto_settle_requested(self):
        """Check and clear the auto-settle request flag."""
        if getattr(self, '_auto_settle_requested', False):
            self._auto_settle_requested = False
            return True
        return False

    def _save_as_default(self):
        self.sync_to_config()
        self.robot_config.save()
        self._flash_status("Saved as default!")

    def _reset_defaults(self):
        from aruco_camera_localizer.robot_config import RobotConfig
        saved = RobotConfig()  # reload from YAML
        for attr, var in self._vars.items():
            val = getattr(saved, attr, None)
            if val is not None:
                var.set(val)
        for (attr, idx), var in self._list_vars.items():
            val_list = getattr(saved, attr, [])
            if idx < len(val_list):
                var.set(str(val_list[idx]))
        self.sync_to_config()
        self._flash_status("Reset to saved defaults")

    @property
    def alive(self):
        return self._alive

    def refresh_from_config(self):
        """Update panel widgets to reflect current robot_config values (e.g. after auto-settle)."""
        if not self._alive or self._window is None:
            return
        def _refresh():
            for attr, var in self._vars.items():
                val = getattr(self.robot_config, attr, None)
                if val is not None:
                    var.set(val if isinstance(val, (bool, str)) else str(val))
            for (attr, idx), var in self._list_vars.items():
                val_list = getattr(self.robot_config, attr, [])
                if idx < len(val_list):
                    var.set(str(val_list[idx]))
        try:
            _run_in_tk(_refresh)
        except Exception:
            pass

    def sync_to_config(self):
        if not self._alive or self._window is None:
            return
        for attr, var in self._vars.items():
            try:
                raw = var.get()
            except tk.TclError:
                continue
            if isinstance(raw, bool):
                setattr(self.robot_config, attr, raw)
            elif isinstance(raw, str):
                # Try to parse as number
                try:
                    if '.' in raw:
                        setattr(self.robot_config, attr, float(raw))
                    else:
                        setattr(self.robot_config, attr, int(raw))
                except ValueError:
                    setattr(self.robot_config, attr, raw)
            else:
                setattr(self.robot_config, attr, raw)
        # Rebuild list attrs
        list_attrs = {}
        for (attr, idx), var in self._list_vars.items():
            try:
                list_attrs.setdefault(attr, {})[idx] = float(var.get())
            except (tk.TclError, ValueError):
                continue
        for attr, idx_map in list_attrs.items():
            current = list(getattr(self.robot_config, attr, []))
            for idx, val in idx_map.items():
                if idx < len(current):
                    current[idx] = val
            setattr(self.robot_config, attr, current)


# ── Shared helpers ────────────────────────────────────────────────────────────

def draw_help_overlay(frame, paused=False):
    h, w = frame.shape[:2]
    text = "[P] Pause/Resume  [S] Save config  [D] Toggle rejected  [Q] Quit"
    if paused:
        text = "[PAUSED] " + text
    cv2.putText(frame, text, (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def handle_key(key, filter_config, paused, panel=None, robot_panel=None):
    if key == ord('q'):
        return True, paused
    if key == ord('s'):
        if panel and panel.alive:
            panel.sync_to_config()
        filter_config.save()
        if robot_panel and robot_panel.alive:
            robot_panel.sync_to_config()
            robot_panel.robot_config.save()
    if key == ord('d'):
        filter_config.debug_show_rejected = not filter_config.debug_show_rejected
        if panel and panel.alive and 'debug_show_rejected' in panel._vars:
            panel._vars['debug_show_rejected'].set(filter_config.debug_show_rejected)
    if key == ord('p'):
        paused = not paused
    return False, paused
