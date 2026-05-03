"""Camera discovery: probe-by-identity (preferred) + legacy interactive scan (fallback).

Probe-by-identity walks /sys/class/video4linux/ to find the camera matching a
USB identity from config (name / vendor id / product id / serial), then picks
the V4L2 sub-device offering the preferred pixel format (e.g. YUYV for RGB,
not Z16 depth or GREY IR). This survives reboots and replug — /dev/videoN
numbering does not.
"""

import os
import re
import subprocess
from pathlib import Path

import cv2


_V4L_SYSFS = Path("/sys/class/video4linux")


def _read(p):
    """Read a sysfs file, stripped. Return None if missing/unreadable."""
    try:
        return Path(p).read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _find_usb_attr(video_dir, attr):
    """Walk up from /sys/class/video4linux/videoN/device/ to find a USB attr file.

    The 'device' symlink points at the V4L2 interface; the USB device (with
    idVendor / idProduct / serial) is typically one or two levels above. We
    walk up until we find the attribute or hit /sys.
    """
    try:
        cur = (Path(video_dir) / "device").resolve()
    except OSError:
        return None
    sys_root = Path("/sys")
    while cur != sys_root and cur.exists():
        candidate = cur / attr
        if candidate.exists():
            val = _read(candidate)
            if val:
                return val
        cur = cur.parent
    return None


def _list_formats(dev_path):
    """Return set of fourcc strings supported by /dev/videoN, e.g. {'YUYV', 'MJPG'}."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", str(dev_path), "--list-formats"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    # Lines look like: "        [0]: 'YUYV' (YUYV 4:2:2)"
    return set(re.findall(r"'([A-Z0-9 ]{4})'", out))


def enumerate_v4l2_devices():
    """Return list of dicts describing every /dev/videoN, sorted by index."""
    devices = []
    if not _V4L_SYSFS.exists():
        return devices
    for entry in sorted(_V4L_SYSFS.iterdir(), key=lambda p: int(re.sub(r"\D", "", p.name) or -1)):
        m = re.match(r"video(\d+)$", entry.name)
        if not m:
            continue
        idx = int(m.group(1))
        dev_path = f"/dev/video{idx}"
        if not os.path.exists(dev_path):
            continue
        devices.append({
            "index": idx,
            "path": dev_path,
            "name": _read(entry / "name") or "",
            "vendor_id": (_find_usb_attr(entry, "idVendor") or "").lower(),
            "product_id": (_find_usb_attr(entry, "idProduct") or "").lower(),
            "serial": _find_usb_attr(entry, "serial") or "",
            "formats": _list_formats(dev_path),
        })
    return devices


def _matches_identity(dev, *, name, vendor_id, product_id, serial):
    """Apply identity filters. None/empty filter = wildcard."""
    if name and name.lower() not in dev["name"].lower():
        return False
    if vendor_id and dev["vendor_id"] != vendor_id.lower():
        return False
    if product_id and dev["product_id"] != product_id.lower():
        return False
    if serial and dev["serial"] != serial:
        return False
    return True


def probe_camera_by_identity(config, *, verbose=True):
    """Find the /dev/videoN matching the camera identity in robot_config.

    Returns the integer video index on success, or None if no match.

    Strategy:
      1. Enumerate all /dev/videoN with their USB identity + supported formats.
      2. Filter by camera_match_name / vendor_id / product_id / serial.
      3. Among matches, pick the one whose formats include camera_prefer_format
         (e.g. YUYV for RGB color, vs Z16 depth or GREY IR sub-devices).
      4. If no format match but identity matches, return the lowest-indexed match.
    """
    name = getattr(config, "camera_match_name", None)
    vendor_id = getattr(config, "camera_match_vendor_id", None)
    product_id = getattr(config, "camera_match_product_id", None)
    serial = getattr(config, "camera_serial", None)
    prefer_format = getattr(config, "camera_prefer_format", None)

    devices = enumerate_v4l2_devices()
    if not devices:
        if verbose:
            print("[camera_probe] No V4L2 devices found under /sys/class/video4linux/")
        return None

    matches = [d for d in devices
               if _matches_identity(d, name=name, vendor_id=vendor_id,
                                    product_id=product_id, serial=serial)]
    if not matches:
        if verbose:
            print(f"[camera_probe] No device matched identity "
                  f"(name~='{name}', vid={vendor_id}, pid={product_id}, serial={serial}).")
            print("[camera_probe] Available devices:")
            for d in devices:
                print(f"  /dev/video{d['index']}: '{d['name']}' "
                      f"vid={d['vendor_id']} pid={d['product_id']} formats={sorted(d['formats'])}")
        return None

    if prefer_format:
        fmt = prefer_format.upper().ljust(4)[:4]
        with_fmt = [d for d in matches if fmt in d["formats"]]
        if with_fmt:
            chosen = with_fmt[0]
            if verbose:
                print(f"[camera_probe] Matched {len(matches)} device(s); "
                      f"picked /dev/video{chosen['index']} "
                      f"('{chosen['name']}', formats={sorted(chosen['formats'])}) "
                      f"for prefer_format='{prefer_format}'")
            return chosen["index"]
        if verbose:
            print(f"[camera_probe] {len(matches)} identity match(es) but none expose "
                  f"format '{prefer_format}'. Falling through to first match.")

    chosen = matches[0]
    if verbose:
        print(f"[camera_probe] Picked /dev/video{chosen['index']} "
              f"('{chosen['name']}', formats={sorted(chosen['formats'])})")
    return chosen["index"]


# ---------------------------------------------------------------------------
# Legacy interactive fallback (used when probe-by-identity returns None)
# ---------------------------------------------------------------------------

def detect_available_cameras(max_cams=15):
    """Try to open camera IDs and return a list of working ones."""
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def select_camera(available_ids):
    """Let user preview and select from available cameras."""
    print("Available camera IDs:", available_ids)
    for cam_id in available_ids:
        cap = cv2.VideoCapture(cam_id)
        print(f"Showing preview for camera ID {cam_id} (press any key to continue, or ESC to select this one)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"PREVIEW OF CAMERA {cam_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press ESC to SELECT this camera", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press any key to SKIP this camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Camera ID {cam_id}", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return cam_id
            elif key != -1:
                break
        cap.release()
        cv2.destroyAllWindows()
    return available_ids[0] if available_ids else None
