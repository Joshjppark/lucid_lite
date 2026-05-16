"""Parse calibration.toml / calibration.json into Camera objects.

Mirrors file-io.js:125–250 (parseCalibrationTOML / parseCalibrationJSON).
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pose_data import Camera


def parse_calibration_toml(path: Path) -> list[Camera]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    cams: list[Camera] = []
    for key, sect in data.items():
        if not isinstance(sect, dict):
            continue
        if "matrix" not in sect:
            continue
        name = sect.get("name") or key.replace("cam_", "").replace("camera_", "")
        cams.append(_build_camera(name, sect))
    return cams


def parse_calibration_json(path: Path) -> list[Camera]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "cameras" in data:
        entries = data["cameras"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [v for v in data.values() if isinstance(v, dict) and "matrix" in v]
    cams: list[Camera] = []
    for i, sect in enumerate(entries):
        name = sect.get("name", f"cam_{i}")
        cams.append(_build_camera(name, sect))
    return cams


def _build_camera(name: str, sect: dict) -> Camera:
    matrix = sect.get("matrix") or sect.get("K") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dist = sect.get("distortions") or sect.get("dist") or [0.0, 0.0, 0.0, 0.0, 0.0]
    rvec = sect.get("rotation") or sect.get("rvec") or [0.0, 0.0, 0.0]
    tvec = sect.get("translation") or sect.get("tvec") or [0.0, 0.0, 0.0]
    size = sect.get("size") or [0, 0]
    return Camera(
        name=str(name),
        matrix=[list(r) for r in matrix],
        dist=list(dist),
        rvec=list(rvec) if hasattr(rvec, "__iter__") else [0.0, 0.0, 0.0],
        tvec=list(tvec) if hasattr(tvec, "__iter__") else [0.0, 0.0, 0.0],
        size=(int(size[0]), int(size[1])) if len(size) >= 2 else (0, 0),
    )


def find_calibration(folder: Path) -> Path | None:
    """Find calib*.toml / calib*.json in folder (case-insensitive)."""
    for p in folder.iterdir():
        low = p.name.lower()
        if "calib" in low and p.suffix.lower() in (".toml", ".json"):
            return p
    return None
