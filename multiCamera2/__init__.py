# -*- coding: utf-8 -*-
"""
Package multiCamera2: Hệ thống đồng bộ ID & Bounding Box đa camera 2D công nghiệp (2 đến 4 camera GigE / USB).
"""

from .config import (
    TARGET_FPS, GEV_PACKET_SIZE, GEV_PACKET_DELAY,
    MERGE_DIST_CM, TRACK_DIST_CM, CALIB_TABLE_POINTS
)
from .camera_driver import IndustrialCamera, build_all_cameras, discover_cameras
from .detector import ObjectDetector
from .coordinator import MasterSlaveCoordinator, MasterTrackedObject
from .calibrate import (
    load_homographies, save_homographies,
    calibrate_camera_interactive, run_full_calibration
)

__all__ = [
    "TARGET_FPS",
    "GEV_PACKET_SIZE",
    "GEV_PACKET_DELAY",
    "MERGE_DIST_CM",
    "TRACK_DIST_CM",
    "CALIB_TABLE_POINTS",
    "IndustrialCamera",
    "build_all_cameras",
    "discover_cameras",
    "ObjectDetector",
    "MasterSlaveCoordinator",
    "MasterTrackedObject",
    "load_homographies",
    "save_homographies",
    "calibrate_camera_interactive",
    "run_full_calibration",
]
