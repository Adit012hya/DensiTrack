import numpy as np

from app import capture_system_webcam, select_image_input


def test_selects_webcam_frame_when_present():
    upload = object()
    webcam = np.zeros((10, 10, 3), dtype=np.uint8)

    assert select_image_input(upload, webcam) is webcam


def test_selects_upload_when_webcam_missing():
    upload = object()

    assert select_image_input(upload, None) is upload


def test_capture_system_webcam_handles_no_camera():
    assert capture_system_webcam(camera_index=999) is None
