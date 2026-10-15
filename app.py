#!/usr/bin/env python3
import datetime
import importlib.util
import os
import socket
import tempfile
from collections import defaultdict
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.solutions.heatmap import Heatmap


def resolve_gradio_port(preferred_port=7860, max_port=7869):
    """Find the first free local port in a small range so the app can launch even when 7860 is busy."""
    for port in range(preferred_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free Gradio port found in range {preferred_port}-{max_port}")

# -------------------------------------------------------------
# Helpers & Inference Pipeline
# -------------------------------------------------------------

def ensure_numpy_image(image):
    if hasattr(image, 'read') or hasattr(image, 'name'):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    return image


def select_image_input(uploaded_image, browser_webcam_frame, system_webcam_frame=None):
    if browser_webcam_frame is not None:
        return browser_webcam_frame
    if system_webcam_frame is not None:
        return system_webcam_frame
    return uploaded_image


def capture_system_webcam(camera_index=0):
    for index in (camera_index, 1, 2):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_dots_on_frame(frame, results, min_conf=0.3, min_box_area=0, track_history=None, tracking_enabled=False):
    annotated = frame.copy()
    if not results or len(results) == 0:
        return annotated
    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if conf < min_conf or area < min_box_area:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), -1)
        if tracking_enabled and hasattr(box, 'id') and box.id is not None and track_history is not None:
            track_id = int(box.id.item())
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            for i in range(1, len(track_history[track_id])):
                cv2.line(annotated, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)
    return annotated


def resolve_model_path(model_id):
    """Use a selected local checkpoint, or the confirmed local YOLO11 fallback."""
    if model_id in {"", None}:
        model_id = "yolo11n.pt"

    normalized = str(model_id).strip().strip('"').strip("'")
    if not normalized.lower().endswith((".pt", ".onnx", ".engine", ".torchscript")):
        normalized += ".pt"
    # Only use a requested model when its checkpoint is present. This prevents
    # hidden downloads and keeps every selector option runnable offline.
    return normalized if os.path.isfile(normalized) else "yolo11n.pt"


def detector_availability_message():
    person_ready = os.path.isfile("yolo11n.pt")
    face_weights_ready = os.path.isfile("yolov8m-face.pt")
    mediapipe_ready = importlib.util.find_spec("mediapipe") is not None
    person_status = "Local person model: YOLO11 Nano (yolo11n.pt)" if person_ready else "YOLO11 Nano weights are missing"
    face_status = []
    if not mediapipe_ready:
        face_status.append("MediaPipe is not installed")
    else:
        face_status.append("MediaPipe is available as a face-detection fallback")
    if not face_weights_ready:
        face_status.append("yolov8m-face.pt is missing")
    else:
        face_status.append("yolov8m-face.pt is used for face markers, separate from people counts")
    return f"{person_status}. Face tools are optional and are not used for people counts: {'; '.join(face_status)}."


MODEL_CHOICES = [
    ("yolo26n.pt", "YOLO26 Nano"),
    ("yolov8n.pt", "YOLOv8 Nano"),
    ("yolov8s.pt", "YOLOv8 Small"),
    ("yolov8m.pt", "YOLOv8 Medium"),
    ("yolov12n.pt", "YOLOv12 Nano"),
    ("yolov12s.pt", "YOLOv12 Small"),
    ("yolov12m.pt", "YOLOv12 Medium"),
    ("yolo11n.pt", "YOLO11 Nano"),
    ("best.pt", "Custom best.pt"),
]


def model_selector_options():
    options = []
    for model_id, label in MODEL_CHOICES:
        actual_model = resolve_model_path(model_id)
        if os.path.isfile(model_id) and actual_model == model_id:
            availability = "local weights"
        else:
            availability = f"uses {os.path.basename(actual_model)} fallback"
        selected = " selected" if model_id == "yolo11n.pt" else ""
        options.append(f'<option value="{model_id}"{selected}>{label} — {availability}</option>')
    return "\n".join(options)

def build_human_warning(count):
    if count >= 8:
        return "CRITICAL: crowd surge detected", "danger"
    if count >= 5:
        return "WARNING: dense human flow detected", "warning"
    if count >= 2:
        return "CAUTION: elevated human activity", "warning"
    return "NORMAL: safe human density", "ok"


def _predict_person_boxes(frame_bgr, model_id, conf_threshold=0.08, max_det=500):
    """Run one YOLO pass for person class and return normalized person boxes/centroids."""
    model = _get_cached_model(resolve_model_path(model_id))
    results = model.predict(
        source=frame_bgr,
        imgsz=960,
        conf=conf_threshold,
        classes=[0],
        max_det=max_det,
        iou=0.45,
        verbose=False,
        augment=True,
    )

    centroids = []
    boxes_out = []
    if results and len(results) > 0 and results[0].boxes is not None:
        h, w = frame_bgr.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cx_hm = cx / w * 480
            cy_hm = cy / h * 190
            radius = max(18, min(60, (y2 - y1) / h * 190 * 0.7))
            centroids.append({"x": round(cx_hm, 1), "y": round(cy_hm, 1), "radius": round(radius, 1)})
            boxes_out.append({
                "x1": round(x1 / w, 4), "y1": round(y1 / h, 4),
                "x2": round(x2 / w, 4), "y2": round(y2 / h, 4),
                "conf": round(conf, 2)
            })

    return centroids, boxes_out


def _predict_face_boxes(frame_bgr, conf_threshold=0.18):
    """Detect visible faces as supplemental evidence; never add them to person counts."""
    face_model_path = "yolov8m-face.pt"
    if os.path.isfile(face_model_path):
        try:
            model = _get_cached_model(face_model_path)
            results = model.predict(source=frame_bgr, imgsz=640, conf=conf_threshold,
                                    max_det=200, verbose=False)
            faces = []
            if results and results[0].boxes is not None:
                h, w = frame_bgr.shape[:2]
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    faces.append({"x1": round(x1/w, 4), "y1": round(y1/h, 4),
                                  "x2": round(x2/w, 4), "y2": round(y2/h, 4),
                                  "conf": round(float(box.conf[0]), 2), "source": "YOLOv8 Face"})
            return faces
        except Exception as exc:
            print(f"[DensiTrack] YOLO face model failed, trying MediaPipe: {exc}")

    if importlib.util.find_spec("mediapipe") is None:
        return []
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=conf_threshold
        ) as detector:
            result = detector.process(rgb)
        h, w = frame_bgr.shape[:2]
        faces = []
        for detection in result.detections or []:
            rel = detection.location_data.relative_bounding_box
            x1, y1 = max(0, rel.xmin), max(0, rel.ymin)
            x2, y2 = min(1, rel.xmin + rel.width), min(1, rel.ymin + rel.height)
            faces.append({"x1": round(x1, 4), "y1": round(y1, 4),
                          "x2": round(x2, 4), "y2": round(y2, 4),
                          "conf": round(float(detection.score[0]), 2), "source": "MediaPipe"})
        return faces
    except Exception as exc:
        print(f"[DensiTrack] Face detection unavailable: {exc}")
        return []


# -------------------------------------------------------------
# YOLO Real-Time Frame Inference Endpoint
# Called by JS every ~1.5s with a base64 JPEG captured from webcam
# Returns JSON: {count, centroids: [{x,y,radius}], boxes: [{x1,y1,x2,y2,conf}]}
# -------------------------------------------------------------
import base64
import json
import threading

# Cache the loaded model so we don't reload on every frame
_model_cache: dict = {}
_model_cache_lock = threading.Lock()

def _get_cached_model(model_id: str):
    """Load and cache YOLO model by model_id to avoid reloading every frame."""
    with _model_cache_lock:
        if model_id not in _model_cache:
            actual = resolve_model_path(model_id)
            try:
                _model_cache[model_id] = YOLO(actual)
                print(f"[DensiTrack] Loaded model: {actual}")
            except Exception as e:
                print(f"[DensiTrack] Failed to load {actual}: {e}")
                raise RuntimeError(f"Could not load {actual}: {e}") from e
        return _model_cache[model_id]


def predict_frame_base64(frame_b64: str, model_id: str) -> str:
    """Analyze one image for people; returns per-zone counts for a 2x2 grid overlay."""
    if not frame_b64:
        raise ValueError("No image frame was provided")
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(frame_b64, validate=True)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("The submitted frame is not a valid image")

    requested_model = resolve_model_path(model_id)
    candidates = [requested_model]
    fallback_model = resolve_model_path("yolo11n.pt")
    if fallback_model not in candidates:
        candidates.append(fallback_model)

    best_centroids, best_boxes = [], []
    active_model = requested_model
    inference_errors = []
    for candidate in candidates:
        try:
            centroids, boxes = _predict_person_boxes(
                frame_bgr, candidate, conf_threshold=0.04, max_det=500
            )
        except Exception as exc:
            inference_errors.append(f"{os.path.basename(candidate)}: {exc}")
            print(f"[DensiTrack] Person detection failed with {candidate}: {exc}")
            continue
        if len(centroids) > len(best_centroids):
            best_centroids, best_boxes = centroids, boxes
            active_model = candidate
        # The fallback is only needed when the selected model reports zero.
        if best_centroids:
            break

    if not best_centroids and inference_errors and len(inference_errors) == len(candidates):
        raise RuntimeError("All person detectors failed: " + "; ".join(inference_errors))

    # Compute per-zone counts using normalised box centres (2x2 grid)
    zone_counts = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}
    for b in best_boxes:
        cx = (b["x1"] + b["x2"]) / 2.0
        cy = (b["y1"] + b["y2"]) / 2.0
        if cx < 0.5 and cy < 0.5:
            zone_counts["zone1"] += 1   # top-left
        elif cx >= 0.5 and cy < 0.5:
            zone_counts["zone2"] += 1  # top-right
        elif cx < 0.5 and cy >= 0.5:
            zone_counts["zone3"] += 1  # bottom-left
        else:
            zone_counts["zone4"] += 1  # bottom-right

    warning, warning_level = build_human_warning(len(best_centroids))
    face_boxes = _predict_face_boxes(frame_bgr)
    return json.dumps({
        "count": len(best_centroids),
        "centroids": best_centroids,
        "boxes": best_boxes,
        "face_count": len(face_boxes),
        "faces": face_boxes,
        "model": os.path.basename(active_model),
        "warning": warning,
        "warning_level": warning_level,
        "zone_counts": zone_counts,
    })


def yolov12_tracker_inference(image, video, model_id, image_size, conf_threshold, mode, viz_mode, use_tracking):
    actual_model = resolve_model_path(model_id)
    try:
        model = YOLO(actual_model)
    except Exception:
        model = YOLO("yolov8n.pt")

    if hasattr(model, 'model') and hasattr(model.model, 'classes'):
        model.model.classes = [0]

    output_video_path = None
    count = 0

    if mode == "Original":
        if video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            return None, video_path, 0, "Original video loaded"
        return None, None, 0, "Original mode requires video input"

    if mode == "Detection":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0])
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes) if results and len(results) > 0 else 0
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_det.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            max_count = 0
            track_history = {}
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, classes=[0])
                current_count = len(results[0].boxes) if results and len(results) > 0 else 0
                if current_count > max_count:
                    max_count = current_count
                if viz_mode == "Dots":
                    annotated_frame = draw_dots_on_frame(frame, results, track_history=track_history, tracking_enabled=True)
                else:
                    annotated_frame = results[0].plot()
                out.write(annotated_frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max pedestrians in video: {max_count}"

    elif mode == "Tracking":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes) if results and len(results) > 0 else 0
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_track.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            track_history = defaultdict(list)
            max_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.track(frame, persist=True, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")[0]
                if results.boxes and results.boxes.id is not None:
                    boxes = results.boxes.xywh.cpu()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    annotated_frame = results.plot()
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((int(x), int(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                    out.write(annotated_frame)
                    if len(track_ids) > max_count:
                        max_count = len(track_ids)
                else:
                    out.write(frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max tracked pedestrians: {max_count}"

    elif mode == "Heatmap":
        if image is not None:
            image = ensure_numpy_image(image)
            return image, None, 0, "Heatmap for static image is calculated during video sequence"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_heatmap.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            heatmap = Heatmap(model=actual_model, imgsz=image_size, conf=conf_threshold, classes=[0])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                heatmap_frame = heatmap.generate_heatmap(frame)
                out.write(heatmap_frame)
            cap.release()
            out.release()
            return None, output_video_path, 0, "Heatmap processing complete"

    return None, None, 0, "No input provided"


# -------------------------------------------------------------
# Data Store for Multi-Camera & Alerts Management
# -------------------------------------------------------------

INITIAL_CAMERAS = [
    {
        "id": "CAM-01",
        "name": "CAM 01 - Main Entrance Plaza",
        "location": "North Gate Section A",
        "type": "System Hardware Webcam",
        "status": "LIVE",
        "assigned_name": "Vikram Rathore",
        "assigned_role": "Crowd Control Lead",
        "density": "Awaiting Analysis",
    },
    {
        "id": "CAM-02",
        "name": "CAM 02 - North Gate Concourse",
        "location": "Sector 3 Checkpoint",
        "type": "IP / RTSP Stream",
        "status": "LIVE",
        "assigned_name": "Sarah Connor",
        "assigned_role": "Perimeter Security Officer",
        "density": "Awaiting Analysis",
    },
    {
        "id": "CAM-03",
        "name": "CAM 03 - West Promenade & Stairs",
        "location": "West Plaza Walkway",
        "type": "PTZ Dome Optical",
        "status": "LIVE",
        "assigned_name": "Dr. Aarav Mehra",
        "assigned_role": "Medical First Responder",
        "density": "Awaiting Analysis",
    },
    {
        "id": "CAM-04",
        "name": "CAM 04 - Sector B Queue Corridor",
        "location": "Central Pavilion Lane 4",
        "type": "Infrared Thermal / Optical",
        "status": "WARNING",
        "assigned_name": "Rajesh Sharma",
        "assigned_role": "Emergency Dispatch Commander",
        "density": "Awaiting Analysis",
    },
]

INITIAL_ALERTS = []

# -------------------------------------------------------------
# HTML Rendering Utilities for Visuvate Dark Aesthetic
# -------------------------------------------------------------

def render_camera_sidebar_html(cameras, active_cam_id=None):
    items = []
    for idx, cam in enumerate(cameras):
        is_selected = (cam["id"] == active_cam_id)
        border_style = "border: 1px solid #0072f5; background: #121829;" if is_selected else "border: 1px solid #1c1c22; background: #0f0f13;"
        status_color = "#22c55e" if cam.get("status") == "LIVE" else "#ef4444"
        items.append(f"""
        <div class="sidebar-cam-card" onclick="selectCameraPage('{cam['id']}')" style="display: flex; align-items: center; justify-content: space-between; padding: 9px 12px; border-radius: 8px; margin-bottom: 5px; {border_style} font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s ease;">
            <div style="display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                <span style="overflow: hidden; text-overflow: ellipsis; color: #ffffff; font-weight: 500;">{cam['name']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: {status_color};"></span>
                <span style="font-size: 11px; opacity: 0.8; color: #a1a1aa;">{cam['id']}</span>
            </div>
        </div>
        """)
    return f"""
    <div style="display: flex; flex-direction: column; gap: 2px;">
        {''.join(items)}
    </div>
    """


def render_camera_detail_full_html(cam=None, alerts=None):
    if cam is None:
        cam = INITIAL_CAMERAS[0]
    if alerts is None:
        alerts = INITIAL_ALERTS

    cam_id = cam.get("id", "CAM-01")
    officer_name = cam.get("assigned_name", "Vikram Rathore")
    officer_role = cam.get("assigned_role", "Crowd Control Lead")
    location = cam.get("location", "North Gate Section A")
    initial = officer_name[0] if officer_name else "V"
    density_str = cam.get("density", "Awaiting Analysis")
    density_pct = 0
    density_color = '#a1a1aa'
    chokepoint_label = 'Standby'

    cam_alerts = [a for a in alerts if a.get("camera_id") == cam_id or a.get("camera_name") == cam.get("name")]
    alert_rows = []
    if cam_alerts:
        for idx, a in enumerate(cam_alerts[:4]):
            badge_bg = "rgba(239, 68, 68, 0.18)" if a["severity"] == "Critical" else ("rgba(245, 158, 11, 0.18)" if a["severity"] == "Warning" else "rgba(34, 197, 94, 0.18)")
            badge_color = "#f87171" if a["severity"] == "Critical" else ("#fbbf24" if a["severity"] == "Warning" else "#4ade80")
            badge_border = "#ef4444" if a["severity"] == "Critical" else ("#f59e0b" if a["severity"] == "Warning" else "#22c55e")
            alert_rows.append(f"""
            <div style="background: #16161c; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-size: 10px; font-weight: 700; color: #a1a1aa; text-transform: uppercase;">Type:</span>
                        <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;">{a['severity']}</span>
                    </div>
                    <span style="font-size: 10px; font-family: monospace; color: #a1a1aa;">{a['timestamp']}</span>
                </div>
                <div style="font-size: 11px; color: #ffffff; line-height: 1.4; margin-bottom: 6px;">
                    <b>Details:</b> {a['message']}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f1f26; padding-top: 6px;">
                    <span style="font-size: 10px; color: #93c5fd;">Assigned: {officer_name}</span>
                    <button onclick="sendAlertToPersonnel('{cam_id}', '{a['severity']}', '{officer_name}')" style="background: #ef4444; color: #ffffff; border: none; padding: 4px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer; transition: background 0.2s;" onmouseover="this.style.background='#dc2626'" onmouseout="this.style.background='#ef4444'">
                        Send Alert
                    </button>
                </div>
            </div>
            """)
    else:
        alert_rows.append(f"""
        <div style="padding: 14px; background: #16161c; border: 1px dashed #272730; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 12px;">
            No active security incidents flagged for {cam_id}. Area status is nominal.
        </div>
        """)
    alerts_markup = "".join(alert_rows)

    return f"""
    <!-- Top Header Bar with Back Button and Model Selector Dropdown -->
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; background: #121216; border: 1px solid #23232b; border-radius: 12px; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <button onclick="returnToHome()" class="btn-visuvate-filter" style="display: inline-flex; align-items: center; gap: 6px; background: #18181f; color: #ffffff; border: 1px solid #2c2c36; padding: 7px 14px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s ease;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
                Back to Home
            </button>
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span id="detail-cam-title" style="font-size: 16px; font-weight: 700; color: #ffffff;">{cam.get('name', 'Camera View')}</span>
                    <span id="detail-cam-id-badge" style="font-size: 11px; background: rgba(0, 114, 245, 0.18); color: #0072f5; border: 1px solid rgba(0, 114, 245, 0.4); padding: 2px 8px; border-radius: 6px; font-weight: 600;">{cam.get('id', 'CAM-01')}</span>
                </div>
                <div style="font-size: 11px; color: #71717a; margin-top: 2px;">
                    Monitored Sector: <span id="detail-cam-location" style="color: #d4d4d8; font-weight: 500;">{location}</span> • Source: <span id="detail-cam-type" style="color: #a1a1aa;">{cam.get('type', 'Standard Feed')}</span>
                </div>
            </div>
        </div>
        
        <!-- Model Selection Dropdown (Replacing static ● REC badge) -->
        <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 5px;">
            <div style="display: flex; align-items: center; gap: 8px; background: #16161c; border: 1px solid #282832; padding: 4px 10px; border-radius: 8px;">
                <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: #a1a1aa; font-weight: 600;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><path d="M7 7h10v10H7z"/></svg>
                    AI Model:
                </span>
                <select id="detail-model-select" onchange="onModelChanged(this.value)" style="background: #0e0e12; color: #ffffff; border: 1px solid #33333f; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600; outline: none; cursor: pointer;">
                    {model_selector_options()}
                </select>
                <span id="model-active-indicator" style="width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e;"></span>
            </div>
            <div style="font-size: 9px; line-height: 1.35; color: #a1a1aa; max-width: 340px;">{detector_availability_message()}</div>
        </div>

        <div id="media-upload-bar" style="display: none; align-items: center; gap: 8px; margin-top: 10px; width: 100%; justify-content: flex-end; flex-wrap: wrap;">
            <input id="media-upload-input" type="file" accept="video/*,image/*" style="display: none;" />
            <button id="media-upload-btn" onclick="triggerMediaUpload()" style="background: linear-gradient(135deg, #0072f5, #2563eb); color: #ffffff; border: none; border-radius: 8px; padding: 8px 14px; font-size: 11px; font-weight: 700; cursor: pointer;">Upload Video</button>
            <button id="media-remove-btn" onclick="removeUploadedMedia()" style="display: none; background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #ffffff; border: none; border-radius: 8px; padding: 8px 14px; font-size: 11px; font-weight: 700; cursor: pointer;">✕ Remove Video</button>
            <button id="media-analyze-btn" onclick="startUploadedMediaAnalysisLoop()" style="display: none; background: linear-gradient(135deg, #16a34a, #15803d); color: #ffffff; border: none; border-radius: 8px; padding: 8px 14px; font-size: 11px; font-weight: 700; cursor: pointer;">▶ Analyse Video</button>
        </div>
    </div>

    <!-- Center Camera Feed: CAM-01 Live Webcam Box -->
    <div id="cam-detail-webcam-box" style="position: relative; width: 100%; min-height: 380px; max-height: 480px; background: #000000; border-radius: 14px; overflow: hidden; border: 1px solid #23232b; box-shadow: 0 10px 30px rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
        <video id="cam-detail-video" autoplay playsinline muted style="width: 100%; height: 100%; max-height: 480px; object-fit: contain; display: none; background: #000;"></video>
        <!-- YOLO Detection Bounding Box Overlay Canvas -->
        <canvas id="cam-detection-overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 5;"></canvas>
        <div id="cam-detail-placeholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding: 24px; text-align: center; width: 100%; min-height: 380px;">
            <div style="width: 52px; height: 52px; border-radius: 50%; background: rgba(0, 114, 245, 0.2); border: 1px solid rgba(0, 114, 245, 0.5); display: flex; align-items: center; justify-content: center; color: #0072f5; box-shadow: 0 0 24px rgba(0, 114, 245, 0.3);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            </div>
            <div style="color: #ffffff; font-size: 15px; font-weight: 600;">Live Camera Feed</div>
            <div style="color: #71717a; font-size: 12px; max-width: 320px;">Click <b>Start Recording</b> below to begin live camera capture and crowd analysis.</div>
        </div>
        <!-- CCTV HUD Overlay -->
        <div style="position: absolute; top: 12px; left: 16px; display: flex; align-items: center; gap: 8px; pointer-events: none; z-index: 10;">
            <span id="cam-rec-badge" style="font-size: 10px; font-family: monospace; font-weight: 700; color: #71717a; background: rgba(0,0,0,0.7); border: 1px solid rgba(113,113,122,0.4); padding: 3px 8px; border-radius: 4px;">● IDLE</span>
            <span style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;">PRIMARY WEBCAM</span>
            <span id="cam-model-badge" style="font-size: 10px; font-family: monospace; color: #60a5fa; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #1e3a8a;">MODEL: yolo11n.pt</span>
        </div>
        <div style="position: absolute; bottom: 12px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center; z-index: 10;">
            <span id="cam-detail-person-metric" style="font-size: 10px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● IDLE • ANALYZED PERSONS: 0</span>
            <!-- Start / End Recording buttons -->
            <div style="display: flex; gap: 8px;">
                <button id="btn-start-recording" onclick="startCameraRecording()" style="background: linear-gradient(135deg, #16a34a, #15803d); color: #fff; border: none; border-radius: 7px; padding: 6px 14px; font-size: 11px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:#4ade80;display:inline-block;"></span> Start Recording
                </button>
                <button id="btn-end-recording" onclick="stopCameraRecording()" disabled style="background: #1c1c24; color: #71717a; border: 1px solid #2c2c36; border-radius: 7px; padding: 6px 14px; font-size: 11px; font-weight: 700; cursor: not-allowed; display: flex; align-items: center; gap: 5px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:#71717a;display:inline-block;"></span> End Recording
                </button>
            </div>
            <span id="cam-detail-timestamp-webcam" style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;"></span>
        </div>
    </div>

    <!-- Center Camera Feed: Optical Canvas Box (For Other Cameras - Uncropped object-fit: contain) -->
    <div id="cam-detail-optical-box" style="position: relative; width: 100%; min-height: 380px; max-height: 480px; background: #000000; border-radius: 14px; overflow: hidden; border: 1px solid #23232b; box-shadow: 0 10px 30px rgba(0,0,0,0.6); display: none; align-items: center; justify-content: center; margin-bottom: 12px;">
        <video id="optical-live-video" autoplay playsinline muted style="display:none;"></video>
        <canvas id="cctv-optical-stream-canvas" width="640" height="380" style="width: 100%; height: 100%; max-height: 480px; object-fit: contain; background: #000; display: block;"></canvas>
        <button id="btn-optical-live" onclick="startOpticalLiveCamera()" style="position:absolute;top:48px;right:16px;z-index:12;background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;border:0;border-radius:7px;padding:7px 12px;font-size:11px;font-weight:700;cursor:pointer;">Start Live Camera</button>
        <div style="position: absolute; top: 12px; left: 16px; display: flex; align-items: center; gap: 8px; pointer-events: none;">
            <span style="font-size: 10px; font-family: monospace; font-weight: 700; color: #22c55e; background: rgba(0,0,0,0.7); border: 1px solid rgba(34, 197, 94, 0.4); padding: 3px 8px; border-radius: 4px;">● RTSP STREAM</span>
            <span style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;">OPTICAL SURVEILLANCE</span>
            <span id="optical-model-badge" style="font-size: 10px; font-family: monospace; color: #60a5fa; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #1e3a8a;">MODEL: yolo11n.pt</span>
        </div>
        <div style="position: absolute; bottom: 12px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center; pointer-events: none;">
            <span id="cam-optical-person-metric" style="font-size: 10px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● WAITING FOR LIVE ANALYSIS • ANALYZED PERSONS: 0</span>
            <span id="cam-detail-timestamp-optical" style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;"></span>
        </div>
    </div>

    <!-- Bottom Split: Heatmap on Left, Personnel Allotted & Alerts on Right -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
        <!-- Left Side: Heatmap based on Crowd from Feed -->
        <div style="background: #111115; border: 1px solid #22222a; border-radius: 12px; padding: 14px; min-height: 290px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 14px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 6px;">
                        Heat Map
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span id="detail-heatmap-count-badge" style="font-size: 10px; background: rgba(0, 114, 245, 0.18); color: #60a5fa; border: 1px solid rgba(0, 114, 245, 0.4); padding: 2px 7px; border-radius: 5px; font-weight: 600;">0 Persons Detected</span>
                    <span style="font-size: 10px; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 7px; border-radius: 5px; font-weight: 600;">JET THERMAL MAP</span>
                </div>
            </div>

            <!-- Heatmap Composite Canvas: renders actual video frame + heatmap overlay -->
            <div style="position: relative; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #1f1f27;">
                <canvas id="heatmap-composite-canvas" width="480" height="190" style="width: 100%; height: 190px; display: block; background: #06070a;"></canvas>
                <div id="heatmap-idle-msg" style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#3f3f46;font-size:11px;font-family:monospace;pointer-events:none;">Heatmap will appear after analysis starts</div>
                <div style="position: absolute; bottom: 8px; right: 10px; display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.75); padding: 4px 8px; border-radius: 6px; border: 1px solid #27272a;">
                    <span style="font-size: 9px; color: #60a5fa; font-family: monospace;">LOW</span>
                    <div style="width: 70px; height: 8px; border-radius: 4px; background: linear-gradient(to right, #3b82f6, #06b6d4, #22c55e, #eab308, #ef4444);"></div>
                    <span style="font-size: 9px; color: #ef4444; font-family: monospace;">HIGH</span>
                </div>
            </div>

            <!-- Zone Status Panel: 4 live zone cards + summary stats -->
            <div style="margin-top: 10px;">
                <div style="font-size: 10px; color: #71717a; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Live Zone Status (2×2 Grid)</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">
                    <div id="zone-panel-zone1" style="background: rgba(34,197,94,0.08); border: 1px solid #22c55e; border-radius: 8px; padding: 8px 10px; transition: all 0.3s ease;">
                        <div style="font-size: 9px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px;">Zone 1 — Top Left</div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="zone-count" style="font-size: 20px; font-weight: 800; color: #4ade80;">0</span>
                            <span class="zone-status" style="font-size: 9px; font-weight: 700; color: #4ade80;">✓ NORMAL</span>
                        </div>
                    </div>
                    <div id="zone-panel-zone2" style="background: rgba(34,197,94,0.08); border: 1px solid #22c55e; border-radius: 8px; padding: 8px 10px; transition: all 0.3s ease;">
                        <div style="font-size: 9px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px;">Zone 2 — Top Right</div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="zone-count" style="font-size: 20px; font-weight: 800; color: #4ade80;">0</span>
                            <span class="zone-status" style="font-size: 9px; font-weight: 700; color: #4ade80;">✓ NORMAL</span>
                        </div>
                    </div>
                    <div id="zone-panel-zone3" style="background: rgba(34,197,94,0.08); border: 1px solid #22c55e; border-radius: 8px; padding: 8px 10px; transition: all 0.3s ease;">
                        <div style="font-size: 9px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px;">Zone 3 — Bottom Left</div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="zone-count" style="font-size: 20px; font-weight: 800; color: #4ade80;">0</span>
                            <span class="zone-status" style="font-size: 9px; font-weight: 700; color: #4ade80;">✓ NORMAL</span>
                        </div>
                    </div>
                    <div id="zone-panel-zone4" style="background: rgba(34,197,94,0.08); border: 1px solid #22c55e; border-radius: 8px; padding: 8px 10px; transition: all 0.3s ease;">
                        <div style="font-size: 9px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px;">Zone 4 — Bottom Right</div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="zone-count" style="font-size: 20px; font-weight: 800; color: #4ade80;">0</span>
                            <span class="zone-status" style="font-size: 9px; font-weight: 700; color: #4ade80;">✓ NORMAL</span>
                        </div>
                    </div>
                </div>
                <!-- Summary stat row below zone cards -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    <div style="background: #16161c; border: 1px solid #24242e; border-radius: 8px; padding: 6px 10px;">
                        <div style="font-size: 9px; color: #71717a; text-transform: uppercase;">Chokepoint</div>
                        <div id="detail-chokepoint-val" style="font-size: 12px; font-weight: 700; color: #a1a1aa; margin-top: 1px;">Standby</div>
                    </div>
                    <div style="background: #16161c; border: 1px solid #24242e; border-radius: 8px; padding: 6px 10px;">
                        <div style="font-size: 9px; color: #71717a; text-transform: uppercase;">Crowd Density</div>
                        <div id="detail-density-val" style="font-size: 12px; font-weight: 700; color: #ffffff; margin-top: 1px;">Awaiting</div>
                    </div>
                </div>
                <div style="display: none;"><span id="detail-model-tag">yolo11n.pt</span></div>
            </div>
        </div>

        <!-- Right Side: Personnel Allotted & Area Incident Alerts Side by Side -->
        <div style="background: #111115; border-radius: 12px; padding: 14px; min-height: 290px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
            <!-- Column 1: Personnel Allotted in Area -->
            <div style="display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #ffffff; font-weight: 700; letter-spacing: 0.5px;">
                        Personnel Allotted in Area
                    </div>
                    <button id="btn-personnel-edit" onclick="toggleAddPersonnelForm()" style="background: #181822; color: #ffffff; border: none; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 4px;">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                        Edit
                    </button>
                </div>

                <!-- Edit / Add Personnel Form (Name, Position, Contact No) -->
                <div id="add-personnel-form" style="display: none; background: #16161e; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="font-size: 11px; color: #ffffff; font-weight: 600; margin-bottom: 6px;">Add Officer:</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px;">
                        <input id="input-personnel-name" type="text" placeholder="Officer Name" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;" />
                        <input id="input-personnel-phone" type="text" placeholder="Contact No (+91 98402 11029)" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;" />
                        <select id="input-personnel-role" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;">
                            <option value="Crowd Control Lead">Crowd Control Lead</option>
                            <option value="Perimeter Security Officer">Perimeter Security Officer</option>
                            <option value="Medical First Responder">Medical First Responder</option>
                            <option value="Emergency Dispatch Commander">Emergency Dispatch Commander</option>
                            <option value="Sector Safety Officer">Sector Safety Officer</option>
                        </select>
                    </div>
                    <div style="display: flex; justify-content: flex-end; gap: 6px;">
                        <button onclick="toggleAddPersonnelForm()" style="background: #202028; color: #ffffff; border: none; padding: 4px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;">Cancel</button>
                        <button onclick="submitAddPersonnel()" style="background: #0072f5; color: #ffffff; border: none; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;">Add</button>
                    </div>
                </div>

                <!-- Personnel Dynamic Cards List -->
                <div id="detail-personnel-list" style="display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto;">
                    <div class="personnel-card" data-idx="0" style="display: flex; align-items: center; justify-content: space-between; background: #16161c; border-radius: 8px; padding: 8px 10px;">
                        <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                            <div class="officer-avatar" style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0072f5, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #ffffff; flex-shrink: 0;">
                                {initial}
                            </div>
                            <div style="overflow: hidden;">
                                <div style="font-size: 12px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 5px;">
                                    <span class="officer-name-text" style="color: #ffffff;">{officer_name}</span>
                                    <span style="width: 5px; height: 5px; border-radius: 50%; background: #22c55e;"></span>
                                </div>
                                <div class="officer-role-text" style="font-size: 10px; color: #93c5fd; font-weight: 500;">{officer_role}</div>
                            </div>
                        </div>
                        <button class="personnel-delete-btn" onclick="deletePersonnel(0)" title="Delete Officer from Area" style="display: none; background: transparent; border: none; color: #a1a1aa; cursor: pointer; padding: 4px; border-radius: 4px;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#a1a1aa'">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Column 2: Area Incident Alerts -->
            <div style="display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 11px; text-transform: uppercase; color: #ffffff; font-weight: 700; letter-spacing: 0.5px;">Area Incident Alerts</span>
                </div>
                <div id="detail-alerts-list" style="max-height: 220px; overflow-y: auto;">
                    {alerts_markup}
                </div>
            </div>
        </div>
    </div>
    """


def render_alerts_table_html(alerts, filter_severity="All", filter_cam="All"):
    filtered = []
    for a in alerts:
        if filter_severity != "All" and a["severity"] != filter_severity:
            continue
        if filter_cam != "All" and (a["camera_id"] != filter_cam and a["camera_name"] != filter_cam):
            continue
        filtered.append(a)

    rows = []
    for a in filtered:
        badge_bg = "rgba(239, 68, 68, 0.15)" if a["severity"] == "Critical" else ("rgba(245, 158, 11, 0.15)" if a["severity"] == "Warning" else "rgba(34, 197, 94, 0.15)")
        badge_color = "#f87171" if a["severity"] == "Critical" else ("#fbbf24" if a["severity"] == "Warning" else "#4ade80")
        badge_border = "#ef4444" if a["severity"] == "Critical" else ("#f59e0b" if a["severity"] == "Warning" else "#22c55e")
        rows.append(f"""
        <tr style="border-bottom: 1px solid #1a1a1f;">
            <td style="padding: 8px 12px; font-family: monospace; font-size: 11px; color: #71717a;">{a['timestamp']}</td>
            <td style="padding: 8px 12px; font-weight: 600; color: #f4f4f5; font-size: 12px;">{a['camera_name']}</td>
            <td style="padding: 8px 12px;">
                <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}50; padding: 2px 7px; border-radius: 5px; font-size: 10px; font-weight: 600;">{a['severity']}</span>
            </td>
            <td style="padding: 8px 12px; color: #d4d4d8; font-size: 12px;">{a['message']}</td>
            <td style="padding: 8px 12px; color: #60a5fa; font-size: 11px; font-weight: 500;">{a['assigned_person']}</td>
            <td style="padding: 8px 12px;">
                <span style="background: #141416; border: 1px solid #27272a; color: #22c55e; padding: 2px 7px; border-radius: 10px; font-size: 10px; font-weight: 500;">{a['status']}</span>
            </td>
        </tr>
        """)

    if not rows:
        tbody = """<tr><td colspan="6" style="padding: 16px; text-align: center; color: #71717a; font-style: italic;">No alerts found matching current filter criteria.</td></tr>"""
    else:
        tbody = "".join(rows)

    return f"""
    <div style="overflow-x: auto; background: #0c0c0e; border: 1px solid #1f1f24; border-radius: 10px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
            <thead>
                <tr style="background: #141417; color: #a1a1aa; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #232328;">
                    <th style="padding: 8px 12px;">Time</th>
                    <th style="padding: 8px 12px;">Camera Source</th>
                    <th style="padding: 8px 12px;">Severity</th>
                    <th style="padding: 8px 12px;">Incident Details</th>
                    <th style="padding: 8px 12px;">Assigned Personnel</th>
                    <th style="padding: 8px 12px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {tbody}
            </tbody>
        </table>
    </div>
    """


def render_full_personnel_table(cameras):
    rows = []
    for c in cameras:
        rows.append(f"""
        <tr style="border-bottom: 1px solid #1c1c22;">
            <td style="padding: 9px 14px; font-weight: 600; color: #f4f4f5;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 26px; height: 26px; border-radius: 50%; background: #0072f5; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: white;">
                        {c['assigned_name'][0]}
                    </div>
                    <span>{c['assigned_name']}</span>
                </div>
            </td>
            <td style="padding: 9px 14px; color: #0072f5; font-weight: 600;">{c['assigned_role']}</td>
            <td style="padding: 9px 14px; color: #a1a1aa; font-family: monospace;">{c['id']} ({c['name']})</td>
            <td style="padding: 9px 14px; color: #e4e4e7;">{c['location']}</td>
            <td style="padding: 9px 14px;">
                <span style="background: rgba(34, 197, 94, 0.12); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); padding: 2px 8px; border-radius: 6px; font-size: 10px; font-weight: 600;">● Active on Duty</span>
            </td>
        </tr>
        """)
    return f"""
    <div style="background: #0c0c0e; border: 1px solid #1f1f24; border-radius: 10px; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
            <thead>
                <tr style="background: #141418; color: #a1a1aa; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; border-bottom: 1px solid #232328;">
                    <th style="padding: 9px 14px;">Officer / Responder Name</th>
                    <th style="padding: 9px 14px;">Assigned Role</th>
                    <th style="padding: 9px 14px;">Assigned Camera</th>
                    <th style="padding: 9px 14px;">Monitored Patrol Section</th>
                    <th style="padding: 9px 14px;">Duty Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


# -------------------------------------------------------------
# Gradio UI Construction with Visuvate Design System
# -------------------------------------------------------------

VISUVATE_CSS = """
/* Reset & Zero out space above header */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #000000 !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 16px 20px 16px !important;
    margin: 0 !important;
    background-color: #000000 !important;
    background-image: 
        radial-gradient(circle at 0% 0%, rgba(0, 114, 245, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 100% 0%, rgba(0, 114, 245, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(0, 114, 245, 0.12) 0%, transparent 50%),
        #000000 !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.gradio-container > .main, #root, .gradio-container div.app {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Glassmorphic Framer Workspace Canvas */
.visuvate-canvas {
    background: #09090b !important;
    border: 1px solid #202026 !important;
    border-radius: 16px !important;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.9), 0 0 40px rgba(0, 114, 245, 0.12) !important;
    overflow: hidden !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}

/* Quad Grid Camera Cards */
.cctv-quad-card {
    background: #0d0d10 !important;
    border: 1px solid #222228 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
    transition: border-color 0.2s ease !important;
    display: flex !important;
    flex-direction: column !important;
}
.cctv-quad-card:hover {
    border-color: #0072f5 !important;
}

.cctv-quad-header {
    background: #141418 !important;
    border-bottom: 1px solid #202026 !important;
    padding: 6px 10px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 11px !important;
}

.rec-pill {
    background: #ef4444 !important;
    color: #ffffff !important;
    font-size: 9px !important;
    font-weight: 700 !important;
    padding: 2px 5px !important;
    border-radius: 3px !important;
    letter-spacing: 0.04em !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

/* Plus Slot in Quadrants */
.cctv-plus-slot {
    height: 165px;
    background: #070709;
    border: 1px dashed #23232c;
    border-radius: 8px;
    margin: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}
.cctv-plus-slot:hover {
    border-color: #0072f5;
    background: rgba(0, 114, 245, 0.05);
}
.cctv-plus-slot .plus-sign {
    font-size: 36px;
    line-height: 1;
    color: #0072f5;
    font-weight: 300;
    margin-bottom: 6px;
}
.cctv-plus-slot .plus-label {
    font-size: 13px;
    font-weight: 600;
    color: #f4f4f5;
}
.cctv-plus-slot .plus-sub {
    font-size: 11px;
    color: #71717a;
    margin-top: 3px;
}
.dashboard-camera-placeholder { position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px;background:#08080bcc;color:#f4f4f5;text-align:center;font-size:12px;pointer-events:none }
.dashboard-camera-placeholder span { color:#a1a1aa;font-size:10px }
.dashboard-camera-count { position:absolute;left:8px;top:7px;background:#050508bb;border:1px solid #3333;border-radius:5px;padding:3px 7px;font:10px monospace;pointer-events:none }

/* Eradicate Gradio Image Toolbar & HUD Controls (▶, 🔍, ✋, 1X, 🌙, 🔲) */
.gradio-container .image-toolbar,
.gradio-container .toolbar,
.gradio-container div[data-testid="image-toolbar"],
.gradio-container button.tool,
.gradio-container button[aria-label="Zoom in"],
.gradio-container button[aria-label="Zoom out"],
.gradio-container button[aria-label="Reset"],
.gradio-container button[aria-label="Pan"],
.gradio-container button[title="Zoom"],
.gradio-container button[title="Pan"],
.gradio-container button[title="Reset"],
.gradio-container button[aria-label="Clear"],
.gradio-container .svelte-1p9fsmv,
.cctv-quad-card .image-toolbar,
.cctv-quad-card .toolbar,
.cctv-quad-card button.tool {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Eradicate ALL White Boxes / White Badges from Form Controls & Dropdowns */
.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container fieldset,
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container .input-container,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container div[data-testid],
.gradio-container .panel,
.gradio-container .tab-item,
.gradio-container [class*="svelte-"] {
    background-color: transparent !important;
    border: none !important;
    color: #f4f4f5 !important;
    box-shadow: none !important;
}

/* Target Labels to eliminate white badge effect */
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container span[data-testid="block-info"],
.gradio-container .gr-input-label,
.gradio-container span.svelte-1ed2p3z,
.gradio-container label.svelte-1ed2p3z,
.gradio-container span.svelte-11l35i6,
.gradio-container span.svelte-1gf128n {
    background: transparent !important;
    background-color: transparent !important;
    color: #a1a1aa !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding-left: 0 !important;
}

/* Input, Select, Textarea Dark styling - SOLID OPAQUE DARK BACKGROUND */
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap-inner,
.gradio-container .single-select,
select {
    background: #111116 !important;
    background-color: #111116 !important;
    color: #ffffff !important;
    border: 1px solid #23232c !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

/* Ensure all select options and dropdown buttons have solid background and crisp white text */
select,
select option,
.gradio-container select option,
.gradio-container .secondary-wrap,
.gradio-container .secondary-wrap span,
.gradio-container .single-select,
.gradio-container .single-select span,
.gradio-container .single-select input,
.gradio-container ul.options,
.gradio-container ul.options li,
.gradio-container .dropdown-button,
.gradio-container div[data-testid="dropdown"],
.gradio-container div[data-testid="dropdown"] button,
.gradio-container div[data-testid="dropdown"] span,
.gradio-container div[data-testid="dropdown"] input {
    color: #ffffff !important;
    background: #111116 !important;
    background-color: #111116 !important;
    border-color: #23232c !important;
    font-size: 12px !important;
    opacity: 1 !important;
}

select option:hover,
select option:checked,
.gradio-container ul.options li:hover,
.gradio-container ul.options li.selected {
    background-color: #1c1c24 !important;
    color: #ffffff !important;
}

/* Accordion Dark obsidian styling */
.gradio-container .accordion,
.gradio-container details {
    background: #0d0d11 !important;
    border: 1px solid #202026 !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    overflow: hidden !important;
}
.gradio-container .accordion > .label-wrap,
.gradio-container details > summary {
    background: #141419 !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    color: #f4f4f5 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Custom Visuvate Button Styles */
button.btn-visuvate-filter {
    background: #0072f5 !important;
    background-color: #0072f5 !important;
    color: #ffffff !important;
    border: 1px solid #0072f5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-filter:hover {
    background: #005bc4 !important;
    box-shadow: none !important;
}

button.btn-visuvate-alert {
    background: #ef4444 !important;
    background-color: #ef4444 !important;
    color: #ffffff !important;
    border: 1px solid #ef4444 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 9px 20px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-alert:hover {
    background: #dc2626 !important;
    box-shadow: none !important;
}

button.btn-visuvate-capture {
    background: #18181d !important;
    background-color: #18181d !important;
    color: #e4e4e7 !important;
    border: 1px solid #272732 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 5px 10px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-capture:hover {
    border-color: #0072f5 !important;
    color: #ffffff !important;
}

/* Hide Gradio Default Footer & API links */
footer,
.gradio-container footer,
.show-api,
.api-docs,
a[href*="gradio.app"],
a[href*="huggingface.co"],
.built-with {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Floating Glow Navigation Bar */
.floating-nav-wrapper {
    position: sticky;
    top: 6px;
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 6px 0 12px 0;
    width: 100%;
}

.glow-menu-bar {
    position: relative;
    display: inline-flex;
    align-items: center;
    padding: 5px 8px;
    border-radius: 20px;
    background: rgba(15, 15, 19, 0.82) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.65), 0 0 1px 1px rgba(255, 255, 255, 0.06) !important;
    overflow: hidden;
    transition: all 0.3s ease;
}

.glow-menu-bg {
    position: absolute;
    inset: -6px;
    border-radius: 26px;
    background: radial-gradient(circle at 50% 50%, rgba(59,130,246,0.2) 0%, rgba(168,85,247,0.15) 35%, rgba(239,68,68,0.12) 75%, transparent 100%);
    opacity: 0;
    transition: opacity 0.5s ease;
    pointer-events: none;
    z-index: 0;
}
.glow-menu-bar:hover .glow-menu-bg {
    opacity: 1;
}

.glow-menu-list {
    display: flex;
    align-items: center;
    gap: 4px;
    list-style: none;
    margin: 0;
    padding: 0;
    position: relative;
    z-index: 1;
}

.glow-menu-item {
    position: relative;
}

.glow-menu-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 14px;
    background: transparent;
    border: none;
    color: #ffffff !important;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    position: relative;
    transition: color 0.25s ease;
    outline: none;
    user-select: none;
}

.glow-menu-link span {
    color: #ffffff !important;
}

.glow-menu-link:hover,
.glow-menu-item.active .glow-menu-link {
    color: #ffffff !important;
}

.glow-pill-backdrop {
    position: absolute;
    inset: 0;
    border-radius: 14px;
    opacity: 0;
    transform: scale(0.85);
    transition: opacity 0.35s ease, transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    pointer-events: none;
    z-index: -1;
}

.glow-menu-item:hover .glow-pill-backdrop,
.glow-menu-item.active .glow-pill-backdrop {
    opacity: 1;
    transform: scale(1);
}

.glow-icon-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    color: #ffffff !important;
    stroke: #ffffff !important;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.glow-icon-wrap svg,
.glow-menu-link svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
}

.glow-menu-item:hover .glow-icon-wrap,
.glow-menu-item.active .glow-icon-wrap {
    transform: scale(1.18);
}

.sidebar-cam-card:hover {
    background: #181822 !important;
    border-color: #0072f5 !important;
    transform: translateX(2px);
}

#cam-detail-page-container {
    display: none !important;
}
#cam-detail-page-container.active {
    display: block !important;
}
#home-page-container.hidden {
    display: none !important;
}
#dispatch-page-container { display: none !important; }
#dispatch-page-container.active { display: block !important; }
#settings-page-container { display: none !important; }
#settings-page-container.active { display: block !important; }
"""

HEAD_JS = """
<script>
window.cctvGlobalStream = null;
window.selectedModelId = 'yolo11n.pt';
window.currentActiveCamId = 'CAM-01';
window.cameraDashboardData = {};
window.uploadedMediaByCam = {};
window.dashboardAnalysisBusy = false;
window.alertThresholds = { crowding: 30, overcrowding: 50 };
try {
    const savedThresholds = JSON.parse(localStorage.getItem('densitrack-alert-thresholds') || '{}');
    if (Number.isFinite(savedThresholds.crowding) && Number.isFinite(savedThresholds.overcrowding) && savedThresholds.overcrowding > savedThresholds.crowding) window.alertThresholds = savedThresholds;
} catch (_) {}
function isCrowdingCount(count) { return Number(count) > window.alertThresholds.crowding; }
function isOvercrowdingCount(count) { return Number(count) > window.alertThresholds.overcrowding; }
function saveAlertThresholds(crowdingValue, overcrowdingValue) {
    const crowding = Math.max(1, Math.min(100, Number(crowdingValue) || 30));
    const overcrowding = Math.max(crowding + 1, Math.min(150, Number(overcrowdingValue) || 50));
    window.alertThresholds = { crowding, overcrowding };
    try { localStorage.setItem('densitrack-alert-thresholds', JSON.stringify(window.alertThresholds)); } catch (_) {}
    const crowdSlider = document.getElementById('settings-crowding-threshold');
    const overSlider = document.getElementById('settings-overcrowding-threshold');
    const crowdLabel = document.getElementById('settings-crowding-value');
    const overLabel = document.getElementById('settings-overcrowding-value');
    if (crowdSlider) crowdSlider.value = crowding;
    if (overSlider) overSlider.value = overcrowding;
    if (crowdLabel) crowdLabel.textContent = `Above ${crowding} people`;
    if (overLabel) overLabel.textContent = `Above ${overcrowding} people`;
    renderNotificationPanel();
    renderDispatchZoneDetails(getDispatchSelectedValue());
}
function initializeSettingsPage() {
    saveAlertThresholds(window.alertThresholds.crowding, window.alertThresholds.overcrowding);
}
window.analyzedPersonCentroids = [];
window.heatmapAnalysisComplete = false;
window.isAnalyzingFeed = false;
window.lastAlertTime = 0;

// Personnel store mapped per camera
window.cctvPersonnelStore = {
    'CAM-01': [
        { name: 'Vikram Rathore', role: 'Crowd Control Lead', phone: '+91 98402 11029' }
    ],
    'CAM-02': [
        { name: 'Sarah Connor', role: 'Perimeter Security Officer', phone: '+91 97103 44210' }
    ],
    'CAM-03': [
        { name: 'Dr. Aarav Mehra', role: 'Medical First Responder', phone: '+91 98210 55901' }
    ],
    'CAM-04': [
        { name: 'Rajesh Sharma', role: 'Emergency Dispatch Commander', phone: '+91 98840 77312' }
    ]
};

window.cctvCameraRegistry = {
    'CAM-01': {
        id: 'CAM-01',
        name: 'CAM 01 - Main Entrance Plaza',
        location: 'North Gate Section A',
        type: 'System Hardware Webcam',
        density: 'Awaiting Analysis',
        densityLevel: 'Standby',
        densityColor: '#a1a1aa',
        alerts: []
    },
    'CAM-02': {
        id: 'CAM-02',
        name: 'CAM 02 - North Gate Concourse',
        location: 'Sector 3 Checkpoint',
        type: 'IP / RTSP Stream',
        density: 'Awaiting Analysis',
        densityLevel: 'Standby',
        densityColor: '#a1a1aa',
        alerts: []
    },
    'CAM-03': {
        id: 'CAM-03',
        name: 'CAM 03 - West Promenade & Stairs',
        location: 'West Plaza Walkway',
        type: 'PTZ Dome Optical',
        density: 'Awaiting Analysis',
        densityLevel: 'Standby',
        densityColor: '#a1a1aa',
        alerts: []
    },
    'CAM-04': {
        id: 'CAM-04',
        name: 'CAM 04 - Sector B Queue Corridor',
        location: 'Central Pavilion Lane 4',
        type: 'Infrared Thermal / Optical',
        density: 'Awaiting Analysis',
        densityLevel: 'Standby',
        densityColor: '#a1a1aa',
        alerts: []
    }
};

function onModelChanged(modelVal) {
    window.selectedModelId = modelVal;
    const tagEl = document.getElementById('detail-model-tag');
    if (tagEl) tagEl.innerText = modelVal;
    const badgeW = document.getElementById('cam-model-badge');
    if (badgeW) badgeW.innerText = 'MODEL: ' + modelVal;
    const badgeO = document.getElementById('optical-model-badge');
    if (badgeO) badgeO.innerText = 'MODEL: ' + modelVal;
    
    const indicator = document.getElementById('model-active-indicator');
    if (indicator) {
        indicator.style.background = '#60a5fa';
        indicator.style.boxShadow = '0 0 10px #60a5fa';
        setTimeout(() => {
            indicator.style.background = '#22c55e';
            indicator.style.boxShadow = '0 0 8px #22c55e';
        }, 400);
    }
}

// -------------------------------------------------------------
// Personnel Management (Add / Delete / Edit)
// -------------------------------------------------------------
window.isEditPersonnelMode = false;

function toggleAddPersonnelForm() {
    window.isEditPersonnelMode = !window.isEditPersonnelMode;
    const form = document.getElementById('add-personnel-form');
    if (form) {
        form.style.display = window.isEditPersonnelMode ? 'block' : 'none';
    }
    const editBtn = document.getElementById('btn-personnel-edit');
    if (editBtn) {
        editBtn.style.background = window.isEditPersonnelMode ? '#0072f5' : '#181822';
    }
    document.querySelectorAll('.personnel-delete-btn').forEach(btn => {
        btn.style.display = window.isEditPersonnelMode ? 'flex' : 'none';
    });
}

function renderPersonnelList() {
    const camId = window.currentActiveCamId || 'CAM-01';
    const listEl = document.getElementById('detail-personnel-list');
    if (!listEl) return;

    const roster = window.cctvPersonnelStore[camId] || [];
    if (roster.length === 0) {
        listEl.innerHTML = '<div style="padding: 12px; background: #16161c; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 11px;">No personnel currently allotted. Click Edit to add officers.</div>';
        return;
    }

    const delDisplay = window.isEditPersonnelMode ? 'flex' : 'none';

    listEl.innerHTML = roster.map((p, idx) => {
        const init = p.name ? p.name.trim()[0].toUpperCase() : 'O';
        const phoneStr = p.phone ? ` • Ph: ${p.phone}` : '';
        return `
        <div class="personnel-card" data-idx="${idx}" style="display: flex; align-items: center; justify-content: space-between; background: #16161c; border-radius: 8px; padding: 8px 10px; transition: border-color 0.2s ease;">
            <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0072f5, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #ffffff; flex-shrink: 0;">
                    ${init}
                </div>
                <div style="overflow: hidden;">
                    <div style="font-size: 12px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 5px;">
                        <span style="color: #ffffff;">${p.name}</span>
                        <span style="width: 5px; height: 5px; border-radius: 50%; background: #22c55e;"></span>
                    </div>
                    <div style="font-size: 10px; color: #93c5fd; font-weight: 500;">${p.role}${phoneStr}</div>
                </div>
            </div>
            <button class="personnel-delete-btn" onclick="deletePersonnel(${idx})" title="Delete Officer from Area" style="display: ${delDisplay}; background: transparent; border: none; color: #a1a1aa; cursor: pointer; padding: 4px; border-radius: 4px; align-items: center; justify-content: center;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#a1a1aa'">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        </div>
        `;
    }).join('');
}

function submitAddPersonnel() {
    const camId = window.currentActiveCamId || 'CAM-01';
    const nameInput = document.getElementById('input-personnel-name');
    const phoneInput = document.getElementById('input-personnel-phone');
    const roleSelect = document.getElementById('input-personnel-role');
    const name = nameInput ? nameInput.value.trim() : '';
    const phone = phoneInput ? phoneInput.value.trim() : '';
    const role = roleSelect ? roleSelect.value : 'Crowd Control Lead';

    if (!name) {
        alert('Please specify the officer name before saving.');
        return;
    }

    if (!window.cctvPersonnelStore[camId]) {
        window.cctvPersonnelStore[camId] = [];
    }
    window.cctvPersonnelStore[camId].push({ 
        name: name, 
        role: role,
        phone: phone || '+91 98402 11029'
    });

    if (nameInput) nameInput.value = '';
    if (phoneInput) phoneInput.value = '';
    toggleAddPersonnelForm();
    renderPersonnelList();
}

function deletePersonnel(idx) {
    const camId = window.currentActiveCamId || 'CAM-01';
    const roster = window.cctvPersonnelStore[camId] || [];
    if (idx >= 0 && idx < roster.length) {
        roster.splice(idx, 1);
        renderPersonnelList();
    }
}

function sendAlertToPersonnel(camId, sev, name) {
    const roster = window.cctvPersonnelStore[camId] || [];
    const targetOfficer = roster.length > 0 ? roster[0].name : (name || 'Assigned Officer');
    const phone = roster.length > 0 && roster[0].phone ? roster[0].phone : '+91 98402 11029';
    alert(`🚨 Alert Transmitted!\
\
Priority: ${sev}\
Target Area: ${camId}\
Notified: ${targetOfficer} (${phone})\
Channel: SMS & Dispatch Radio`);
}

// -------------------------------------------------------------
// Live Webcam & Media Stream Handling
// -------------------------------------------------------------
function requestGlobalWebcamStream(onSuccess, onError) {
    if (window.cctvGlobalStream && window.cctvGlobalStream.active) {
        if (onSuccess) onSuccess(window.cctvGlobalStream);
        return;
    }
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
            .then(function(stream) {
                window.cctvGlobalStream = stream;
                if (onSuccess) onSuccess(stream);
            })
            .catch(function(err) {
                console.log('Webcam permission error:', err);
                if (onError) onError(err);
            });
    }
}

function initCam01Webcam() {
    const video = document.getElementById('cctv-live-video');
    const placeholder = document.getElementById('cctv-webcam-placeholder');
    const hud = document.getElementById('cctv-hud-overlay');
    const statusPill = document.getElementById('cam01-live-badge');

    requestGlobalWebcamStream(function(stream) {
        if (video) {
            video.srcObject = stream;
            video.play().catch(e => console.log(e));
            video.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
            if (hud) hud.style.opacity = '1';
            if (statusPill) {
                statusPill.innerText = 'STREAMING LIVE';
                statusPill.style.color = '#22c55e';
            }
        }
    }, function(err) {
        if (placeholder) {
            placeholder.innerHTML = '<div style="color: #ef4444; font-size: 11px; text-align: center; padding: 10px;">Camera access requested.<br><span style="color: #0072f5; font-size: 11px; cursor: pointer; text-decoration: underline;" onclick="initCam01Webcam()">Click here to grant / retry camera</span></div>';
        }
    });
}

function initCamDetailWebcam() {
    const video = document.getElementById('cam-detail-video');
    const placeholder = document.getElementById('cam-detail-placeholder');

    requestGlobalWebcamStream(function(stream) {
        if (video) {
            video.srcObject = stream;
            video.play().catch(e => console.log(e));
            video.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        }
    }, function(err) {
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.innerHTML = '<div style="color: #f59e0b; font-size: 12px; text-align: center;">Camera permission denied. Please allow access and retry.</div>';
        }
        // Re-enable Start button if stream failed
        const btnStart = document.getElementById('btn-start-recording');
        const btnEnd   = document.getElementById('btn-end-recording');
        if (btnStart) { btnStart.disabled = false; btnStart.style.cursor = 'pointer'; btnStart.style.opacity = '1'; }
        if (btnEnd)   { btnEnd.disabled = true; btnEnd.style.cursor = 'not-allowed'; btnEnd.style.opacity = '0.4'; }
        const badge = document.getElementById('cam-rec-badge');
        if (badge) { badge.innerText = '● IDLE'; badge.style.color = '#71717a'; badge.style.borderColor = 'rgba(113,113,122,0.4)'; }
        window._camRecording = false;
    });
}

// ── Start / End Recording (CAM-01) ───────────────────────────────────────────
function startCameraRecording() {
    const btnStart = document.getElementById('btn-start-recording');
    const btnEnd   = document.getElementById('btn-end-recording');
    const badge    = document.getElementById('cam-rec-badge');
    const metric   = document.getElementById('cam-detail-person-metric');

    // Update UI immediately
    if (btnStart) { btnStart.disabled = true; btnStart.style.cursor = 'not-allowed'; btnStart.style.opacity = '0.45'; }
    if (btnEnd)   { btnEnd.disabled = false; btnEnd.style.cursor = 'pointer'; btnEnd.style.opacity = '1'; btnEnd.style.background = 'linear-gradient(135deg,#991b1b,#7f1d1d)'; btnEnd.style.color = '#fff'; btnEnd.style.border = 'none'; }
    if (badge)    { badge.innerText = '● REC [LIVE]'; badge.style.color = '#ef4444'; badge.style.borderColor = 'rgba(239,68,68,0.5)'; }
    if (metric)   metric.innerText = '● LIVE • ANALYZED PERSONS: 0';

    window._camRecording = true;
    window.heatmapAnalysisComplete = false;
    initCamDetailWebcam();
    // Give the stream ~400ms to start then begin analysis
    setTimeout(() => {
        if (window._camRecording) {
            startLivePersonAnalysisLoop();
            runHeatmapCompositeAnimation();
        }
    }, 600);
}

// CAM-02 can also be analyzed as a live browser camera when no clip is loaded.
function startOpticalLiveCamera() {
    const video = document.getElementById('optical-live-video');
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    const metric = document.getElementById('cam-optical-person-metric');
    requestGlobalWebcamStream((stream) => {
        if (!video || !canvas) return;
        video.srcObject = stream;
        video.play().catch(() => {});
        video.style.display = 'none';
        window._opticalLiveActive = true;
        if (metric) metric.innerText = '● LIVE CAMERA • ANALYZED PERSONS: 0';
        runOpticalCanvasAnimation();
        startLivePersonAnalysisLoop();
    }, (err) => {
        if (metric) metric.innerText = 'CAMERA ERROR • ' + (err && err.message ? err.message : 'permission denied');
    });
}

function stopCameraRecording() {
    window._camRecording = false;
    window.heatmapAnalysisComplete = false;
    // Stop analysis loop
    if (analysisInterval) { clearInterval(analysisInterval); analysisInterval = null; }
    window.analysisRunning = false;
    // Stop and release webcam stream
    const video = document.getElementById('cam-detail-video');
    if (video) {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
        video.style.display = 'none';
    }
    if (window.cctvGlobalStream) {
        window.cctvGlobalStream.getTracks().forEach(t => t.stop());
        window.cctvGlobalStream = null;
    }
    // Reset overlay canvas
    const overlay = document.getElementById('cam-detection-overlay');
    if (overlay) { const c = overlay.getContext('2d'); if (c) c.clearRect(0,0,overlay.width,overlay.height); }
    // Clear heatmap canvas
    const hc = document.getElementById('heatmap-composite-canvas');
    if (hc) { const c = hc.getContext('2d'); if (c) { c.clearRect(0,0,hc.width,hc.height); c.fillStyle='#06070a'; c.fillRect(0,0,hc.width,hc.height); }}
    const idleMsg = document.getElementById('heatmap-idle-msg');
    if (idleMsg) idleMsg.style.display = 'flex';
    // Reset zone counts
    window.analyzedPersonCentroids = [];
    window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = [];
    window.lastZoneCounts = {zone1:0,zone2:0,zone3:0,zone4:0};
    updateZonePanel(window.lastZoneCounts);
    // Show placeholder
    const placeholder = document.getElementById('cam-detail-placeholder');
    if (placeholder) placeholder.style.display = 'flex';
    // Update buttons & badge
    const btnStart = document.getElementById('btn-start-recording');
    const btnEnd   = document.getElementById('btn-end-recording');
    const badge    = document.getElementById('cam-rec-badge');
    const metric   = document.getElementById('cam-detail-person-metric');
    if (btnStart) { btnStart.disabled = false; btnStart.style.cursor = 'pointer'; btnStart.style.opacity = '1'; }
    if (btnEnd)   { btnEnd.disabled = true; btnEnd.style.cursor = 'not-allowed'; btnEnd.style.background = '#1c1c24'; btnEnd.style.color = '#71717a'; btnEnd.style.border = '1px solid #2c2c36'; }
    if (badge)    { badge.innerText = '● IDLE'; badge.style.color = '#71717a'; badge.style.borderColor = 'rgba(113,113,122,0.4)'; }
    if (metric)   metric.innerText = '● IDLE • ANALYZED PERSONS: 0';
    // Reset heatmap badge
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = '0 Persons Detected';
}

// -------------------------------------------------------------
// Real YOLO-Based Live Feed Analysis via Python Backend
// -------------------------------------------------------------
let analysisInterval = null;
window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = []; // Current detection boxes for overlay canvas
window.analysisRunning = false;

function startLiveAnalysisButtonAction() {
    if (window.analysisRunning) return;
    initCamDetailWebcam();
    if (!window.analysisRunning) {
        startLivePersonAnalysisLoop();
    }
}

function startLivePersonAnalysisLoop() {
    if (analysisInterval) clearInterval(analysisInterval);
    // Run once immediately, then sample the moving feed continuously. The
    // in-flight guard below prevents slow inference from queuing stale frames.
    analyzeActiveFeedYOLO();
    analysisInterval = setInterval(analyzeActiveFeedYOLO, 700);
    window.analysisRunning = true;
}

// Capture one JPEG frame from the webcam video and send to Python YOLO endpoint
async function captureFrameBase64(videoEl, quality) {
    return new Promise((resolve, reject) => {
        try {
            const sourceWidth = videoEl.videoWidth || videoEl.naturalWidth;
            const sourceHeight = videoEl.videoHeight || videoEl.naturalHeight;
            if (!sourceWidth || !sourceHeight) {
                resolve(null);
                return;
            }
            const scale = Math.min(1, 1280 / Math.max(sourceWidth, sourceHeight));
            const offscreen = document.createElement('canvas');
            offscreen.width = Math.max(1, Math.round(sourceWidth * scale));
            offscreen.height = Math.max(1, Math.round(sourceHeight * scale));
            const ctx = offscreen.getContext('2d');
            if (!ctx) throw new Error('Could not create a frame capture canvas');
            ctx.drawImage(videoEl, 0, 0, offscreen.width, offscreen.height);
            offscreen.toBlob((blob) => {
                if (!blob) {
                    reject(new Error('Could not encode the captured frame'));
                    return;
                }
                const reader = new FileReader();
                reader.onloadend = () => {
                    if (typeof reader.result !== 'string' || !reader.result.includes(',')) {
                        reject(new Error('Could not read the captured frame'));
                        return;
                    }
                    const b64 = reader.result.split(',', 2)[1];
                    resolve(b64);
                };
                reader.onerror = () => reject(new Error('Could not read the captured frame'));
                reader.readAsDataURL(blob);
            }, 'image/jpeg', quality || 0.8);
        } catch(e) {
            reject(e);
        }
    });
}

// Call the Gradio predict_frame_base64 function endpoint
async function callYOLOEndpoint(frameB64, modelId) {
    try {
        const payload = { frame_b64: frameB64, model_id: modelId };
        const resp = await fetch('/yolo_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await resp.json();
        if (!resp.ok) {
            throw new Error(result.error || `Detection service returned HTTP ${resp.status}`);
        }
        if (typeof result.count !== 'number') {
            throw new Error(result.error || 'Detection service returned an invalid response');
        }
        return result;
    } catch(e) {
        console.warn('[DensiTrack] YOLO endpoint call failed:', e.message);
        throw e;
    }
}

function showAnalysisError(error) {
    const message = error && error.message ? error.message : String(error);
    console.error('[DensiTrack] Person detection failed:', message);
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = 'Analysis unavailable';
    window.heatmapAnalysisComplete = false;
    const warningEl = document.getElementById('detail-chokepoint-val');
    if (warningEl) {
        warningEl.innerText = `ANALYSIS ERROR: ${message}`;
        warningEl.style.color = '#ef4444';
    }
    const camId = window.currentActiveCamId || 'CAM-01';
    const metricEl = document.getElementById(camId === 'CAM-01' ? 'cam-detail-person-metric' : 'cam-optical-person-metric');
    if (metricEl) metricEl.innerText = '● ANALYSIS ERROR • CHECK SERVER LOG';
}

// ─── Zone grid helpers ───────────────────────────────────────────────────────
const ZONE_OVERCROWD_THRESHOLD = 8; // only dense zones should be critical

function _zoneColorForCount(n) {
    if (n >= ZONE_OVERCROWD_THRESHOLD) return { fill: 'rgba(239,68,68,0.14)', stroke: '#ef4444', label: '#fca5a5' };
    if (n >= 4) return { fill: 'rgba(249,115,22,0.12)', stroke: '#f97316', label: '#fdba74' };
    if (n >= 2) return { fill: 'rgba(234,179,8,0.10)', stroke: '#eab308', label: '#fde047' };
    if (n === 1) return { fill: 'rgba(59,130,246,0.10)', stroke: '#3b82f6', label: '#93c5fd' };
    return { fill: 'rgba(34,197,94,0.05)', stroke: 'rgba(161,161,170,0.55)', label: '#d4d4d8' };
}

// Draw the 2x2 zone grid + heatmap blobs on any canvas that covers a video
function drawZoneGridOnCanvas(ctx, w, h, zoneCounts, centroids) {
    const midX = w / 2;
    const midY = h / 2;
    const zones = [
        { key:'zone1', label:'Zone 1', x:0,    y:0,    bw:midX, bh:midY },
        { key:'zone2', label:'Zone 2', x:midX, y:0,    bw:midX, bh:midY },
        { key:'zone3', label:'Zone 3', x:0,    y:midY, bw:midX, bh:midY },
        { key:'zone4', label:'Zone 4', x:midX, y:midY, bw:midX, bh:midY },
    ];

    ctx.save();

    // ── Heatmap Gaussian blobs (drawn first so grid sits on top) ──
    if (centroids && centroids.length > 0) {
        // centroids use heatmap canvas coords (480×190); rescale to current canvas
        const scaleX = w / 480;
        const scaleY = h / 190;
        centroids.forEach(p => {
            const px = p.x * scaleX;
            const py = p.y * scaleY;
            const r  = (p.radius || 40) * Math.min(scaleX, scaleY) * 1.4;
            const localCount = centroids.filter(other => Math.hypot(other.x - p.x, other.y - p.y) < 80).length;
            const hot = localCount >= 8 ? [239,68,68]
                : localCount >= 4 ? [249,115,22]
                : localCount >= 2 ? [234,179,8]
                : [59,130,246];
            const grad = ctx.createRadialGradient(px, py, 0, px, py, r);
            grad.addColorStop(0,    `rgba(${hot[0]},${hot[1]},${hot[2]},0.94)`);
            grad.addColorStop(0.35, `rgba(${hot[0]},${hot[1]},${hot[2]},0.62)`);
            grad.addColorStop(0.65, `rgba(${hot[0]},${hot[1]},${hot[2]},0.20)`);
            grad.addColorStop(0.85, `rgba(${hot[0]},${hot[1]},${hot[2]},0.10)`);
            grad.addColorStop(1,    'rgba(0,0,0,0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(px, py, r, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // ── Zone fills & borders ──
    zones.forEach(z => {
        const count = (zoneCounts && zoneCounts[z.key]) || 0;
        const col = _zoneColorForCount(count);
        ctx.fillStyle = col.fill;
        ctx.fillRect(z.x, z.y, z.bw, z.bh);
        ctx.strokeStyle = col.stroke;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(z.x + 1, z.y + 1, z.bw - 2, z.bh - 2);
        ctx.setLineDash([]);
    });

    // ── Cross-hair dividing lines ──
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 4]);
    ctx.beginPath(); ctx.moveTo(midX, 0); ctx.lineTo(midX, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(w, midY); ctx.stroke();
    ctx.setLineDash([]);

    // ── Zone labels + count badges ──
    zones.forEach(z => {
        const count = (zoneCounts && zoneCounts[z.key]) || 0;
        const col = _zoneColorForCount(count);
        const lx = z.x + 10;
        const ly = z.y + 22;

        // Zone name pill
        ctx.font = 'bold 12px monospace';
        const zoneText = `${z.label}: ${count}`;
        const labelW = ctx.measureText(zoneText).width + 10;
        ctx.fillStyle = 'rgba(0,0,0,0.72)';
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(lx - 2, ly - 14, labelW, 18, 4)
                      : ctx.rect(lx - 2, ly - 14, labelW, 18);
        ctx.fill();
        ctx.fillStyle = col.label;
        ctx.fillText(zoneText, lx + 3, ly);
    });

    ctx.restore();
}

// Draw bounding box overlay + zone grid on the canvas stacked over the live video
function drawDetectionOverlay(boxes, zoneCounts, centroids) {
    const canvas = document.getElementById('cam-detection-overlay');
    if (!canvas) return;
    const video = document.getElementById('cam-detail-video');
    if (!video || !video.videoWidth) return;

    canvas.width = video.clientWidth || canvas.offsetWidth || 640;
    canvas.height = video.clientHeight || canvas.offsetHeight || 360;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw person bounding boxes first
    boxes.forEach((b, i) => {
        const x1 = b.x1 * canvas.width;
        const y1 = b.y1 * canvas.height;
        const bw = (b.x2 - b.x1) * canvas.width;
        const bh = (b.y2 - b.y1) * canvas.height;

        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1, y1, bw, bh);

        // Corner accents
        const cs = 8;
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(x1, y1+cs); ctx.lineTo(x1, y1); ctx.lineTo(x1+cs, y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1+bw-cs, y1); ctx.lineTo(x1+bw, y1); ctx.lineTo(x1+bw, y1+cs); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1+bh-cs); ctx.lineTo(x1, y1+bh); ctx.lineTo(x1+cs, y1+bh); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1+bw-cs, y1+bh); ctx.lineTo(x1+bw, y1+bh); ctx.lineTo(x1+bw, y1+bh-cs); ctx.stroke();

        // Label
        const label = `P${i+1} ${Math.round(b.conf*100)}%`;
        ctx.font = 'bold 10px monospace';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.fillRect(x1, y1 - 16, tw + 6, 14);
        ctx.fillStyle = '#4ade80';
        ctx.fillText(label, x1 + 3, y1 - 4);
    });

    // Overlay zone grid + heatmap on top
    drawZoneGridOnCanvas(ctx, canvas.width, canvas.height, zoneCounts || window.lastZoneCounts, centroids || window.analyzedPersonCentroids);
    drawFaceMarkers(ctx, window.detectedFaceBoxes || [], canvas.width, canvas.height);
}

function drawFaceMarkers(ctx, faces, width, height) {
    (faces || []).forEach((face, index) => {
        const x = ((Number(face.x1) + Number(face.x2)) / 2) * width;
        const y = ((Number(face.y1) + Number(face.y2)) / 2) * height;
        const radius = Math.max(5, Math.min(10, (Number(face.x2) - Number(face.x1)) * width * 0.45));
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(34,211,238,0.8)';
        ctx.fill();
        ctx.strokeStyle = '#ecfeff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.font = 'bold 8px monospace';
        ctx.fillStyle = '#cffafe';
        ctx.fillText(`F${index + 1}`, x + radius + 2, y + 3);
        ctx.restore();
    });
}

function renderUploadedMediaToCanvas() {
    const state = window.uploadedMediaState;
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!state && window._opticalLiveActive) {
        const liveVideo = document.getElementById('optical-live-video');
        if (liveVideo && liveVideo.readyState >= 2 && liveVideo.videoWidth) {
            ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);
        }
        drawUploadedDetectionOverlay(window.yoloDetectionBoxes || [], window.lastZoneCounts, window.analyzedPersonCentroids);
        return;
    }
    if (!state) {
        ctx.fillStyle = 'rgba(96, 165, 250, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // Still draw zone grid even on empty canvas
        drawZoneGridOnCanvas(ctx, canvas.width, canvas.height, window.lastZoneCounts || {zone1:0,zone2:0,zone3:0,zone4:0}, []);
        return;
    }

    if (state.type === 'image' && state.imgEl) {
        ctx.drawImage(state.imgEl, 0, 0, canvas.width, canvas.height);
    } else if (state.type === 'video' && state.videoEl) {
        ctx.drawImage(state.videoEl, 0, 0, canvas.width, canvas.height);
    }
    // Always overlay detection boxes + zone grid after drawing the frame
    drawUploadedDetectionOverlay(window.yoloDetectionBoxes || [], window.lastZoneCounts, window.analyzedPersonCentroids);
}

function heatColor(count) {
    return Number(count) > 50 ? '#ef4444' : Number(count) > 30 ? '#f97316' : Number(count) > 1 ? '#facc15' : '#3b82f6';
}
function drawDashboardCameraTile(camId) {
    const canvas = document.getElementById(`dashboard-camera-canvas-${camId}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const state = window.uploadedMediaByCam[camId];
    let source = state && (state.videoEl || state.imgEl);
    if (!source && camId === 'CAM-01') source = document.getElementById('cctv-live-video');
    if (!source && camId === 'CAM-02' && window._opticalLiveActive) source = document.getElementById('optical-live-video');
    const ready = source && (source.tagName === 'VIDEO' ? source.readyState >= 2 && source.videoWidth : source.complete && source.naturalWidth);
    ctx.fillStyle = '#050508'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    const placeholder = document.getElementById(`dashboard-camera-placeholder-${camId}`);
    if (ready) {
        try {
            const sw = source.videoWidth || source.naturalWidth, sh = source.videoHeight || source.naturalHeight;
            const scale = Math.max(canvas.width / sw, canvas.height / sh);
            const dw = sw * scale, dh = sh * scale;
            ctx.drawImage(source, (canvas.width-dw)/2, (canvas.height-dh)/2, dw, dh);
            if (placeholder) placeholder.style.display = 'none';
        } catch (_) { if (placeholder) placeholder.style.display = 'flex'; }
    } else if (placeholder) placeholder.style.display = 'flex';

    const data = window.cameraDashboardData[camId] || {};
    const boxes = data.boxes || [];
    const count = Number(data.count) || 0;
    boxes.forEach((box, i) => {
        const x = Number(box.x1 || 0) * canvas.width, y = Number(box.y1 || 0) * canvas.height;
        const w = (Number(box.x2 || 0) - Number(box.x1 || 0)) * canvas.width;
        const h = (Number(box.y2 || 0) - Number(box.y1 || 0)) * canvas.height;
        const dot = heatColor(Math.max(count, boxes.length));
        ctx.save(); ctx.shadowColor = dot; ctx.shadowBlur = 18;
        ctx.beginPath(); ctx.arc(x + w/2, y + h/2, 9, 0, Math.PI*2); ctx.fillStyle = dot; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
        ctx.strokeStyle = dot; ctx.lineWidth = 2; ctx.strokeRect(x,y,w,h);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.fillText(`P${i+1}`, x, Math.max(13,y-4));
    });
    const countEl = document.getElementById(`dashboard-camera-count-${camId}`);
    if (countEl) { countEl.textContent = `${count} PEOPLE`; countEl.style.color = heatColor(count); }
}
function renderAllDashboardCameraTiles() {
    ['CAM-01','CAM-02','CAM-03','CAM-04'].forEach(drawDashboardCameraTile);
}
function startDashboardPreviewLoop() {
    if (window.dashboardPreviewFrame) return;
    const tick = () => { renderAllDashboardCameraTiles(); window.dashboardPreviewFrame = requestAnimationFrame(tick); };
    window.dashboardPreviewFrame = requestAnimationFrame(tick);
}
async function analyzeDashboardFeeds() {
    if (window.dashboardAnalysisBusy || document.hidden) return;
    const cams = ['CAM-01','CAM-02','CAM-03','CAM-04'];
    const camId = cams[window.dashboardAnalysisIndex = ((window.dashboardAnalysisIndex || 0) + 1) % cams.length];
    const state = window.uploadedMediaByCam[camId];
    let source = state ? (state.videoEl || state.imgEl) : null;
    if (!source && camId === 'CAM-01') source = document.getElementById('cctv-live-video');
    if (!source && camId === 'CAM-02' && window._opticalLiveActive) source = document.getElementById('optical-live-video');
    if (!source) return;
    const ready = source.tagName === 'VIDEO' ? source.readyState >= 2 && source.videoWidth : source.complete && source.naturalWidth;
    if (!ready) return;
    window.dashboardAnalysisBusy = true;
    try {
        const frame = await captureFrameBase64(source, 0.7);
        if (!frame) return;
        const result = await callYOLOEndpoint(frame, window.selectedModelId || 'yolo11n.pt');
        if (!result || typeof result.count !== 'number') return;
        window.cameraDashboardData[camId] = {count:result.count, boxes:result.boxes||[], centroids:result.centroids||[], zoneCounts:result.zone_counts||{}, updatedAt:Date.now()};
        renderNotificationPanel(); renderAllDashboardCameraTiles();
    } catch (error) { console.warn('Dashboard camera analysis failed', camId, error); }
    finally { window.dashboardAnalysisBusy = false; }
}
setInterval(analyzeDashboardFeeds, 900);

function drawUploadedDetectionOverlay(boxes, zoneCounts, centroids) {
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    if (boxes && boxes.length > 0) {
        boxes.forEach((b, i) => {
            const x1 = Number(b.x1 || 0) * canvas.width;
            const y1 = Number(b.y1 || 0) * canvas.height;
            const bw = (Number(b.x2 || 0) - Number(b.x1 || 0)) * canvas.width;
            const bh = (Number(b.y2 || 0) - Number(b.y1 || 0)) * canvas.height;

            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x1, y1, bw, bh);

            const label = `P${i+1} ${Math.round((b.conf || 0) * 100)}%`;
            ctx.font = 'bold 10px monospace';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(x1, Math.max(0, y1 - 16), tw + 6, 14);
            ctx.fillStyle = '#4ade80';
            ctx.fillText(label, x1 + 3, Math.max(12, y1 - 4));

            // A bright, numbered dot marks the center of every detected person.
            const cx = x1 + bw / 2;
            const cy = y1 + bh / 2;
            const localDensity = boxes.filter(other => {
                const ox = (Number(other.x1) + Number(other.x2)) * canvas.width / 2;
                const oy = (Number(other.y1) + Number(other.y2)) * canvas.height / 2;
                return Math.hypot(ox - cx, oy - cy) < Math.min(canvas.width, canvas.height) * 0.2;
            }).length;
            const dotColor = localDensity >= 4 ? '#ef4444' : localDensity >= 2 ? '#fb923c' : '#fde047';
            ctx.beginPath();
            ctx.arc(cx, cy, 7, 0, Math.PI * 2);
            ctx.fillStyle = dotColor;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }
    // Zone grid + heatmap overlay
    drawZoneGridOnCanvas(ctx, canvas.width, canvas.height, zoneCounts || window.lastZoneCounts, centroids || window.analyzedPersonCentroids);
    drawFaceMarkers(ctx, window.detectedFaceBoxes || [], canvas.width, canvas.height);
    ctx.restore();
}

function updateHeatmapAndAlertUI(personCount, sourceLabel) {
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = `${personCount} People • ${(window.detectedFaceBoxes || []).length} Faces Detected`;

    const densVal = document.getElementById('detail-density-val');
    const chokepointVal = document.getElementById('detail-chokepoint-val');

    let densityLevelStr = 'Optimal Flow';
    let densityPctStr = `${Math.min(96, Math.max(4, personCount * 16))}%`;
    let color = '#22c55e';

    if (isOvercrowdingCount(personCount)) {
        densityLevelStr = 'Critical Surge';
        color = '#ef4444';
    } else if (isCrowdingCount(personCount)) {
        densityLevelStr = 'High Density';
        color = '#f97316';
    } else if (personCount >= 2) {
        densityLevelStr = 'Elevated Flow';
        color = '#f59e0b';
    }

    if (densVal) densVal.innerText = `${densityLevelStr.split(' ')[0]} (${densityPctStr})`;
    if (chokepointVal) {
        chokepointVal.innerText = densityLevelStr;
        chokepointVal.style.color = color;
    }

    const camId = window.currentActiveCamId || 'CAM-02';
    const alertStatus = isOvercrowdingCount(personCount) ? 'Critical' : (isCrowdingCount(personCount) ? 'Warning' : (personCount > 0 ? 'Info' : 'Normal'));
    const alertGapMs = personCount > 0 ? 5000 : 20000;
    if (personCount > 0 && (Date.now() - window.lastAlertTime > alertGapMs)) {
        window.lastAlertTime = Date.now();
        pushDynamicIncidentAlert(camId, personCount, densityLevelStr, alertStatus, sourceLabel);
    }

    if (sourceLabel) {
        const metricEl = document.getElementById(camId === 'CAM-01' ? 'cam-detail-person-metric' : 'cam-optical-person-metric');
        if (metricEl) metricEl.innerText = `● ${sourceLabel} • ANALYZED PERSONS: ${personCount} (${window.selectedModelId})`;
    }
}

function triggerMediaUpload() {
    const input = document.getElementById('media-upload-input');
    if (input) input.click();
}

function syncMediaActionButtons() {
    const badge = document.getElementById('detail-cam-id-badge');
    const camFromPage = badge ? badge.innerText.trim() : '';
    const camId = window.currentActiveCamId || camFromPage || 'CAM-02';
    const analyzeBtn = document.getElementById('media-analyze-btn');
    const uploadBtn = document.getElementById('media-upload-btn');
    const bar = document.getElementById('media-upload-bar');
    if (!analyzeBtn || !uploadBtn || !bar) return;

    const isUploadCam = true;
    bar.style.display = isUploadCam ? 'flex' : 'none';
    uploadBtn.style.display = 'inline-flex';
    uploadBtn.innerText = 'Upload Video / Image';

    const showAnalyze = isUploadCam;
    analyzeBtn.style.display = showAnalyze ? 'inline-flex' : 'none';
    analyzeBtn.innerText = 'Analyze Feed';
}

function setMediaUploadMode(camId) {
    const bar = document.getElementById('media-upload-bar');
    const btn = document.getElementById('media-upload-btn');
    const input = document.getElementById('media-upload-input');
    const analyzeBtn = document.getElementById('media-analyze-btn');
    if (!bar || !btn || !input) return;

    {
        bar.style.display = 'flex';
        btn.style.display = 'inline-flex';
        btn.innerText = 'Upload Video / Image';
        input.accept = 'video/*,image/*';
        input.multiple = false;
        if (analyzeBtn) analyzeBtn.innerText = 'Analyze Feed';
    }
    syncMediaActionButtons();
}

async function handleUploadedMediaSelection(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    const camId = window.currentActiveCamId || 'CAM-02';

    // Stop any running analysis first
    if (window._uploadAnalysisTimer) { clearInterval(window._uploadAnalysisTimer); window._uploadAnalysisTimer = null; }

    if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = 'auto';
        video.onloadeddata = () => {
            const state = { camId, file, type: 'video', videoEl: video, imgEl: null, frameB64: null };
            window.uploadedMediaByCam[camId] = state;
            window.uploadedMediaState = state;
            window.analyzedPersonCentroids = [];
            window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = [];
            window.lastZoneCounts = {zone1:0,zone2:0,zone3:0,zone4:0};
            window.heatmapAnalysisComplete = false;
            video.play().catch(() => {});
            const webcamBox = document.getElementById('cam-detail-webcam-box');
            const opticalBox = document.getElementById('cam-detail-optical-box');
            if (webcamBox) webcamBox.style.display = 'none';
            if (opticalBox) opticalBox.style.display = 'flex';
            // Render first frame on canvas (no analysis yet)
            renderUploadedMediaToCanvas();
            // Show Remove & Analyse buttons
            const removeBtn  = document.getElementById('media-remove-btn');
            const analyzeBtn = document.getElementById('media-analyze-btn');
            if (removeBtn)  removeBtn.style.display  = 'inline-flex';
            if (analyzeBtn) analyzeBtn.style.display = 'inline-flex';
            // Start heatmap canvas loop (shows video frame, idle until analysis runs)
            runHeatmapCompositeAnimation();
            // Start detection as soon as the uploaded clip is ready.
            startUploadedMediaAnalysisLoop();
            renderAllDashboardCameraTiles();
        };
        video.src = URL.createObjectURL(file);
    } else if (file.type.startsWith('image/')) {
        const image = new Image();
        image.onload = function() {
            const reader = new FileReader();
            reader.onload = function() {
                const state = { camId, file, type: 'image', imgEl: image, videoEl: null, frameB64: String(reader.result).split(',')[1] };
                window.uploadedMediaByCam[camId] = state;
                window.uploadedMediaState = state;
                window.analyzedPersonCentroids = [];
                window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = [];
                window.lastZoneCounts = {zone1:0,zone2:0,zone3:0,zone4:0};
                window.heatmapAnalysisComplete = false;
                renderUploadedMediaToCanvas();
                const removeBtn  = document.getElementById('media-remove-btn');
                const analyzeBtn = document.getElementById('media-analyze-btn');
                if (removeBtn)  removeBtn.style.display  = 'inline-flex';
                if (analyzeBtn) { analyzeBtn.style.display = 'inline-flex'; analyzeBtn.innerText = '▶ Analyse Image'; }
                runHeatmapCompositeAnimation();
                startUploadedMediaAnalysisLoop();
                renderAllDashboardCameraTiles();
            };
            reader.readAsDataURL(file);
        };
        image.src = URL.createObjectURL(file);
    } else { alert('Choose a video or image file.'); }

    event.target.value = '';
}

// Remove Video button handler
function removeUploadedMedia() {
    // Stop analysis
    if (window._uploadAnalysisTimer) { clearInterval(window._uploadAnalysisTimer); window._uploadAnalysisTimer = null; }
    // Pause & release video element
    const state = window.uploadedMediaState;
    if (state && state.videoEl) {
        state.videoEl.pause();
        state.videoEl.src = '';
    }
    if (state) delete window.uploadedMediaByCam[state.camId];
    window.uploadedMediaState = null;
    window.analyzedPersonCentroids = [];
    window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = [];
    window.lastZoneCounts = {zone1:0,zone2:0,zone3:0,zone4:0};
    updateZonePanel(window.lastZoneCounts);
    // Clear main canvas
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    if (canvas) { const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#000'; ctx.fillRect(0,0,canvas.width,canvas.height); }
    // Hide Remove & Analyse, reset metric
    const removeBtn  = document.getElementById('media-remove-btn');
    const analyzeBtn = document.getElementById('media-analyze-btn');
    if (removeBtn)  removeBtn.style.display  = 'none';
    if (analyzeBtn) analyzeBtn.style.display = 'none';
    const metricEl = document.getElementById('cam-optical-person-metric');
    if (metricEl) metricEl.innerText = '● WAITING FOR UPLOAD • ANALYZED PERSONS: 0';
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = '0 Persons Detected';
    const densVal = document.getElementById('detail-density-val');
    if (densVal) densVal.innerText = 'Awaiting';
    const chokepointVal = document.getElementById('detail-chokepoint-val');
    if (chokepointVal) { chokepointVal.innerText = 'Standby'; chokepointVal.style.color = '#a1a1aa'; }
    // Heatmap canvas shows idle again
    runHeatmapCompositeAnimation();
    renderAllDashboardCameraTiles();
}

// ── Continuous uploaded-media analysis loop ─────────────────────────────────
// For VIDEO: runs YOLO every ~800ms while the video is playing.
// For IMAGE: runs YOLO once (re-runs if Analyze button is pressed).
window._uploadAnalysisTimer = null;
window._uploadAnalysisInFlight = false;

async function _doOneUploadedFrame() {
    const camId = window.currentActiveCamId || 'CAM-02';
    const state = window.uploadedMediaState;
    if (!state || state.camId !== camId || window._uploadAnalysisInFlight) return;
    window._uploadAnalysisInFlight = true;

    try {
        let frameB64 = null;
        if (state.type === 'image') {
            frameB64 = state.frameB64;
        } else if (state.type === 'video' && state.videoEl) {
            const video = state.videoEl;
            // Wait until video is ready
            if (video.readyState < 2 || !video.videoWidth) return;
            frameB64 = await captureFrameBase64(video, 0.72);
        }
        if (!frameB64) return;

        const result = await callYOLOEndpoint(frameB64, window.selectedModelId || 'yolo11n.pt');
        // A request can finish after navigation or after the user replaces the clip.
        if (window.uploadedMediaState !== state || window.currentActiveCamId !== camId) return;
        if (!result || typeof result.count !== 'number') return;

        window.analyzedPersonCentroids = result.centroids || [];
        window.lastZoneCounts = result.zone_counts || {zone1:0,zone2:0,zone3:0,zone4:0};
        window.yoloDetectionBoxes = result.boxes || [];
        window.detectedFaceBoxes = result.faces || [];
        window.heatmapAnalysisComplete = true;
        window.activeModelId = result.model || window.selectedModelId || 'yolo11n.pt';
        const modelBadge = document.getElementById('optical-model-badge');
        if (modelBadge) modelBadge.innerText = `MODEL: ${window.activeModelId}`;

        const personCount = result.count;
        window.cameraDashboardData[camId] = { count: personCount, boxes: result.boxes || [], centroids: result.centroids || [], zoneCounts: result.zone_counts || {}, updatedAt: Date.now() };
        renderNotificationPanel();
        const sourceLabel = state.type === 'video' ? 'VIDEO ANALYSIS' : 'IMAGE ANALYSIS';
        const warning = result.warning || 'NORMAL: safe human density';
        const warningLevel = result.warning_level || 'ok';

        // Update chokepoint warning text
        const warningEl = document.getElementById('detail-chokepoint-val');
        if (warningEl) {
            warningEl.innerText = warning;
            warningEl.style.color = warningLevel === 'danger' ? '#ef4444' : (warningLevel === 'warning' ? '#f59e0b' : '#22c55e');
        }

        // Refresh canvas (draws frame + boxes + zone grid in one call)
        renderUploadedMediaToCanvas();

        // Refresh zone panel cards
        updateZonePanel(window.lastZoneCounts);

        // Update heatmap badge, density text, and generate alerts
        updateHeatmapAndAlertUI(personCount, sourceLabel);

        // Update person metric label
        const metricEl = document.getElementById('cam-optical-person-metric');
        if (metricEl) metricEl.innerText = `● ${sourceLabel} • ANALYZED PERSONS: ${personCount} (${window.selectedModelId || 'yolo11n.pt'})`;

    } catch (e) {
        showAnalysisError(e);
    } finally {
        window._uploadAnalysisInFlight = false;
    }
}

function startUploadedMediaAnalysisLoop() {
    // Clear any existing timer
    if (window._uploadAnalysisTimer) {
        clearInterval(window._uploadAnalysisTimer);
        window._uploadAnalysisTimer = null;
    }
    const state = window.uploadedMediaState;
    if (!state) return;

    // Run immediately
    _doOneUploadedFrame();

    if (state.type === 'video') {
        // For video: keep analyzing frames continuously
        window._uploadAnalysisTimer = setInterval(() => {
            const s = window.uploadedMediaState;
            if (!s || s !== state || s.type !== 'video' || s.videoEl.ended) {
                clearInterval(window._uploadAnalysisTimer);
                window._uploadAnalysisTimer = null;
                return;
            }
            _doOneUploadedFrame();
        }, 800);
    }
    // For images: single run is enough (button re-triggers if needed)
}

// Legacy name kept for compatibility (called from handleUploadedMediaSelection)
async function analyzeUploadedMediaCycle() {
    startUploadedMediaAnalysisLoop();
}

async function analyzeActiveFeedYOLO() {
    if (window._liveAnalysisInFlight) return;
    window._liveAnalysisInFlight = true;
    try {
        await _analyzeActiveFeedYOLOFrame();
    } finally {
        window._liveAnalysisInFlight = false;
    }
}

async function _analyzeActiveFeedYOLOFrame() {
    const camId = window.currentActiveCamId || 'CAM-01';

    if (window.uploadedMediaState && window.uploadedMediaState.camId === camId) {
        await analyzeUploadedMediaCycle();
        return;
    }

    let personCount = 0;
    let centroids = [];

    if (camId === 'CAM-01' || window._opticalLiveActive) {
        const video = camId === 'CAM-01' ? document.getElementById('cam-detail-video') : document.getElementById('optical-live-video');
        if (!video || video.readyState < 2 || !video.videoWidth) return;
        try {
            const frameB64 = await captureFrameBase64(video, 0.75);
            if (!frameB64) throw new Error('Could not capture a frame from the active camera');
            const result = await callYOLOEndpoint(frameB64, window.selectedModelId || 'yolo11n.pt');
            if ((camId === 'CAM-01' && !window._camRecording) || (camId !== 'CAM-01' && !window._opticalLiveActive) || window.currentActiveCamId !== camId) return;
            personCount = result.count;
            window.cameraDashboardData[camId] = { count: personCount, updatedAt: Date.now() };
            renderNotificationPanel();
            centroids = result.centroids || [];
            window.yoloDetectionBoxes = result.boxes || [];
        window.detectedFaceBoxes = result.faces || [];
            window.heatmapAnalysisComplete = true;
            window.activeModelId = result.model || window.selectedModelId || 'yolo11n.pt';
            const modelBadge = document.getElementById(camId === 'CAM-01' ? 'cam-model-badge' : 'optical-model-badge');
            if (modelBadge) modelBadge.innerText = `MODEL: ${window.activeModelId}`;
            window.lastZoneCounts = result.zone_counts || {zone1:0,zone2:0,zone3:0,zone4:0};
            if (camId === 'CAM-01') drawDetectionOverlay(window.yoloDetectionBoxes, window.lastZoneCounts, centroids);
            else renderUploadedMediaToCanvas();
            updateZonePanel(window.lastZoneCounts);
            const warning = result.warning || 'NORMAL: safe human density';
            const warningLevel = result.warning_level || 'ok';
            const warningEl = document.getElementById('detail-chokepoint-val');
            if (warningEl) {
                warningEl.innerText = warning;
                warningEl.style.color = warningLevel === 'danger' ? '#ef4444' : (warningLevel === 'warning' ? '#f59e0b' : '#22c55e');
            }
        } catch (e) {
            showAnalysisError(e);
            return;
        }
        const metricEl = document.getElementById('cam-detail-person-metric');
        if (metricEl) metricEl.innerText = `● 1080p HD • ANALYZED PERSONS: ${personCount} (${window.selectedModelId})`;
    } else {
        const metricEl = document.getElementById('cam-optical-person-metric');
        const state = window.uploadedMediaState;
        if (state && state.camId === camId) {
            if (state.type === 'image' || state.type === 'video') {
                await analyzeUploadedMediaCycle();
                return;
            }
        }
        if (metricEl) metricEl.innerText = `● WAITING FOR LIVE ANALYSIS • ANALYZED PERSONS: 0 (${window.selectedModelId})`;
    }

    window.analyzedPersonCentroids = centroids;
    if (camId !== 'CAM-01' && window._opticalLiveActive) renderUploadedMediaToCanvas();
    runHeatmapCompositeAnimation();

    // Update Crowd Density Stats & Badges
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = `${personCount} People • ${(window.detectedFaceBoxes || []).length} Faces Detected`;

    const densVal = document.getElementById('detail-density-val');
    const chokepointVal = document.getElementById('detail-chokepoint-val');

    let densityLevelStr = 'Optimal Flow';
    let densityPctStr = `${Math.min(96, Math.max(4, personCount * 16))}%`;
    let color = '#22c55e';

    if (isOvercrowdingCount(personCount)) {
        densityLevelStr = 'Critical Surge';
        color = '#ef4444';
    } else if (isCrowdingCount(personCount)) {
        densityLevelStr = 'High Density';
        color = '#f97316';
    } else if (personCount >= 2) {
        densityLevelStr = 'Elevated Flow';
        color = '#f59e0b';
    }

    if (densVal) densVal.innerText = `${densityLevelStr.split(' ')[0]} (${densityPctStr})`;
    if (chokepointVal) {
        chokepointVal.innerText = densityLevelStr;
        chokepointVal.style.color = color;
    }

    // Trigger incident alert if threshold is breached (debounced to 15s)
    if (isCrowdingCount(personCount) && (Date.now() - window.lastAlertTime > 15000)) {
        window.lastAlertTime = Date.now();
        pushDynamicIncidentAlert(camId, personCount, densityLevelStr, isOvercrowdingCount(personCount) ? 'Critical' : 'Warning', 'LIVE ANALYSIS');
    }
    if (personCount === 0 && camId && document.getElementById('detail-alerts-list')) {
        const pending = document.getElementById('detail-alerts-list').innerText.toLowerCase();
        if (!pending.includes('waiting for live detection') && !pending.includes('person')) {
            document.getElementById('detail-alerts-list').innerHTML = '<div style="padding: 14px; background: #16161c; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 12px;">Waiting for live detection data from the active feed.</div>';
        }
    }
}

function pushDynamicIncidentAlert(camId, count, level, severity = 'Critical', sourceLabel = 'LIVE ANALYSIS') {
    const alertsList = document.getElementById('detail-alerts-list');
    if (!alertsList) return;

    const d = new Date();
    const timeStr = d.toTimeString().split(' ')[0];
    const roster = window.cctvPersonnelStore[camId] || [];
    const officer = roster.length > 0 ? roster[0].name : 'Duty Officer';
    const severityColor = severity === 'Critical' ? '#f87171' : (severity === 'Warning' ? '#fbbf24' : '#4ade80');
    const severityBg = severity === 'Critical' ? 'rgba(239, 68, 68, 0.2)' : (severity === 'Warning' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(34, 197, 94, 0.2)');
    const severityBorder = severity === 'Critical' ? '#ef4444' : (severity === 'Warning' ? '#f59e0b' : '#22c55e');
    const actionText = severity === 'Critical' ? 'Critical' : (severity === 'Warning' ? 'Warning' : 'Info');

    const newAlertHtml = `
    <div style="background: #16161c; border-radius: 8px; padding: 10px; margin-bottom: 8px; animation: fadeIn 0.3s ease;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 10px; font-weight: 700; color: #a1a1aa; text-transform: uppercase;">Type:</span>
                <span style="background: ${severityBg}; color: ${severityColor}; border: 1px solid ${severityBorder}; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;">${actionText}</span>
            </div>
            <span style="font-size: 10px; font-family: monospace; color: #a1a1aa;">${timeStr}</span>
        </div>
        <div style="font-size: 11px; color: #ffffff; line-height: 1.4; margin-bottom: 6px;">
            <b>Details:</b> ${sourceLabel} reported <b>${count} persons</b> in the monitored area. Current status: <b>${level}</b>.
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f1f26; padding-top: 6px;">
            <span style="font-size: 10px; color: #93c5fd;">Assigned: ${officer}</span>
            <button onclick="sendAlertToPersonnel('${camId}', '${actionText}', '${officer}')" style="background: ${severity === 'Critical' ? '#ef4444' : (severity === 'Warning' ? '#f59e0b' : '#22c55e')}; color: #ffffff; border: none; padding: 4px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer;">
                Send Alert
            </button>
        </div>
    </div>
    `;

    alertsList.insertAdjacentHTML('afterbegin', newAlertHtml);
}

// -------------------------------------------------------------
// Navigation & Page Switching
// -------------------------------------------------------------
function selectCameraPage(camId) {
    window.currentActiveCamId = camId;
    const homeEl = document.getElementById('home-page-container');
    const detailEl = document.getElementById('cam-detail-page-container');
    const dispatchEl = document.getElementById('dispatch-page-container');
    const settingsEl = document.getElementById('settings-page-container');
    if (homeEl) homeEl.classList.add('hidden');
    if (detailEl) detailEl.classList.add('active');
    if (dispatchEl) dispatchEl.classList.remove('active');
    if (settingsEl) settingsEl.classList.remove('active');

    const cam = window.cctvCameraRegistry ? window.cctvCameraRegistry[camId] : null;
    if (cam) {
        const titleEl = document.getElementById('detail-cam-title');
        if (titleEl) titleEl.innerText = cam.name;
        const idBadge = document.getElementById('detail-cam-id-badge');
        if (idBadge) idBadge.innerText = cam.id;
        const locEl = document.getElementById('detail-cam-location');
        if (locEl) locEl.innerText = cam.location;
        const typeEl = document.getElementById('detail-cam-type');
        if (typeEl) typeEl.innerText = cam.type;

        const densVal = document.getElementById('detail-density-val');
        if (densVal) densVal.innerText = 'Awaiting Analysis';
        const chokepointVal = document.getElementById('detail-chokepoint-val');
        if (chokepointVal) {
            chokepointVal.innerText = 'Standby';
            chokepointVal.style.color = '#a1a1aa';
        }

        const countBadge = document.getElementById('detail-heatmap-count-badge');
        if (countBadge) countBadge.innerText = '0 Persons Detected';

        const alertsList = document.getElementById('detail-alerts-list');
        if (alertsList) {
            alertsList.innerHTML = '<div style="padding: 14px; background: #16161c; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 12px;">Waiting for live detection data from the active feed.</div>';
        }

        const roster = window.cctvPersonnelStore[camId] || [];
        const officer = roster.length > 0 ? roster[0].name : 'Duty Officer';

        renderPersonnelList();

        const webcamBox = document.getElementById('cam-detail-webcam-box');
        const opticalBox = document.getElementById('cam-detail-optical-box');
        const activeUpload = window.uploadedMediaByCam[cam.id] || null;
        window.uploadedMediaState = activeUpload;
        if (cam.id === 'CAM-01' && !activeUpload) {
            if (webcamBox) webcamBox.style.display = 'flex';
            if (opticalBox) opticalBox.style.display = 'none';
            setMediaUploadMode(cam.id);
            // Do NOT auto-start webcam — user clicks Start Recording
            window._camRecording = false;
            const btnStart = document.getElementById('btn-start-recording');
            const btnEnd   = document.getElementById('btn-end-recording');
            const badge    = document.getElementById('cam-rec-badge');
            if (btnStart) { btnStart.disabled = false; btnStart.style.opacity = '1'; btnStart.style.cursor = 'pointer'; }
            if (btnEnd)   { btnEnd.disabled = true; btnEnd.style.opacity = '0.4'; btnEnd.style.cursor = 'not-allowed'; }
            if (badge)    { badge.innerText = '● IDLE'; badge.style.color = '#71717a'; }
        } else {
            if (webcamBox) webcamBox.style.display = 'none';
            if (opticalBox) opticalBox.style.display = 'flex';
            setMediaUploadMode(cam.id);
            const input = document.getElementById('media-upload-input');
            if (input) {
                input.accept = 'video/*,image/*';
                input.onchange = handleUploadedMediaSelection;
            }
            const uploadBtn  = document.getElementById('media-upload-btn');
            const removeBtn  = document.getElementById('media-remove-btn');
            const analyzeBtn = document.getElementById('media-analyze-btn');
            if (uploadBtn)  uploadBtn.innerText = 'Upload Video / Image';
            // Hide Remove/Analyse until a file is loaded
            if (removeBtn)  removeBtn.style.display  = 'none';
            if (analyzeBtn) analyzeBtn.style.display = 'none';
            if (!activeUpload) {
                window.uploadedMediaState = null;
                window.analyzedPersonCentroids = [];
                window.yoloDetectionBoxes = [];
window.detectedFaceBoxes = [];
                window.lastZoneCounts = {zone1:0,zone2:0,zone3:0,zone4:0};
                updateZonePanel(window.lastZoneCounts);
                const metricEl = document.getElementById('cam-optical-person-metric');
                if (metricEl) metricEl.innerText = `● UPLOAD A VIDEO TO BEGIN • ANALYZED PERSONS: 0`;
                setTimeout(runOpticalCanvasAnimation, 200);
                runHeatmapCompositeAnimation();
            } else {
                // Already have media for this cam — show it but don't auto-analyse
                renderUploadedMediaToCanvas();
                if (removeBtn)  removeBtn.style.display  = 'inline-flex';
                if (analyzeBtn) analyzeBtn.style.display = 'inline-flex';
                runHeatmapCompositeAnimation();
                if (activeUpload.type === 'video') activeUpload.videoEl.play().catch(() => {});
                startUploadedMediaAnalysisLoop();
            }
            if (cam.id === 'CAM-01' && activeUpload) {
                if (webcamBox) webcamBox.style.display = 'none';
                if (opticalBox) opticalBox.style.display = 'flex';
            }
        }
    }

    setTimeout(runHeatmapAnimation, 200);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function returnToHome() {
    const homeEl = document.getElementById('home-page-container');
    const detailEl = document.getElementById('cam-detail-page-container');
    const dispatchEl = document.getElementById('dispatch-page-container');
    const settingsEl = document.getElementById('settings-page-container');
    if (homeEl) homeEl.classList.remove('hidden');
    if (detailEl) detailEl.classList.remove('active');
    if (dispatchEl) dispatchEl.classList.remove('active');
    if (settingsEl) settingsEl.classList.remove('active');
    const active = window.uploadedMediaByCam[window.currentActiveCamId];
    if (active) window.uploadedMediaState = active;
    startDashboardPreviewLoop();
    renderAllDashboardCameraTiles();
    closeAlertDrawer();
    if (!window.uploadedMediaByCam['CAM-01']) setTimeout(initCam01Webcam, 200);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
function showSettingsPage() {
    const home = document.getElementById('home-page-container');
    const detail = document.getElementById('cam-detail-page-container');
    const dispatch = document.getElementById('dispatch-page-container');
    const settings = document.getElementById('settings-page-container');
    if (home) home.classList.add('hidden');
    if (detail) detail.classList.remove('active');
    if (dispatch) dispatch.classList.remove('active');
    if (settings) settings.classList.add('active');
    closeAlertDrawer();
    initializeSettingsPage();
    window.scrollTo({top:0,behavior:'smooth'});
}

function escapeDispatchText(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function renderDispatchZoneDetails(selectedValue) {
    const panel = document.getElementById('dispatch-zone-assigned-personnel');
    if (!panel) return;
    const match = String(selectedValue || '').match(/CAM-\d+/);
    const camId = match ? match[0] : 'CAM-01';
    const camera = (window.cctvCameraRegistry || {})[camId] || {id:camId,name:camId,location:'Assigned zone'};
    const roster = (window.cctvPersonnelStore || {})[camId] || [];
    const data = (window.cameraDashboardData || {})[camId];
    const count = data && Number.isFinite(Number(data.count)) ? Number(data.count) : 0;
    const status = isOvercrowdingCount(count) ? ['OVERCROWDING','#ef4444'] : isCrowdingCount(count) ? ['CROWDING','#f97316'] : count ? ['MONITORING','#60a5fa'] : ['AWAITING DETECTION','#a1a1aa'];
    const peopleHtml = roster.length ? roster.map((person, index) => `
        <div style="display:flex;gap:12px;align-items:center;padding:13px;background:#17171e;border:1px solid #2c2c36;border-radius:10px;margin-top:9px">
          <div style="width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0072f5,#a855f7);color:#fff;font-weight:700">${escapeDispatchText((person.name||'P').trim().charAt(0).toUpperCase())}</div>
          <div style="min-width:0"><div style="font-size:13px;color:#fff;font-weight:700">${escapeDispatchText(person.name||`Assigned person ${index+1}`)}</div><div style="font-size:11px;color:#93c5fd;margin-top:4px">${escapeDispatchText(person.role||'Zone personnel')}</div><div style="font-size:10px;color:#a1a1aa;margin-top:3px">${escapeDispatchText(person.phone||'No contact number')}</div></div>
        </div>`).join('') : '<div style="padding:14px;border:1px dashed #3f3f46;border-radius:10px;margin-top:10px;color:#a1a1aa;font-size:12px">No personnel are assigned to this zone yet.</div>';
    panel.innerHTML = `<div style="padding:18px;background:#111117;border:1px solid #292932;border-radius:14px;color:#f4f4f5;min-height:320px">
      <div style="font-size:11px;color:#60a5fa;font-weight:700;letter-spacing:.08em">ZONE ASSIGNMENT</div>
      <div style="font-size:18px;font-weight:750;margin-top:8px">${escapeDispatchText(camera.name||camId)}</div>
      <div style="font-size:12px;color:#a1a1aa;margin-top:5px">${escapeDispatchText(camera.location||'')}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:18px;padding:12px;background:#191920;border-radius:9px"><span style="font-size:11px;color:#a1a1aa">People detected</span><strong style="font-size:15px;color:#fff">${count}</strong></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding:12px;background:#191920;border-radius:9px"><span style="font-size:11px;color:#a1a1aa">Zone status</span><strong style="font-size:10px;color:${status[1]}">${status[0]}</strong></div>
      <div style="margin-top:21px;font-size:13px;font-weight:700;color:#fff">Assigned personnel <span style="font-size:11px;color:#71717a">(${roster.length})</span></div>${peopleHtml}
    </div>`;
}
function showDispatchPage() {
    const home = document.getElementById('home-page-container');
    const detail = document.getElementById('cam-detail-page-container');
    const dispatch = document.getElementById('dispatch-page-container');
    const settings = document.getElementById('settings-page-container');
    if (home) home.classList.add('hidden');
    if (detail) detail.classList.remove('active');
    if (dispatch) dispatch.classList.add('active');
    if (settings) settings.classList.remove('active');
    renderDispatchZoneDetails(getDispatchSelectedValue());
    window.scrollTo({top:0,behavior:'smooth'});
}
function getDispatchSelectedValue() {
    const root = document.getElementById('dispatch-camera-select');
    if (!root) return 'CAM-01';
    const control = root.querySelector('select, input, [role="combobox"]');
    return control ? (control.value || control.getAttribute('aria-valuetext') || control.textContent || 'CAM-01') : root.textContent || 'CAM-01';
}
function renderNotificationPanel() {
    const list = document.getElementById('overcrowding-alert-list');
    const badge = document.getElementById('alert-bell-count');
    const alerts = Object.entries(window.cameraDashboardData || {})
        .filter(([camId, data]) => isOvercrowdingCount(data.count))
        .sort((a,b) => (b[1].updatedAt||0) - (a[1].updatedAt||0));
    if (badge) {
        badge.textContent = String(alerts.length);
        badge.style.display = alerts.length ? 'flex' : 'none';
    }
    if (!list) return;
    list.innerHTML = alerts.length ? alerts.map(([camId,data]) => `
      <button onclick="closeAlertDrawer();selectCameraPage('${camId}')" style="width:100%;text-align:left;background:#241316;border:1px solid #7f1d1d;border-radius:10px;padding:12px;margin-bottom:9px;color:#fff;cursor:pointer">
        <div style="display:flex;justify-content:space-between;align-items:center"><b style="color:#f87171">OVERCROWDING</b><span style="font:11px monospace;color:#a1a1aa">${camId}</span></div>
        <div style="font-size:13px;margin-top:7px">${Number(data.count)||0} people detected</div>
        <div style="font-size:10px;color:#a1a1aa;margin-top:4px">Open this camera zone</div>
      </button>`).join('') : '<div style="padding:18px 12px;text-align:center;color:#a1a1aa;font-size:12px;border:1px dashed #333;border-radius:10px">No active overcrowding alerts.</div>';
}
function toggleAlertDrawer() {
    const drawer = document.getElementById('overcrowding-alert-drawer');
    if (!drawer) return;
    drawer.style.display = drawer.style.display === 'block' ? 'none' : 'block';
    renderNotificationPanel();
}
function closeAlertDrawer() {
    const drawer = document.getElementById('overcrowding-alert-drawer');
    if (drawer) drawer.style.display = 'none';
}
document.addEventListener('change', event => {
    if (event.target && event.target.closest && event.target.closest('#dispatch-camera-select')) renderDispatchZoneDetails(getDispatchSelectedValue());
});
setInterval(() => {
    const page = document.getElementById('dispatch-page-container');
    if (page && page.classList.contains('active')) renderDispatchZoneDetails(getDispatchSelectedValue());
}, 1200);

function handleGlowNav(target) {
    document.querySelectorAll('.glow-menu-item').forEach(el => el.classList.remove('active'));
    const item = document.getElementById('nav-item-' + target);
    if (item) item.classList.add('active');

    if (target === 'home') {
        returnToHome();
    } else if (target === 'notifications') {
        returnToHome();
        toggleAlertDrawer();
    } else if (target === 'dispatch') {
        showDispatchPage();
    } else if (target === 'settings') {
        showSettingsPage();
    }
}

// Auto-register upload input handler
window.addEventListener('DOMContentLoaded', () => { 
    const uploadInput = document.getElementById('media-upload-input');
    if (uploadInput) uploadInput.onchange = handleUploadedMediaSelection;
    const drawer = document.getElementById('overcrowding-alert-drawer');
    if (drawer) document.body.appendChild(drawer);
    startDashboardPreviewLoop();
    renderPersonnelList();
    // DO NOT auto-start webcam — requires Start Recording button
});
window.addEventListener('load', () => { 
    const uploadInput = document.getElementById('media-upload-input');
    if (uploadInput) uploadInput.onchange = handleUploadedMediaSelection;
    const drawer = document.getElementById('overcrowding-alert-drawer');
    if (drawer && drawer.parentElement !== document.body) document.body.appendChild(drawer);
    startDashboardPreviewLoop();
    renderPersonnelList();
});

setInterval(function() {
    const d = new Date();
    const timeStr = d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0').slice(0, 2);
    const ts = document.getElementById('cctv-timestamp');
    if (ts) ts.innerText = timeStr;
    const tsW = document.getElementById('cam-detail-timestamp-webcam');
    if (tsW) tsW.innerText = timeStr;
    const tsO = document.getElementById('cam-detail-timestamp-optical');
    if (tsO) tsO.innerText = timeStr;
}, 200);

// ── Zone panel updater: refresh the small zone status grid below the heatmap ──
function updateZonePanel(zoneCounts) {
    const zc = zoneCounts || {zone1:0,zone2:0,zone3:0,zone4:0};
    ['zone1','zone2','zone3','zone4'].forEach((k, i) => {
        const panelEl = document.getElementById('zone-panel-' + k);
        if (!panelEl) return;
        const count = zc[k] || 0;
        const overcrowded = count >= ZONE_OVERCROWD_THRESHOLD;
        panelEl.style.borderColor = overcrowded ? '#ef4444' : '#22c55e';
        panelEl.style.background = overcrowded ? 'rgba(239,68,68,0.12)' : 'rgba(34,197,94,0.08)';
        const countEl = panelEl.querySelector('.zone-count');
        if (countEl) {
            countEl.textContent = count;
            countEl.style.color = overcrowded ? '#f87171' : '#4ade80';
        }
        const statusEl = panelEl.querySelector('.zone-status');
        if (statusEl) {
            statusEl.textContent = overcrowded ? '⚠ OVERCROWDED' : '✓ NORMAL';
            statusEl.style.color = overcrowded ? '#f87171' : '#4ade80';
        }
    });
}

// ── HEATMAP COMPOSITE CANVAS ─────────────────────────────────────────────────
// Renders the actual video/image frame + thermal heatmap overlay in sync.
// This replaces the old static crowd-heatmap-canvas approach.
window._heatmapCompFrameId = null;

function runHeatmapCompositeAnimation() {
    const canvas = document.getElementById('heatmap-composite-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Cancel any existing loop
    if (window._heatmapCompFrameId) { cancelAnimationFrame(window._heatmapCompFrameId); window._heatmapCompFrameId = null; }

    const idleMsg = document.getElementById('heatmap-idle-msg');

    function paintFrame() {
        window._heatmapCompFrameId = requestAnimationFrame(paintFrame);

        const W = canvas.width;
        const H = canvas.height;
        const camId   = window.currentActiveCamId || 'CAM-01';
        const state   = window.uploadedMediaState;
        const centroids = window.analyzedPersonCentroids || [];
        const boxes = window.yoloDetectionBoxes || [];
        const zc      = window.lastZoneCounts || {zone1:0,zone2:0,zone3:0,zone4:0};
        // Use one point per detected person. Box centers map directly to the
        // frame; centroids are the fallback if boxes were omitted.
        const personPoints = boxes.length ? boxes.map(b => ({
            x: ((Number(b.x1) + Number(b.x2)) / 2) * W,
            y: ((Number(b.y1) + Number(b.y2)) / 2) * H,
            radius: Math.max(20, (Number(b.y2) - Number(b.y1)) * H * 0.85)
        })) : centroids.map(p => ({
            x: (p.x / 480) * W,
            y: (p.y / 190) * H,
            radius: Math.max(20, (p.radius || 40) / 480 * W * 1.5)
        }));
        const personCount = personPoints.length;
        const heatColor = personCount >= 8 ? '#ef4444'
            : personCount >= 4 ? '#f97316'
            : personCount >= 2 ? '#eab308'
            : '#3b82f6';
        const hasData = personCount > 0;

        // ── Background ──
        ctx.fillStyle = '#06070a';
        ctx.fillRect(0, 0, W, H);

        // ── Draw underlying video / image frame ──
        let drewFrame = false;
        if (camId === 'CAM-01' && window._camRecording) {
            const vidEl = document.getElementById('cam-detail-video');
            if (vidEl && vidEl.readyState >= 2 && vidEl.videoWidth > 0) {
                ctx.drawImage(vidEl, 0, 0, W, H);
                drewFrame = true;
            }
        } else if (window._opticalLiveActive && camId !== 'CAM-01') {
            const opticalVideo = document.getElementById('optical-live-video');
            if (opticalVideo && opticalVideo.readyState >= 2 && opticalVideo.videoWidth > 0) {
                ctx.drawImage(opticalVideo, 0, 0, W, H);
                drewFrame = true;
            }
        } else if (state) {
            if (state.type === 'video' && state.videoEl && state.videoEl.readyState >= 2 && state.videoEl.videoWidth > 0) {
                ctx.drawImage(state.videoEl, 0, 0, W, H);
                drewFrame = true;
            } else if (state.type === 'image' && state.imgEl) {
                ctx.drawImage(state.imgEl, 0, 0, W, H);
                drewFrame = true;
            }
        }

        if (!drewFrame) {
            // Show subtle grid while idle
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            ctx.lineWidth = 1;
            for (let x = 0; x < W; x += 30) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
            for (let y = 0; y < H; y += 30) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
        }

        // ── Idle message ──
        if (idleMsg) idleMsg.style.display = (!drewFrame && !hasData) ? 'flex' : 'none';

        if (!hasData && !drewFrame) return; // nothing to paint

        // ── Heatmap Gaussian blobs (scale from the 480x190 normalised coords) ──
        if (personPoints.length > 0) {
            ctx.save();
            // Each person contributes a soft circle. Overlaps add together,
            // making crowded spots visibly hotter.
            ctx.globalCompositeOperation = 'lighter';
            ctx.globalAlpha = 1.0;
            personPoints.forEach(p => {
                const px = p.x;
                const py = p.y;
                const r  = Math.max(22, p.radius);
            const localCount = personPoints.filter(other => Math.hypot(other.x - px, other.y - py) < Math.min(W, H) * 0.2).length;
            const color = localCount >= 8 ? '#ef4444' : localCount >= 4 ? '#f97316' : localCount >= 2 ? '#eab308' : '#3b82f6';
                const grad = ctx.createRadialGradient(px, py, 0, px, py, r);
                grad.addColorStop(0, `${color}e6`);
                grad.addColorStop(0.35, `${color}99`);
                grad.addColorStop(0.72, `${color}44`);
                grad.addColorStop(1,    'rgba(0,0,0,0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(px, py, r, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.restore();
        }

        // Face detections are marked separately and never increase person counts.
        drawFaceMarkers(ctx, window.detectedFaceBoxes || [], W, H);

        // ── Mini zone border overlays ──
        if (hasData || drewFrame) {
            ctx.save();
            ctx.globalAlpha = 0.9;
            const midX = W/2, midY = H/2;
            const miniZones = [
                {key:'zone1', label:'Z1', x:0,    y:0,    bw:midX, bh:midY},
                {key:'zone2', label:'Z2', x:midX, y:0,    bw:midX, bh:midY},
                {key:'zone3', label:'Z3', x:0,    y:midY, bw:midX, bh:midY},
                {key:'zone4', label:'Z4', x:midX, y:midY, bw:midX, bh:midY},
            ];
            miniZones.forEach(z => {
                const count = zc[z.key] || 0;
                const overcrowded = count >= ZONE_OVERCROWD_THRESHOLD;
                ctx.strokeStyle = overcrowded ? 'rgba(239,68,68,0.85)' : 'rgba(34,197,94,0.6)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([5,3]);
                ctx.strokeRect(z.x+1, z.y+1, z.bw-2, z.bh-2);
                ctx.setLineDash([]);
                ctx.font = 'bold 9px monospace';
                ctx.fillStyle = overcrowded ? 'rgba(248,113,113,0.95)' : 'rgba(74,222,128,0.95)';
                ctx.fillText(`${z.label}:${count}`, z.x + 4, z.y + 12);
            });
            // Cross-hair
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 1;
            ctx.setLineDash([6,3]);
            ctx.beginPath(); ctx.moveTo(midX,0); ctx.lineTo(midX,H); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0,midY); ctx.lineTo(W,midY); ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // ── Legend label ──
        if (personPoints.length > 0) {
            ctx.save();
            personPoints.forEach(p => {
                const localCount = personPoints.filter(other => Math.hypot(other.x - p.x, other.y - p.y) < Math.min(W, H) * 0.2).length;
                const dotColor = localCount >= 8 ? '#ef4444' : localCount >= 4 ? '#f97316' : localCount >= 2 ? '#eab308' : '#3b82f6';
                ctx.beginPath();
                ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = dotColor;
                ctx.fill();
                ctx.lineWidth = 4;
                ctx.strokeStyle = '#09090b';
                ctx.stroke();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
            });
            ctx.restore();
        }

        if (drewFrame) {
            const status = personCount > 0
                ? `${personCount} PEOPLE DETECTED`
                : (window.heatmapAnalysisComplete ? 'NO PEOPLE DETECTED IN THIS FRAME' : 'ANALYZING FRAME…');
            ctx.font = 'bold 10px sans-serif';
            const statusWidth = ctx.measureText(status).width + 14;
            ctx.fillStyle = 'rgba(0,0,0,0.78)';
            ctx.fillRect(W - statusWidth - 6, 6, statusWidth, 22);
            ctx.fillStyle = personCount > 0 ? heatColor : '#ffffff';
            ctx.fillText(status, W - statusWidth + 1, 21);
        }

        if (hasData) {
            ctx.fillStyle = heatColor;
            ctx.font = '8px monospace';
            ctx.fillText(`DETECTED PEOPLE: ${personCount} • HEATMAP`, 6, H - 5);
        }
    }
    paintFrame();
}

// Keep runHeatmapAnimation as an alias so old call-sites still work
function runHeatmapAnimation() {
    runHeatmapCompositeAnimation();
}




// Live optical canvas updates — now with zone grid overlay
function runOpticalCanvasAnimation() {
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    function animateOptical() {
        if (window.uploadedMediaState && window.uploadedMediaState.camId === window.currentActiveCamId) {
            renderUploadedMediaToCanvas();
            window.opticalAnimationFrame = requestAnimationFrame(animateOptical);
            return;
        }
        if (!document.getElementById('cctv-optical-stream-canvas')) return;
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (window._opticalLiveActive) {
            const liveVideo = document.getElementById('optical-live-video');
            if (liveVideo && liveVideo.readyState >= 2 && liveVideo.videoWidth) ctx.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);
        }

        const scanY = (Date.now() / 18) % canvas.height;
        ctx.strokeStyle = 'rgba(0,114,245,0.15)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(canvas.width, scanY); ctx.stroke();

        const boxes = window.yoloDetectionBoxes || [];
        if (Array.isArray(boxes) && boxes.length > 0) {
            boxes.forEach((b, idx) => {
                const x1 = Number(b.x1||0) * canvas.width;
                const y1 = Number(b.y1||0) * canvas.height;
                const bw = (Number(b.x2||0)-Number(b.x1||0)) * canvas.width;
                const bh = (Number(b.y2||0)-Number(b.y1||0)) * canvas.height;
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(x1, y1, bw, bh);
                ctx.fillStyle = '#4ade80';
                ctx.font = '9px monospace';
                ctx.fillText(`P${idx+1}`, x1+3, Math.max(11, y1-6));
            });
        }

        // Zone grid overlay
        drawZoneGridOnCanvas(ctx, canvas.width, canvas.height, window.lastZoneCounts, window.analyzedPersonCentroids);

        window.opticalAnimationFrame = requestAnimationFrame(animateOptical);
    }
    window.opticalAnimationFrame = requestAnimationFrame(animateOptical);
}
</script>
"""

with gr.Blocks(title="DensiTrack — Real-Time CCTV Command Studio") as demo:
    camera_store = gr.State(INITIAL_CAMERAS)
    alert_store = gr.State(INITIAL_ALERTS)

    # Injected Visuvate Design System Stylesheet
    gr.HTML(f"<style>{VISUVATE_CSS}</style>")

    # 1. Floating Top Navigation Bar (Glow Menu) - White Logos & Text
    gr.HTML("""
    <div class="floating-nav-wrapper">
        <nav class="glow-menu-bar" id="glow-nav-bar">
            <div class="glow-menu-bg"></div>
            <ul class="glow-menu-list">
                <!-- Home -->
                <li class="glow-menu-item active" id="nav-item-home">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('home')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
                        </span>
                        <span style="color: #ffffff;">Home</span>
                    </a>
                </li>
                <!-- Dispatch Team -->
                <li class="glow-menu-item" id="nav-item-dispatch">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('dispatch')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                        </span>
                        <span style="color: #ffffff;">Dispatch Team</span>
                    </a>
                </li>
                <!-- Settings -->
                <li class="glow-menu-item" id="nav-item-settings">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('settings')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                        </span>
                        <span style="color: #ffffff;">Settings</span>
                    </a>
                </li>
                <li class="glow-menu-item" id="nav-item-notifications"><button onclick="toggleAlertDrawer()" aria-label="Open notifications" title="Notifications" style="position:relative;width:36px;height:36px;margin:0 5px;border:1px solid #34343e;border-radius:50%;background:#17171d;color:#f4f4f5;display:flex;align-items:center;justify-content:center;cursor:pointer"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/></svg><span id="alert-bell-count" style="display:none;position:absolute;right:-3px;top:-3px;min-width:16px;height:16px;border-radius:9px;background:#ef4444;color:white;font:10px sans-serif;align-items:center;justify-content:center;padding:0 4px">0</span></button></li>
            </ul>
        </nav>
    </div>
    """)

    # Hidden JS-to-Python Bridge Inputs for Page Switching
    hidden_selected_cam = gr.Textbox(elem_id="selected-cam-input", visible=False)
    hidden_cam_trigger = gr.Button(elem_id="selected-cam-trigger", visible=False)
    hidden_home_trigger = gr.Button(elem_id="nav-to-home-trigger", visible=False)

    # =========================================================
    # PAGE 1: HOME — CCTV Wall & Surveillance Command Center
    # =========================================================
    with gr.Column(visible=True, elem_classes=["visuvate-canvas"], elem_id="home-page-container") as home_view:
        gr.HTML("""
        <div id="overcrowding-alert-drawer" style="display:none;position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.34)" onclick="if(event.target===this)closeAlertDrawer()">
          <aside style="position:absolute;right:0;top:0;width:25vw;min-width:300px;max-width:440px;height:100%;box-sizing:border-box;background:#111116;border-left:1px solid #2a2a34;padding:20px;overflow:auto;box-shadow:-18px 0 50px #0008">
            <div style="display:flex;justify-content:space-between;align-items:center;margin:8px 0 18px"><div><div style="font-size:17px;font-weight:700;color:#fff">Notifications</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px">Crowding alerts by camera zone</div></div><button onclick="closeAlertDrawer()" style="background:#22222a;color:white;border:1px solid #383842;border-radius:6px;padding:5px 9px;cursor:pointer">Close</button></div>
            <div id="overcrowding-alert-list"></div>
          </aside>
        </div>
        """)
        with gr.Row():
            # -------------------------------------------------
            # LEFT COLUMN: Camera Directory (Sidebar)
            # -------------------------------------------------
            with gr.Column(scale=1, min_width=270):
                gr.HTML("""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: #141417; border-radius: 8px; margin-bottom: 8px; border: 1px solid #232328;">
                    <span style="font-size: 12px; font-weight: 600; color: #f4f4f5;">Cameras</span>
                    <span style="font-size: 10px; color: #22c55e; font-weight: 600;">● Active</span>
                </div>
                """)

                camera_sidebar = gr.HTML(value=render_camera_sidebar_html(INITIAL_CAMERAS))

                with gr.Accordion("➕ Deploy New Camera Layer", open=False):
                    new_cam_name = gr.Textbox(label="Camera Identifier", placeholder="CAM 05 - East Ticket Counter")
                    new_cam_location = gr.Textbox(label="Monitored Zone", placeholder="Terminal 2 Gate 4")
                    new_cam_type = gr.Dropdown(
                        label="Source Protocol",
                        choices=["IP / RTSP Stream", "PTZ Dome Optical", "System Hardware Webcam", "Simulated Video Feed"],
                        value="IP / RTSP Stream",
                    )
                    new_cam_personnel = gr.Textbox(label="Designated Officer", placeholder="Officer K. Verma")
                    new_cam_role = gr.Dropdown(
                        label="Operational Role",
                        choices=["Crowd Control Lead", "Perimeter Security Officer", "Medical First Responder", "Emergency Dispatch Commander", "Sector Supervisor"],
                        value="Crowd Control Lead",
                    )
                    add_cam_btn = gr.Button("Add Camera to Canvas", variant="primary", elem_classes=["btn-visuvate-filter"])
                    add_cam_status = gr.Markdown(value="", visible=True)

            # -------------------------------------------------
            # CENTER / RIGHT: Compact CCTV Canvas Wall (Square Quad Grid)
            # Note: "CCTV Canvas Feeds (Quad Grid)" and "QUAD VIEW" headers have been removed
            # -------------------------------------------------
            with gr.Column(scale=3):
                # QUADRANTS 1 & 2 (Top Row)
                with gr.Row():
                    # Quadrant 1: CAM 01 Live Hardware Webcam
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-01')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #ffffff;">CAM 01 — MAIN ENTRANCE</span>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span class="rec-pill">● REC</span>
                                <span id="cam01-live-badge" style="color: #22c55e; font-size: 10px; font-weight: 600;">LIVE FEED</span>
                            </div>
                        </div>
                        <div style="position: relative; width: 100%; height: 160px; background: #09090b; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid #202026; cursor: pointer;" onclick="selectCameraPage('CAM-01')">
                            <video id="cctv-live-video" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; display: none;"></video>
                            <canvas id="dashboard-camera-canvas-CAM-01" width="640" height="360" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none"></canvas>
                            <div style="position:absolute;left:9px;top:8px;background:#050508bb;border-radius:5px;padding:3px 7px;font:10px monospace;z-index:2"><span id="dashboard-camera-count-CAM-01">0 PEOPLE</span></div>
                            
                            <div id="cctv-webcam-placeholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; cursor: pointer; width: 100%; height: 100%; padding: 12px; text-align: center;" onclick="event.stopPropagation(); initCam01Webcam();">
                                <div style="width: 40px; height: 40px; border-radius: 50%; background: rgba(0, 114, 245, 0.2); border: 1px solid rgba(0, 114, 245, 0.5); display: flex; align-items: center; justify-content: center; color: #0072f5; box-shadow: 0 0 16px rgba(0, 114, 245, 0.25);">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
                                </div>
                                <div style="color: #ffffff; font-size: 12px; font-weight: 600;">Start Live Camera Feed</div>
                                <div style="color: #71717a; font-size: 10px;">Click to stream live camera into CAM 01</div>
                            </div>

                            <!-- Live CCTV Overlay HUD -->
                            <div id="cctv-hud-overlay" style="position: absolute; bottom: 6px; left: 8px; right: 8px; display: flex; justify-content: space-between; align-items: center; pointer-events: none; opacity: 0; transition: opacity 0.3s ease;">
                                <span style="font-size: 9px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● 30 FPS • 720p HD</span>
                                <span id="cctv-timestamp" style="font-size: 9px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px;"></span>
                            </div>
                        </div>
                        """)
                        start_webcam_ui_btn = gr.Button("🔴 Connect Live Webcam", elem_classes=["btn-visuvate-capture"])

                    # Quadrant 2: CAM 02 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-02')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 02 — SECTOR 3 CONCOURSE</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div style="position:relative;height:165px;margin:6px;border:1px solid #202026;border-radius:8px;overflow:hidden;background:#050508;cursor:pointer" onclick="selectCameraPage('CAM-02')"><canvas id="dashboard-camera-canvas-CAM-02" width="640" height="360" style="width:100%;height:100%;object-fit:cover"></canvas><div id="dashboard-camera-placeholder-CAM-02" class="dashboard-camera-placeholder"><b>CAM 02 FEED</b><span>Upload a video or image to preview and analyze</span></div><div class="dashboard-camera-count"><span id="dashboard-camera-count-CAM-02">0 PEOPLE</span></div></div>
                        """)

                # QUADRANTS 3 & 4 (Bottom Row)
                with gr.Row():
                    # Quadrant 3: CAM 03 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-03')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 03 — WEST PROMENADE</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div style="position:relative;height:165px;margin:6px;border:1px solid #202026;border-radius:8px;overflow:hidden;background:#050508;cursor:pointer" onclick="selectCameraPage('CAM-03')"><canvas id="dashboard-camera-canvas-CAM-03" width="640" height="360" style="width:100%;height:100%;object-fit:cover"></canvas><div id="dashboard-camera-placeholder-CAM-03" class="dashboard-camera-placeholder"><b>CAM 03 FEED</b><span>Upload a video or image to preview and analyze</span></div><div class="dashboard-camera-count"><span id="dashboard-camera-count-CAM-03">0 PEOPLE</span></div></div>
                        """)

                    # Quadrant 4: CAM 04 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-04')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 04 — QUEUE CORRIDOR</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div style="position:relative;height:165px;margin:6px;border:1px solid #202026;border-radius:8px;overflow:hidden;background:#050508;cursor:pointer" onclick="selectCameraPage('CAM-04')"><canvas id="dashboard-camera-canvas-CAM-04" width="640" height="360" style="width:100%;height:100%;object-fit:cover"></canvas><div id="dashboard-camera-placeholder-CAM-04" class="dashboard-camera-placeholder"><b>CAM 04 FEED</b><span>Upload a video or image to preview and analyze</span></div><div class="dashboard-camera-count"><span id="dashboard-camera-count-CAM-04">0 PEOPLE</span></div></div>
                        """)

    with gr.Column(visible=True, elem_classes=["visuvate-canvas"], elem_id="dispatch-page-container") as dispatch_view:
        gr.HTML("""
        <div style="max-width:1320px;margin:18px auto 14px;padding:0 18px;display:flex;justify-content:space-between;align-items:center;gap:16px">
          <div><div style="font-size:22px;font-weight:750;color:#fff">Dispatch Team</div><div style="font-size:12px;color:#a1a1aa;margin-top:5px">Send zone specific directions and review assigned personnel.</div></div>
        </div>
        """)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                with gr.Group():
                    dispatch_cam = gr.Dropdown(
                        label="Target Camera / Zone",
                        choices=[f"{c['id']} - {c['name']}" for c in INITIAL_CAMERAS],
                        value=f"{INITIAL_CAMERAS[0]['id']} - {INITIAL_CAMERAS[0]['name']}",
                        elem_id="dispatch-camera-select",
                    )
                    dispatch_severity = gr.Dropdown(
                        label="Alert Urgency",
                        choices=["Critical", "Warning", "Info"],
                        value="Critical",
                    )
                    dispatch_msg = gr.Textbox(
                        label="Incident Directive / Instructions",
                        value="Immediate attention required! Deploy crowd dispersion barriers at exit gate.",
                        lines=5,
                    )
                    send_alert_btn = gr.Button("Send Dispatch Alert", elem_classes=["btn-visuvate-alert"])
                    dispatch_status_box = gr.Markdown(value="", visible=True)
            with gr.Column(scale=2, min_width=330):
                dispatch_zone_details = gr.HTML(
                    value='<div style="padding:18px;background:#111117;border:1px solid #292932;border-radius:14px;color:#a1a1aa">Select a zone to view its assigned personnel.</div>',
                    elem_id="dispatch-zone-assigned-personnel",
                )

        gr.HTML("""<div style="height:1px;background:#292932;margin:22px 18px 14px"></div>""")
        with gr.Column():
            gr.HTML("""
            <div id="section-notifications" style="display:flex;justify-content:space-between;align-items:center;margin:0 18px 8px;scroll-margin-top:60px">
              <span style="font-size:16px;font-weight:700;color:#fff">Live Incident Stream &amp; Dispatch Control</span>
              <span style="font-size:11px;color:#71717a">Active Surveillance Logs</span>
            </div>
            """)
            with gr.Row():
                filter_severity = gr.Dropdown(label="Filter Severity", choices=["All", "Critical", "Warning", "Info"], value="All", scale=1)
                filter_camera = gr.Dropdown(label="Filter by Camera Source", choices=["All"] + [c['id'] for c in INITIAL_CAMERAS], value="All", scale=1)
                apply_filter_btn = gr.Button("Apply Filters", scale=1, elem_classes=["btn-visuvate-filter"])
            alerts_table = gr.HTML(value=render_alerts_table_html(INITIAL_ALERTS))

    gr.HTML("""
    <div id="settings-page-container" style="max-width:1050px;margin:20px auto;padding:24px;background:#101015;border:1px solid #292932;border-radius:16px;color:#f4f4f5;min-height:65vh;box-sizing:border-box">
      <div style="border-bottom:1px solid #2c2c35;padding-bottom:18px;margin-bottom:22px"><div style="font-size:22px;font-weight:750">Alert Settings</div><div style="font-size:12px;color:#a1a1aa;margin-top:5px">Set the people-count limits used for crowding and overcrowding alerts.</div></div>
      <div style="max-width:760px;padding:18px;background:#16161d;border:1px solid #292933;border-radius:12px;margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px"><div><div style="font-size:14px;font-weight:700;color:#fb923c">Crowding threshold</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px">Warning status begins above this count.</div></div><strong id="settings-crowding-value" style="color:#fdba74;white-space:nowrap">Above 30 people</strong></div>
        <input id="settings-crowding-threshold" type="range" min="1" max="100" value="30" oninput="saveAlertThresholds(this.value,document.getElementById('settings-overcrowding-threshold').value)" style="width:100%;margin-top:18px;accent-color:#f97316">
        <div style="display:flex;justify-content:space-between;color:#71717a;font-size:10px"><span>1 person</span><span>100 people</span></div>
      </div>
      <div style="max-width:760px;padding:18px;background:#16161d;border:1px solid #292933;border-radius:12px;margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:12px"><div><div style="font-size:14px;font-weight:700;color:#f87171">Overcrowding threshold</div><div style="font-size:11px;color:#a1a1aa;margin-top:4px">Critical status and bell alerts begin above this count.</div></div><strong id="settings-overcrowding-value" style="color:#fca5a5;white-space:nowrap">Above 50 people</strong></div>
        <input id="settings-overcrowding-threshold" type="range" min="2" max="150" value="50" oninput="saveAlertThresholds(document.getElementById('settings-crowding-threshold').value,this.value)" style="width:100%;margin-top:18px;accent-color:#ef4444">
        <div style="display:flex;justify-content:space-between;color:#71717a;font-size:10px"><span>2 people</span><span>150 people</span></div>
      </div>
      <div style="max-width:760px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 2px;color:#a1a1aa;font-size:11px"><span>Changes save automatically in this browser.</span><button onclick="saveAlertThresholds(30,50)" style="background:#1c1c24;color:#d4d4d8;border:1px solid #373742;border-radius:7px;padding:7px 10px;cursor:pointer">Reset defaults</button></div>
    </div>
    """)

    # =========================================================
    # PAGE 2: DEDICATED CAMERA VIEW PAGE
    # - Menu bar on top (persistent)
    # - Center: Camera feed (Webcam for CAM-01, optical for others)
    # - Bottom Left: Crowd Density Heatmap based on feed
    # - Bottom Right: Person allotted in area & Camera Alerts
    # =========================================================
    with gr.Column(visible=True, elem_classes=["visuvate-canvas"], elem_id="cam-detail-page-container") as cam_detail_view:
        cam_detail_content = gr.HTML(value=render_camera_detail_full_html(INITIAL_CAMERAS[0], INITIAL_ALERTS))

    # 2. Hardware Webcam Connection
    start_webcam_ui_btn.click(
        fn=None,
        js="() => { if (typeof initCam01Webcam === 'function') initCam01Webcam(); }"
    )

    # 3. Add New Camera
    def handle_add_camera(name, location, ctype, officer, role, current_cameras, current_alerts):
        if not name or not location:
            return (
                current_cameras,
                render_camera_sidebar_html(current_cameras),
                gr.update(choices=[f"{c['id']} - {c['name']}" for c in current_cameras]),
                gr.update(choices=["All"] + [c['id'] for c in current_cameras]),
                "⚠️ Please provide both Camera Name and Monitored Zone.",
                current_alerts,
                render_alerts_table_html(current_alerts),
            )

        new_id = f"CAM-0{len(current_cameras) + 1}"
        new_entry = {
            "id": new_id,
            "name": name.strip(),
            "location": location.strip(),
            "type": ctype,
            "status": "LIVE",
            "assigned_name": officer.strip() if officer else "Duty Officer",
            "assigned_role": role,
            "density": "Normal (35%)",
        }
        updated_cameras = current_cameras + [new_entry]

        now = datetime.datetime.now().strftime("%H:%M:%S")
        new_alert = {
            "id": f"ALT-{100 + len(current_alerts) + 1}",
            "camera_id": new_id,
            "camera_name": name.strip(),
            "severity": "Info",
            "message": f"New camera layer deployed and assigned to {officer} ({role}).",
            "assigned_person": f"{officer} ({role})",
            "timestamp": now,
            "status": "Active",
        }
        updated_alerts = [new_alert] + current_alerts

        cam_choices = [f"{c['id']} - {c['name']}" for c in updated_cameras]
        filter_choices = ["All"] + [c['id'] for c in updated_cameras]

        msg = f"✅ **{name}** ({new_id}) successfully registered on Canvas and mapped to **{officer}**!"
        return (
            updated_cameras,
            render_camera_sidebar_html(updated_cameras),
            gr.update(choices=cam_choices, value=cam_choices[-1]),
            gr.update(choices=filter_choices),
            msg,
            updated_alerts,
            render_alerts_table_html(updated_alerts),
        )

    add_cam_btn.click(
        fn=handle_add_camera,
        inputs=[new_cam_name, new_cam_location, new_cam_type, new_cam_personnel, new_cam_role, camera_store, alert_store],
        outputs=[camera_store, camera_sidebar, dispatch_cam, filter_camera, add_cam_status, alert_store, alerts_table],
    )

    # 4. Filter Alerts
    def handle_filter_alerts(severity, cam_id, alerts):
        return render_alerts_table_html(alerts, filter_severity=severity, filter_cam=cam_id)

    apply_filter_btn.click(
        fn=handle_filter_alerts,
        inputs=[filter_severity, filter_camera, alert_store],
        outputs=[alerts_table],
    )

    # 5. Dispatch Alert
    def handle_send_alert(target_cam_str, severity, message, current_alerts, cameras):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        cam_id = target_cam_str.split(" - ")[0] if " - " in target_cam_str else target_cam_str

        assigned_text = "Security Dispatch"
        for c in cameras:
            if c["id"] == cam_id:
                assigned_text = f"{c['assigned_name']} ({c['assigned_role']})"
                break

        new_alert = {
            "id": f"ALT-{100 + len(current_alerts) + 1}",
            "camera_id": cam_id,
            "camera_name": target_cam_str,
            "severity": severity,
            "message": message,
            "assigned_person": assigned_text,
            "timestamp": now,
            "status": "Dispatched",
        }

        updated_alerts = [new_alert] + current_alerts
        status_msg = f"🔔 **Alert transmitted!** [{now}] Notified **{assigned_text}** for **{target_cam_str}** with **{severity}** priority."

        return (
            updated_alerts,
            render_alerts_table_html(updated_alerts),
            status_msg,
        )

    send_alert_btn.click(
        fn=handle_send_alert,
        inputs=[dispatch_cam, dispatch_severity, dispatch_msg, alert_store, camera_store],
        outputs=[alert_store, alerts_table, dispatch_status_box],
    )

    demo.load(
        fn=None,
        js="() => { console.log('[DensiTrack] App ready. YOLO endpoint at /yolo_predict'); initializeSettingsPage(); renderNotificationPanel(); renderDispatchZoneDetails(getDispatchSelectedValue()); }"
    )


# -------------------------------------------------------
# Mount a custom FastAPI route /yolo_predict on the Gradio server
# so the browser JS can POST base64 frames and get YOLO results
# -------------------------------------------------------
from fastapi import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse

async def yolo_predict_route(request: _Request):
    try:
        body = await request.json()
        frame_b64 = body.get("frame_b64", "")
        model_id = body.get("model_id", "yolo11n.pt")
        result_json = predict_frame_base64(frame_b64, model_id)
        import json as _json
        return _JSONResponse(content=_json.loads(result_json))
    except Exception as e:
        print(f"[DensiTrack] /yolo_predict failed: {e}")
        return _JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=resolve_gradio_port(),
        css=VISUVATE_CSS,
        head=HEAD_JS,
        prevent_thread_lock=True,
    )
    # Gradio 6 rebuilds demo.app inside launch(), so custom routes registered
    # before launch disappear. Attach the frame detector to the serving app.
    demo.app.add_api_route("/yolo_predict", yolo_predict_route, methods=["POST"])
    demo.block_thread()
