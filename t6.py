import cv2
import torch
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# =============================
# CONFIG
# =============================
INPUT_PATH = "loneebird.mp4"
OUTPUT_PATH = "output_far_hybrid.mp4"

INFER_WIDTH = 960
INFER_HEIGHT = 540

CONF_THRESHOLD = 0.1
FULL_SWEEP_INTERVAL = 10  # run full-frame ML every N frames
MAX_ROIS = 3              # max ROI-based inference per frame

# =============================
# CHECK CUDA
# =============================
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# =============================
# LOAD MODEL
# =============================
model = RFDETRNano()
model.optimize_for_inference()

# =============================
# VIDEO SETUP
# =============================
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

# Background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

frame_index = 0

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    # -------------------------
    # Motion detection
    # -------------------------
    fgMask = backSub.apply(frame)
    _, thresh = cv2.threshold(fgMask.copy(), 200, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        if cv2.contourArea(contour) < 150:  # filter noise
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rois.append((x, y, w, h))

    # -------------------------
    # Decide inference mode
    # -------------------------
    # ROI mode if 1-3 motion blobs, else full-frame
    if len(rois) == 0 or len(rois) > MAX_ROIS or frame_index % FULL_SWEEP_INTERVAL == 0:
        run_full_frame = True
        mode_reason = "full-frame fallback"
    else:
        run_full_frame = False
        mode_reason = f"ROI-based, {len(rois)} blobs"

    boxes_to_draw = []

    if run_full_frame:
        # Full-frame inference
        resized = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            detections = model.predict(rgb, threshold=CONF_THRESHOLD)

        scale_x = orig_w / INFER_WIDTH
        scale_y = orig_h / INFER_HEIGHT

        for box in detections.xyxy:
            x1, y1, x2, y2 = box.astype(int)
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            boxes_to_draw.append((x1, y1, x2, y2))

    else:
        # ROI-based inference
        for (x, y, w, h) in rois[:MAX_ROIS]:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            # Resize ROI for speed
            scale = 320 / max(w, h)
            small_w, small_h = int(w*scale), int(h*scale)
            roi_small = cv2.resize(roi, (small_w, small_h))
            rgb_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                det = model.predict(rgb_roi, threshold=CONF_THRESHOLD)

            # Map back to original frame coordinates
            for box in det.xyxy:
                bx1, by1, bx2, by2 = box.astype(int)
                bx1 = int(bx1 / scale) + x
                by1 = int(by1 / scale) + y
                bx2 = int(bx2 / scale) + x
                by2 = int(by2 / scale) + y
                boxes_to_draw.append((bx1, by1, bx2, by2))

    # -------------------------
    # Draw boxes
    # -------------------------
    for (x1, y1, x2, y2) in boxes_to_draw:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

    # Log frame info
    if frame_index % 30 == 0:
        print(f"Frame {frame_index}: Mode={mode_reason}, ROIs={len(rois)}, Boxes={len(boxes_to_draw)}")

cap.release()
out.release()
print("Done. Saved to:", OUTPUT_PATH)