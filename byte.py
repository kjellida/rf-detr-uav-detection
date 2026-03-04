import cv2
import torch
from rfdetr import RFDETRBase
from yolox.tracker.byte_tracker import BYTETracker  # from ByteTrack repo
import numpy as np

def video_detection_with_tracking(
    input_path,
    output_path,
    infer_width=560,
    infer_height=560,
    conf_threshold=0.2,
    full_sweep_interval=10,
    max_rois=3,
    max_contour_area=30,
    track_buffer=30,
    track_thresh=0.5
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    if device=="cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load model
    model = RFDETRBase()
    model.optimize_for_inference()
    model.to(device)
    model.eval()

    # Video setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    # Initialize ByteTrack
    tracker = BYTETracker(track_thresh=track_thresh, track_buffer=track_buffer)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        # Motion detection
        fgMask = backSub.apply(frame)
        _, thresh = cv2.threshold(fgMask.copy(), 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= max_contour_area]

        # Decide inference mode
        run_full_frame = len(rois)==0 or len(rois)>max_rois or frame_index%full_sweep_interval==0

        detections = []
        if run_full_frame:
            # Full-frame
            resized = cv2.resize(frame, (infer_width, infer_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                det = model.predict(rgb, threshold=conf_threshold)
            scale_x = orig_w / infer_width
            scale_y = orig_h / infer_height
            for i, box in enumerate(det.xyxy):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = int(x1*scale_x), int(y1*scale_y)
                x2, y2 = int(x2*scale_x), int(y2*scale_y)
                score = float(det.conf[i])
                detections.append([x1, y1, x2, y2, score])
        else:
            # ROI-based
            for (x, y, w, h) in rois[:max_rois]:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                scale = 320 / max(w, h)
                roi_small = cv2.resize(roi, (int(w*scale), int(h*scale)))
                rgb_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    det = model.predict(rgb_roi, threshold=conf_threshold)
                for i, box in enumerate(det.xyxy):
                    bx1, by1, bx2, by2 = box.astype(int)
                    bx1 = int(bx1/scale)+x
                    by1 = int(by1/scale)+y
                    bx2 = int(bx2/scale)+x
                    by2 = int(by2/scale)+y
                    score = float(det.conf[i])
                    detections.append([bx1, by1, bx2, by2, score])

        # -------------------------
        # Update tracker
        # -------------------------
        if len(detections)>0:
            dets = np.array(detections)  # shape [N,5]
        else:
            dets = np.empty((0,5))
        online_targets = tracker.update(dets, [orig_h, orig_w], [orig_h, orig_w])

        # Draw boxes with tracking IDs
        for t in online_targets:
            tlbr = t.tlbr  # top-left bottom-right
            track_id = t.track_id
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        out.write(frame)

        if frame_index % 30 == 0:
            print(f"Frame {frame_index}: Tracked objects={len(online_targets)}")

    cap.release()
    out.release()
    print("Done. Video saved to:", output_path)