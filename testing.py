import cv2
import torch
import numpy as np
from rfdetr import RFDETRBase
from scipy.io import loadmat

def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def video_detection_with_eval(
    input_path,
    output_path,
    mat_label_path,
    infer_width=960,
    infer_height=540,
    conf_threshold=0.2,
    full_sweep_interval=10,
    max_rois=3,
    max_contour_area=30,
):
    # Load MATLAB ground truth
    mat_data = loadmat(mat_label_path)
    label_data = mat_data['gTruth']['LabelData'][0,0]  # Access the LabelData table
    classes = ['AIRPLANE','BIRD','DRONE','HELICOPTER']
    
    # Convert MATLAB bounding boxes into a Python dict {frame_index: {class: [boxes]}}
    gt_dict = {}
    num_frames = label_data[classes[0]].shape[0]
    for frame_idx in range(num_frames):
        frame_boxes = {}
        for cls in classes:
            boxes_cell = label_data[cls][frame_idx,0]  # MATLAB cell array
            if boxes_cell.size == 0:
                frame_boxes[cls] = []
            else:
                # Convert MATLAB [x,y,w,h] to [x1,y1,x2,y2]
                boxes_list = []
                for box in boxes_cell:
                    x, y, w, h = box[0]
                    boxes_list.append([int(x), int(y), int(x+w), int(y+h)])
                frame_boxes[cls] = boxes_list
        gt_dict[frame_idx] = frame_boxes

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    model = RFDETRBase().to(device)
    model.optimize_for_inference()

    # Video setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_size = (orig_w, orig_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    frame_index = 0
    iou_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgMask = backSub.apply(frame)
        _, thresh = cv2.threshold(fgMask.copy(), 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= max_contour_area]

        frame_index += 1
        # Determine full-frame vs ROI
        if len(rois)==0 or len(rois)>max_rois or frame_index % full_sweep_interval==0:
            run_full_frame = True
        else:
            run_full_frame = False

        boxes_to_draw = []

        if run_full_frame:
            resized = cv2.resize(frame, (infer_width, infer_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                dets = model.predict(rgb, threshold=conf_threshold)
            scale_x = orig_w / infer_width
            scale_y = orig_h / infer_height
            for box in dets.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                boxes_to_draw.append([int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)])
        else:
            for (x, y, w, h) in rois[:max_rois]:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0: continue
                scale = 320 / max(w,h)
                roi_small = cv2.resize(roi, (int(w*scale), int(h*scale)))
                rgb_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    det = model.predict(rgb_roi, threshold=conf_threshold)
                for box in det.xyxy:
                    bx1, by1, bx2, by2 = box.astype(int)
                    bx1 = int(bx1/scale)+x
                    by1 = int(by1/scale)+y
                    bx2 = int(bx2/scale)+x
                    by2 = int(by2/scale)+y
                    boxes_to_draw.append([bx1, by1, bx2, by2])

        # Draw predicted boxes
        for b in boxes_to_draw:
            cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)

        # Draw ground truth boxes
        if (frame_index-1) in gt_dict:
            gt_boxes_frame = []
            for cls in classes:
                gt_boxes_frame += gt_dict[frame_index-1][cls]
            for b in gt_boxes_frame:
                cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,0,255), 2)
            # Compute IoU for this frame (take best match for each GT)
            for gt_box in gt_boxes_frame:
                best_iou = 0
                for pred_box in boxes_to_draw:
                    iou = compute_iou(gt_box, pred_box)
                    best_iou = max(best_iou, iou)
                iou_list.append(best_iou)

        out.write(frame)
        if frame_index % 30 == 0:
            print(f"Frame {frame_index}: Predicted {len(boxes_to_draw)}, GT {len(gt_boxes_frame)}")

    cap.release()
    out.release()
    print("Done. Saved to:", output_path)
    print(f"Average IoU over frames: {np.mean(iou_list):.3f}")

# =====================
if __name__ == "__main__":
    video_detection_with_eval(
        input_path="input_video.mp4",
        output_path="output_video.mp4",
        mat_label_path="IR_AIRPLANE_001_LABELS.mat",
        infer_width=960,
        infer_height=540,
        conf_threshold=0.2,
        full_sweep_interval=10,
        max_rois=3,
        max_contour_area=30,
    )