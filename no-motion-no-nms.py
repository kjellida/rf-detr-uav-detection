import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from src.byte_tracker import BYTETracker  # Make sure PYTHONPATH includes this


def video_detection(
   input_path,
   output_path,
   conf_threshold=0.3,
   #nms_threshold=0.5,
   display_padding=20,
   skip_initial_frames=5,  # ignore first 5 frames in FPS avg
):
   # --- CUDA check ---
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))


   # --- Load detector ---
   model = RFDETRBase()
   model.optimize_for_inference()


   # --- Video setup ---
   cap = cv2.VideoCapture(input_path)
   if not cap.isOpened():
       raise RuntimeError("Cannot open input video")


   orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = cap.get(cv2.CAP_PROP_FPS) or 30
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))


   # --- Tracker setup ---
   tracker = BYTETracker(track_thresh=0.4, track_buffer=30, match_thresh=0.8, fuse_score=False, frame_rate=fps)

   frame_index = 0
   start_time = None  # start timing after skipped frames


   while True:
       ret, frame = cap.read()
       if not ret:
           break
       frame_index += 1
       frame_start = time.time()


       boxes, scores = [], []


       # --- Inference ---
       
       rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       with torch.no_grad():
            detections = model.predict(rgb, threshold=conf_threshold)
       for box in detections.xyxy:
               x1, y1, x2, y2 = box.astype(int)
               boxes.append([x1, y1, x2-x1, y2-y1])
               scores.append(1.0)
      

       # --- Tracker update ---
       if boxes: #dont know if correctly removed nms
           #indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
           #indices = [i[0] if isinstance(i,(list,tuple,np.ndarray)) else i for i in indices]
           #boxes = [boxes[i] for i in indices]
           #scores = [scores[i] for i in indices]
           dets_for_tracker = np.array([[x, y, x+w, y+h, s] for (x,y,w,h),s in zip(boxes,scores)], dtype=np.float32)
       else:
           dets_for_tracker = np.empty((0,5), dtype=np.float32)


       if dets_for_tracker.shape[0] > 0:
           online_targets = tracker.update(dets_for_tracker, [orig_h, orig_w], [orig_h, orig_w])
       else:
           online_targets = []


       # --- Draw tracked boxes ---
       for t in online_targets:
           x, y, w, h = t.tlwh
           x2, y2 = x + w, y + h
           x_disp = max(0, int(x) - display_padding)
           y_disp = max(0, int(y) - display_padding)
           x2_disp = min(orig_w, int(x2) + display_padding)
           y2_disp = min(orig_h, int(y2) + display_padding)
           cv2.rectangle(frame, (x_disp, y_disp), (x2_disp, y2_disp), (0, 255, 0), 2)
           cv2.putText(frame, f"ID:{t.track_id} Conf:{t.score:.2f}", (x_disp, y_disp-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


       out.write(frame)


       # --- FPS calculation ---
       frame_end = time.time()
       instant_fps = 1 / max(frame_end - frame_start, 1e-6)


       # Start counting average FPS after skipping first few frames
       if frame_index == skip_initial_frames:
           start_time = frame_end
           avg_fps = instant_fps
       elif frame_index > skip_initial_frames:
           total_elapsed = frame_end - start_time
           avg_fps = (frame_index - skip_initial_frames) / total_elapsed
       else:
           avg_fps = 0.0


       print(f"Frame {frame_index}: , Boxes={len(boxes)}, "
             f"Instant FPS={instant_fps:.2f}, Avg FPS={avg_fps:.2f}")


   cap.release()
   out.release()
   print("Done. Saved to:", output_path)