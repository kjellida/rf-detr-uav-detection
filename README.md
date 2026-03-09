![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
# Adaptive Inference for UAV Object Detection using RF-DETR

This project implements adaptive inference for detection of UAVs using RF-DETR.
The goal is to reduce computational cost while maintaining detection accuracy by dynamically selecting the region of the image used for inference.

### 1. Motion-Based ROI Inference

Instead of running the detector on the entire frame, inference is performed only on Regions of Interest (ROI) determined by motion.

### 2. Full Frame Inference

In this mode, RF-DETR processes the entire frame without ROI filtering.
