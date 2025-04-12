import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch

model = YOLO("yolov8n-seg.pt") 

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").eval()

WALL_ID = 0     # wall

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
count = 0
save_wall_mask = None

# camera run loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # wall/floor identification
    annotated_frame = frame.copy()

    if count % 10 == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (256, 256))
        inputs = feature_extractor(images=frame_small, return_tensors="pt")
        with torch.no_grad():
            outputs = segformer(**inputs)
        logits = outputs.logits  # shape: [1, num_classes, H, W]
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy()  # [H, W]
        preds = cv2.resize(preds, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # wall/floor masks
        wall_mask = (preds == WALL_ID).astype(np.uint8)

        # overlay masks
        wall_overlay = np.zeros_like(frame)
        wall_overlay[:, :, 2] = wall_mask * 200    # red for wall
        save_wall_mask = wall_mask
    results = model(frame)[0]

    if results.masks is not None:
        masks = results.masks.data  # (N, H, W) binary masks
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)  # convert to numpy
            color = [254,0,0]#np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask = np.stack([mask * c for c in color], axis=-1)
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
        masks = results.masks.data.cpu().numpy()  # shape: (N, H, W), all binary (0 or 1)
        combined_mask = np.max(masks, axis=0).astype(np.uint8)  # shape (H, W)
        ground_mask = 1 - combined_mask - save_wall_mask
        ground_color = np.array([50, 200, 50], dtype=np.uint8)  # light green
        ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)
        for result in results:
            xy = result.masks.xy  # mask in polygon format
            masks = result.masks.data  # mask in matrix format (num_objects x H x W)
            # print("MASKS", masks)

    frame_placeholder.image(annotated_frame, channels="BGR")
    count += 1
cap.release()
cv2.destroyAllWindows()
