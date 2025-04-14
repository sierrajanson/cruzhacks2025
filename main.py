import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import heapq

def neighbors(x, y, max_x, max_y):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < max_x and 0 <= ny < max_y:
            yield nx, ny

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, end), 0, start, [start]))
    visited = set()

    while open_set:
        est_total, cost, current, path = heapq.heappop(open_set)
        if current == end:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                heapq.heappush(open_set, (cost + 1 + heuristic((nx, ny), end), cost + 1, (nx, ny), path + [(nx, ny)]))
    return None

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
            # Resize colored mask to match annotated frame
            colored_mask = cv2.resize(colored_mask, (annotated_frame.shape[1], annotated_frame.shape[0]))
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)

        masks = results.masks.data.cpu().numpy()  # shape: (N, H, W), all binary (0 or 1)
        combined_mask = np.max(masks, axis=0).astype(np.uint8)  # shape (H, W)
        ground_mask = 1 - combined_mask - save_wall_mask
        ground_color = np.array([50, 200, 50], dtype=np.uint8)  # light green
        ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
        # Resize ground_overlay to match annotated frame
        ground_overlay = cv2.resize(ground_overlay, (annotated_frame.shape[1], annotated_frame.shape[0]))
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)
        for result in results:
            xy = result.masks.xy  # mask in polygon format
            masks = result.masks.data  # mask in matrix format (num_objects x H x W)
            # print("MASKS", masks)

    frame_placeholder.image(annotated_frame, channels="BGR")
    count += 1
cap.release()
cv2.destroyAllWindows()
