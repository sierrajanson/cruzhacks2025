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

model = YOLO("yolov8n-seg.pt") 

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
# count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    results = model(frame)[0]

    annotated_frame = frame.copy()
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
        ground_mask = 1 - combined_mask
        ground_color = np.array([50, 200, 50], dtype=np.uint8)  # light green
        ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
        # Resize ground_overlay to match annotated frame
        ground_overlay = cv2.resize(ground_overlay, (annotated_frame.shape[1], annotated_frame.shape[0]))
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)
        # print(results.masks)
        # for result in results:
        #     xy = result.masks.xy  # mask in polygon format

        #     masks = result.masks.data  # mask in matrix format (num_objects x H x W)
        #     print("MASKS", masks)

        # Add safety margin (Might need to remove after wall detection)
        kernel = np.ones((3,3), np.uint8)
        blocked = cv2.dilate(combined_mask, kernel, iterations=1)
        ground_mask = 1 - blocked

        # Path finding
        small_ground = cv2.resize(ground_mask, (64, 64), interpolation=cv2.INTER_NEAREST) # reduce size
        grid = (small_ground == 1).astype(np.uint8)
        # Set start and end points (e.g., top-left to bottom-right)
        start = (0, 30)
        end = (63, 30)
        path = astar(grid, start, end) #start and end of path based on smaller size
        if path is not None:
            # Scale back to original size
            scale_x = frame.shape[1] / grid.shape[1] # Find scaling factor
            scale_y = frame.shape[0] / grid.shape[0] # Find scaling factor
            # Draw lines between scaled points in the path
            for i in range(1, len(path)):
                x1, y1 = path[i - 1]
                x2, y2 = path[i]
                pt1 = (int(y1 * scale_x), int(x1 * scale_y))  # (col, row) -> (x, y)
                pt2 = (int(y2 * scale_x), int(x2 * scale_y))
                cv2.line(annotated_frame, pt1, pt2, (255,105,180), thickness=15)
        else:
            st.write("Failed to find path")



    # annotated_frame = results.plot(labels=False, boxes=False)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_placeholder.image(annotated_frame, channels="BGR")
    # count += 1
cap.release()
cv2.destroyAllWindows()
