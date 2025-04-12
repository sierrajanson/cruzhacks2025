# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for local testing

# Load your YOLO model once at startup.
model = YOLO("yolov8n-seg.pt")

def detect_objects(frame: np.ndarray):
    results = model(frame)[0]
    boxes = []
    # Use the results.boxes.xyxy (if available) to extract bounding boxes.
    if results.boxes is not None:
        # results.boxes.xyxy is a tensor with shape (N, 4)
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            # Convert to [x, y, width, height] format.
            boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return boxes

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Expect the image as a base64 string (e.g. "data:image/jpeg;base64,...")
    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': f'Invalid image encoding: {e}'}), 400
    # Convert bytes to a NumPy array.
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400
    
    boxes = detect_objects(frame)
    return jsonify({'boxes': boxes})

if __name__ == '__main__':
    # Run on localhost:5000 for example.
    app.run(host='0.0.0.0', port=5000)
