from ultralytics import YOLO

# Load the nano model (smallest and fastest)
model = YOLO("yolov8n.pt")
model = YOLO("yolov8n.pt").to("cpu")  # Force CPU mode
import cv2
from ultralytics import YOLO

# Load model and force CPU
model = YOLO("yolov8n.pt").to("cpu")

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for built-in camera
# In your video capture loop:
print("Starting webcam. Press 'q' to quit.")

ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))  # Smaller resolution = faster

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Run detection on CPU (disable tracking for simplicity)
    results = model(frame, verbose=False)  # verbose=False hides logs
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Show FPS (approximate)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    cv2.imshow("CPU Object Detection", annotated_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO

# Convert YOLOv8n to ONNX
model = YOLO("yolov8n.pt")
model.export(format="onnx")  # Creates 'yolov8n.onnx'
import cv2
import numpy as np
from onnxruntime import InferenceSession

# Load ONNX model
session = InferenceSession("yolov8n.onnx")

# Preprocess frame (resize, normalize, etc.)
def preprocess(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = frame.transpose(2, 0, 1)  # HWC to CHW
    frame = np.expand_dims(frame, 0).astype(np.float32) / 255.0
    return frame

# Run inference
frame = preprocess(frame)
outputs = session.run(None, {"images": frame})
results = model(frame, conf=0.4)  # Default is 0.25