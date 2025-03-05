# YOLOv8 Object Detection (CPU + ONNX)

This project implements real-time object detection using the YOLOv8 model with options to run on CPU and export the model to ONNX format for optimized inference.

## Features

- **Real-time Object Detection**: Capture video from your webcam and detect objects using YOLOv8.
- **CPU Support**: Run the detection on CPU for compatibility with machines without a GPU.
- **ONNX Model Export & Inference**: Convert YOLOv8 to ONNX and run inference with ONNX Runtime for optimized performance.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-repo/yolo-cpu-onnx
cd yolo-cpu-onnx
```

2. Install dependencies:

```bash
pip install ultralytics opencv-python onnxruntime numpy
```

3. Download the YOLOv8 model weights:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

### Real-Time Object Detection (CPU)

Run the following script to start webcam-based object detection:

```bash
python detect.py
```

Press **'q'** to exit the webcam view.

### Export to ONNX

Convert the YOLOv8 model to ONNX format for efficient inference:

```bash
python export.py
```

This creates a `yolov8n.onnx` file.

### Inference with ONNX

Run object detection using the exported ONNX model:

```bash
python onnx_inference.py
```

## Code Structure

- `detect.py`: Real-time detection via webcam.
- `export.py`: Exports YOLOv8 model to ONNX format.
- `onnx_inference.py`: Runs object detection using the ONNX model.

## Performance Tips

- **Resize Frames**: Reducing frame size (e.g., 640x480) speeds up inference.
- **Confidence Threshold**: Adjust `conf` to control detection sensitivity.
- **Batch Inference**: For videos, consider batch processing frames to improve throughput.

## Example Output

<img src="example_output.png" alt="YOLOv8 Detection" width="600"/>

## Conclusion

This project makes object detection accessible even on devices without GPUs. You can easily extend the functionality or deploy it in lightweight environments using ONNX runtime.

Feel free to contribute or reach out for improvements! 🚀

---
SAMIYA SHAHZAD 
**LinkedIn:** (https://www.linkedin.com/in/samiya-shahzad-148a8529b/)


