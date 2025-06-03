# 🦆 YOLOv12n — High-Speed Video Object Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv12n-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📖 Overview

This repository contains **YOLOv12n**, an ultra-light single-frame detector optimized for real-time video. It is based on **Ultralytics YOLOv12** with pretrained weights, further fine-tuned on your custom dataset.

---

## 🎯 TL;DR — Results (Validation Set)

| Metric        | Value     |
|---------------|----------:|
| **mAP @50**   | `0.XX`    |
| **mAP @50-95**| `0.XX`    |
| **FPS (640²)**| ~`YY` fps |

<sub>*Values obtained with `epochs=150`, `batch=16`, `imgsz=640` on an NVIDIA RTX 3090 GPU.*</sub>

---

## 📂 Project Structure

```text
├── train-yolo12.ipynb            # Jupyter notebook for training YOLOv12n
├── data/
│   ├── data.yaml                 # Dataset config (Ultralytics format)
│   ├── train/
│   │   ├── images/               # Training images
│   │   └── labels/               # YOLO-formatted labels (*.txt)
│   ├── val/
│   │   ├── images/               # Validation images
│   │   └── labels/               # YOLO-formatted labels (*.txt)
│   └── test/
│       └── images/               # Images for test inference
├── weights/
│   └── yolov12n.pt               # Pretrained or custom weights
├── runs/
│   └── detect/
│       ├── train/                # Training logs: weights, metrics, plots
│       └── val/                  # Validation results: metrics, examples
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## ⚙️ Training — `train-yolo12.ipynb`

### Installing Dependencies

```bash
pip install -U ultralytics
pip install -r requirements.txt
```

### Preparing the Data

- `train/` — images and labels for training  
- `val/` — images and labels for validation  
- `test/` — images for test inference

### Initializing the Model

```python
from ultralytics import YOLO

model = YOLO('weights/yolov12n.pt')
```

### Running Training

```python
model.train(
    data='data/data.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    workers=8,
    project='runs/detect',
    name='yolov12n_train'
)
```

### Validation and Metrics

```python
metrics = model.val(data='data/data.yaml')
print(f"mAP50-95: {metrics.box.map:.3f}, mAP50: {metrics.box.map50:.3f}")
```

### Inference on Images

```python
results = model.predict(
    source='data/test/images',
    imgsz=640,
    conf=0.25,
    save=True
)
```
