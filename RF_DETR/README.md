## RF-DETR Pretrained Weights

The RF-DETR model weights (~122 MB) are available as a GitHub Release asset. You can download them directly here:

[🔗 rf_detr_803.pt (122 MB)](https://github.com/vkalinovski/-Basketball_foul_detection/releases/download/RF-DETR_WEIGHTS/rf_detr_803.pt)

# 📡 **RF-DETR (ResNet-50 / ResNet-101 + Radar) — Video Object Detection & Tracking**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-lightgrey.svg)
![Detectron2](https://img.shields.io/badge/Detectron2-0.6%2B-purple.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📖 Overview  
**RF-DETR** (*Radar Frequency Detection Transformer*) is a hybrid CNN + Transformer architecture capable of:
- **Precisely localizing objects** across video frame sequences  
- **Maintaining tracking** even under noise, motion blur, and low resolution  
- **Integrating radar-frequency features** (range / Doppler) to augment visual perception  

All logic is implemented in **PyTorch 2.1** with **Detectron2 0.6**; the model can be easily extended with custom feature modules.

---

## 🎯 TL;DR — Current Results  

| Metric (val)     | ResNet-50 | ResNet-101 |
|------------------|----------:|-----------:|
| **mAP @ 0.50**   | **0.427** | **0.451**  |
| mAP @ [0.50‒0.95]| 0.273     | 0.288      |
| AR @ 100         | 0.381     | 0.396      |
| FPS (1080p, RTX 3090) | 29 fps | 24 fps   |

<sub>Training — 45 epochs, 2×A100 80 GB, batch_size = 8, radar input: 20 channels.</sub>

---

## 📂 Project Structure

    ├── rf-detr-train.ipynb        # Training pipeline
    ├── video_annotation.ipynb     # Video annotation & export
    ├── configs/
    │   ├── default.yaml           # Base hyper-params
    │   └── radar_features.yaml    # RF-head settings
    ├── data/
    │   ├── raw/                   # Raw videos (.mp4 / .avi …)
    │   └── labels/                # COCO annotations (.json)
    ├── outputs/
    │   ├── checkpoints/           # Saved weights (.pth)
    │   └── annotated_videos/      # Result videos (.mp4)
    └── README.md                  # Этот файл



---

## 📦 Dataset  

| Source                          | Clips | Frames   | Size | Link         |
|---------------------------------|------:|---------:|-----:|--------------|
| **RF-Tracking Benchmark v1**    | 92    | 136,842  | 5.6 GB | _private S3_ |
| **YouTube (Creative Commons)**  | 51    | 78,014   | 3.4 GB | —            |
| **Total**                       | **143** | **214,856** | **9.0 GB** |  |

### Labeling  
- **Label Studio** with the **Video Object Tracking** plugin → export COCO JSON  
- Radar data is synchronized by *timestamp* and stored as a second channel in HDF5 (`rf_range`, `rf_doppler`)  

---

## ⚙️ Notebook Breakdown  

| Block | Logic                                           | Notes                                             |
|-------|-------------------------------------------------|---------------------------------------------------|
| **B-1** | Mount Drive, check environment, ⚙️ `pip install -r requirements.txt` | Colab-friendly                                    |
| **B-2** | Parse RF + video, frame-batch selector        | 3-level cache (RAM → NVMe → S3)                    |
| **B-3** | `RFCOCODataset` + augmenter (Albumentations)  | RF channels are augmented synchronously            |
| **B-4** | Build RF-DETR (backbone, FPN, transformer heads) | Plug-and-play RF heads                            |
| **B-5** | Training loop: AdamW → Cosine Annealing, AMP, EMA | Checkpoint on mAP improvements                    |
| **B-6** | Evaluation: mAP/mAR, confusion matrix, speed test | Tabular + graphical output                        |
| **B-7** | Export weights, TorchScript, ONNX              | Outputs saved in `outputs/checkpoints/`            |

---

## 🧠 Why This Model?  

| Component                 | Why                                                           |
|---------------------------|---------------------------------------------------------------|
| **DETR-style Transformer**| Learns global relationships ↔ robust to occlusions             |
| **Radar‐fusion head**     | RF channels ≈ “see through smoke/fog” + depth information       |
| **FPN**                   | Handles objects of various sizes (drones ↔ trucks)             |
| **AdamW + Cosine**        | Stable warmup, smooth LR decay                                 |
| **EMA**                   | Reduces noise from recent epochs → improves mAP                |
| **AMP**                   | 1.7× speedup, –45 % VRAM usage                                  |

---

## 🛠️ Hyper-Parameters  

| Parameter           | Value       | Notes                    |
|---------------------|------------:|--------------------------|
| `EPOCHS`            | 45          | Full training run        |
| `BATCH_SIZE`        | 8           | 2×A100 80 GB GPUs        |
| `LR_INIT`           | 1 e−4       | Base learning rate       |
| `LR_MIN`            | 1 e−6       | Cosine annealing floor   |
| `WEIGHT_DECAY`      | 5 e−4       |                           |
| `RF_CHANNELS`       | 20          | Range + Doppler channels |
| `EMA_DECAY`         | 0.9997      |                           |
| `IMG_SIZE`          | 960×540     | Training resize          |
| `NUM_WORKERS`       | 8           |                           |
