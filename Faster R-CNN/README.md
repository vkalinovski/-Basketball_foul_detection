# 🏀 Faster R-CNN (ResNet-50 + FPN) — Basketball Foul Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)
![TorchMetrics](https://img.shields.io/badge/TorchMetrics-0.11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Kaggle API](https://img.shields.io/badge/Kaggle-API-orange.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📖 Overview
This repository contains a notebook that trains a **single-frame** basketball-foul detector.  
Based on **Faster R-CNN** with **ResNet-50 + Feature Pyramid Network**, pretrained on COCO and fine-tuned on Label Studio–annotated frames.

---

## 🎯 TL;DR – Current Results  
| Metric (val)   | Value *   |
|----------------|-----------|
| **mAP @ 0.50** | **0.0105**|
| AR @ 10        | 0.0749    |

\* Results after 30 epochs on an A100 40 GB GPU.

---
## 📁 Labeling & Dataset Upload
- **Label Studio** ([labelstud.io](https://labelstud.io/)) with the **Video Object Tracking** plugin was used for manually annotating short clips: each segment was labeled as `foul` or `no_foul` to form precise ground truth.  
- After labeling, JSON files were exported, organized by video and frame, and the final dataset was uploaded to **Kaggle** ([Foul Detection (test)](https://www.kaggle.com/datasets/sesmlhs/foul-detection-test/data?select=M16.mp4)), enabling easy data loading and scalable training.

---

## 📦 Dataset

| Source                           | Size  | Link                                                                                                                   |
|----------------------------------|------:|------------------------------------------------------------------------------------------------------------------------|
| Kaggle: **Foul Detection (test)**| 1.3 GB| [https://www.kaggle.com/datasets/sesmlhs/foul-detection-test](https://www.kaggle.com/datasets/sesmlhs/foul-detection-test) |

---

## ⚙️ Notebook Breakdown

| Block     | Core Logic                                                                                             | Why / Notes                                 |
|-----------|--------------------------------------------------------------------------------------------------------|---------------------------------------------|
| **BLK-1** | Install dependencies, mount Drive, `kaggle datasets download`                                          | Checkpoints stored on Drive                 |
| **BLK-2** | Read JSON → `{frame : bbox}`                                                                           | Handles missing entries and empty bboxes    |
| **BLK-3** | Stratified 80/20 split, incremental-training logic                                                     | `seen_videos.txt` — prevent reusing old data|
| **BLK-4** | `FrameDataset`: take the **center frame**, resize to 512, normalize, augment (ColorJitter, flip, blur, rotate) + safe wrapper | Center frame ≈ likely foul moment  |
| **BLK-5** | Create Faster R-CNN ResNet-50-FPN, replace head → 2 classes                                            | “background” / “foul”                       |
| **BLK-6** | EMA update, metrics, learning-rate schedulers                                                          | —                                           |
| **BLK-7** | Training: **AdamW + OneCycle → CosineWarmRestarts → SWA** (+ EMA, AMP). Checkpoint on mAP↑, save SWA    | Stable DataLoader (`num_workers=0`)         |
| **BLK-8** | End-to-end validation: mAP @[0.50–0.95], @0.50, @0.75, AR @ 1/10/100                                 | Tabular output                              |

---
## 🧠 Why This Model?

| Component                  | Reason                              |
|----------------------------|-------------------------------------|
| **Faster R-CNN**           | Reliable accuracy/speed balance; easy to fine-tune |
| **ResNet-50**              | Sufficient depth with moderate FLOPs |
| **FPN**                    | Handles multi-scale: fouls can be large or small |
| **AdamW + OneCycle**       | Fast warmup and smooth LR decay    |
| **SWA + EMA**              | Averaging reduces noise in later steps → +mAP |
| **AMP**                    | ~×1.5 speedup, –40 % VRAM usage     |
| **WeightedRandomSampler**  | Compensates for class imbalance (“background” >> “foul”) |

---

## 🛠️ Hyper-Parameters

| Parameter            |   Value | Description                |
|----------------------|--------:|----------------------------|
| `EPOCHS`             |      30 | Full training run          |
| `PHASE1` / `PHASE2`  |     10 / 20 | OneCycle → Cosine       |
| `swa_start`          |      25 | SWA start epoch            |
| `batch_size`         |       4 | For 12 GB VRAM             |
| `IMG_SIZE`           | 512 → 384 → 512 | Progressive resizing |
| `lr`                 | 1 e−3   | Base learning rate         |
| `weight_decay`       | 1 e−4   | Regularization             |
| `EMA_DECAY`          | 0.9999  | EMA inertia                |
| `num_workers`        |   0 → 2 | Stability → speed          |

---
## 🗂️ Artifacts

| File                              | Contents                                        |
|-----------------------------------|-------------------------------------------------|
| `checkpoints/best_detector.pth`   | Weights with best mAP@0.50 (saved every 5 epochs) |
| `checkpoints/swa_final.pth`       | Final SWA-averaged model                         |
| `train_videos.txt` / `val_videos.txt` | Lists of videos for reproducibility         |
| `seen_videos.txt`                 | Videos already used in previous runs             |
