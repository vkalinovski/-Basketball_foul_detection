---
<div align="center">
  <img src="images/image.png" alt="Pipeline Overview" width="600" />
</div>

# 🏀 Basketball-Foul-Detection

---

An automated pipeline for detecting fouls in basketball games.  
The main model is **RF-DETR**; **Faster R-CNN** and **YOLOv12** were considered as baseline models but showed inferior results and are included here for completeness.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.2-blue?logo=pytorch" />
  <img src="https://img.shields.io/badge/Lightning-2.2.1-blueviolet?logo=lightning" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📚 Table of Contents
1. [Repository Structure](#-repository-structure)
2. [Datasets](#-datasets)
3. [Validation Pipeline](#-validation-pipeline)

---

## 📂 Repository Structure
```text
📦 project-root
├── Faster R-CNN/        # experiments with Faster R-CNN
├── RF_DETR/             # main model and inference scripts
├── YOLOv12/             # experiments with YOLOv12
├── data_collection/     # collection of cleaned videos
├── dataset/
│   └── main/            # train / valid / test frames
├── ref_validation/      # LLM vs. RF-DETR comparison
├── weights/             # model checkpoints
└── README.md            # you are reading this 😊
```
---

## 📊 Datasets

| Purpose                      | Format            | Size        | Link                                                                                            |
|------------------------------|-------------------|-------------|-------------------------------------------------------------------------------------------------|
| Frame Classification         | `.jpg` / `.json`  | 9,162 frames| [Kaggle](https://www.kaggle.com/datasets/vladimirkalinovski/nba-foul-detection)                 |
| Short Video Clips            | `.mp4`            | 3 h         | *in the process of publication*                                                                 |
| Unannotated Full-Game Videos | `.mkv` / `.mp4`   | 25 GB       | Google Drive (access upon request)                                                              |
 | COCO dataset                 | `.jpg` with annotations| 5700 frames| [COCO](https://github.com/vkalinovski/-Basketball_foul_detection/tree/main/dataset/main)        |
| YOLO dataset                 | `.jpg` with annotations| 5700 frames| [YOLO](https://github.com/vkalinovski/-Basketball_foul_detection/tree/main/dataset/mainYolov12) |





---

## 🔄 Validation Pipeline

> The full pipeline compares **LLM** and **RF-DETR** on the same video segments.

1. **Video Loading** → decoding + audio extraction (OpenCV / FFmpeg).  
2. **ASR** → Whisper generates a transcript with timestamps.  
3. **LLM** → prompt ⟶ *`foul`* / *`no_foul`* + `confidence`.  
4. **RF-DETR** → frame-by-frame detection + foul heuristics → verdict.  
5. **Comparison** → Accuracy, Precision, Recall, F1, Agreement Rate.  






## 📁 Model Validation

| Stage                | Tool                                               |
|----------------------|----------------------------------------------------|
| ⚙️ Model Inference   | `RF_DETR/video_annotation.ipynb`                   |
| 📖 LLM Inference     | `ref_validation/ref_validation.ipynb`              |
| 🏁 Metric Calculation | `ref_validation/ref_validation.ipynb` (section *Evaluation*) |

Output saved to `ref_validation/results.csv`.

---



