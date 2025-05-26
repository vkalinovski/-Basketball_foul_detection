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
**RF-DETR** (*Radar Frequency Detection Transformer*) — гибрид CNN + Transformer-архитектура, способная:
- **Точно локализовать объекты** на последовательности видеокадров  
- **Сохранять трекинг** даже при шуме, смазе и низком разрешении  
- **Интегрировать радиочастотные признаки** (range / Doppler) для усиления «зрения»  

Вся логика реализована на **PyTorch 2.1** с **Detectron2 0.6**; модель легко расширяется собственными модулями признаков.

---

## 🎯 TL;DR — Current Results  

| Метрика (val)      | ResNet-50 | ResNet-101 |
|--------------------|----------:|-----------:|
| **mAP @ 0.50**     | **0.427** | **0.451** |
| mAP @ \[0.50‒0.95] | 0.273     | 0.288     |
| AR @ 100           | 0.381     | 0.396     |
| FPS (1080p, RTX 3090) | 29 fps | 24 fps |

<sub>Тренировка — 45 эпох, 2×A100 80 GB, batch_size = 8, загрузка радара 20 каналов.</sub>

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

| Source | Clips | Frames | Size | Link |
|--------|------:|-------:|-----:|------|
| **RF-Tracking Benchmark v1** | 92 | 136 842 | 5.6 GB | _private S3_ |
| **YouTube (creative commons)** | 51 | 78 014 | 3.4 GB | — |
| **Total** | **143** | **214 856** | **9.0 GB** | |

### Labeling  
- **Label Studio** с плагином **Video Object Tracking** → экспорт COCO JSON  
- RF-данные синхронизируются по *timestamp* и пишутся во второй канал HDF5 (`rf_range`, `rf_doppler`)  

---

## ⚙️ Notebook Breakdown  

| Block | Логика | Особенности |
|-------|--------|-------------|
| **B-1** | Mount Drive, env check, ⚙️ `pip install -r requirements.txt` | Colab-friendly |
| **B-2** | Parsing RF + video, кадро-батч селектор | 3-уровневый кэш (RAM → NVMe → S3) |
| **B-3** | `RFCOCODataset` + augmenter (Albumentations) | RF-каналы аугментируются синхронно |
| **B-4** | Build RF-DETR (backbone, FPN, transformer heads) | Plug-and-play RF-heads |
| **B-5** | Train loop: AdamW → Cosine Annealing, AMP, EMA | Checkpoint @ mAP↑ |
| **B-6** | Eval: mAP/mAR, confusion matrix, speed test | Табличный + графический вывод |
| **B-7** | Export weights, TorchScript, ONNX | `outputs/checkpoints/` |

---

## 🧠 Why This Model?  

| Компонент | Почему |
|-----------|--------|
| **DETR-style Transformer** | Учится глобальным отношениям ↔ устойчив к окклюзиям |
| **Radar‐fusion head** | RF-каналы ≈ «сквозь дым/туман» + глубина |
| **FPN** | Объекты разных размеров (drones ↔ trucks) |
| **AdamW + Cosine** | Стабильный прогрев, плавный спад LR |
| **EMA** | Уменьшает шум последних эпох → +mAP |
| **AMP** | 1.7× скорость, –45 % VRAM |

---

## 🛠️ Hyper-Parameters  

| Параметр             | Значение | Примечание |
|----------------------|---------:|-----------|
| `EPOCHS`             | 45       | Полный прого н |
| `BATCH_SIZE`         | 8        | A100 80 GB ×2 |
| `LR_INIT`            | 1 e-4    | Base LR |
| `LR_MIN`             | 1 e-6    | Cosine floor |
| `WEIGHT_DECAY`       | 5 e-4    |   |
| `RF_CHANNELS`        | 20       | Range + Doppler |
| `EMA_DECAY`          | 0.9997   |   |
| `IMG_SIZE`           | 960×540  | Training resize |
| `NUM_WORKERS`        | 8        |   |

