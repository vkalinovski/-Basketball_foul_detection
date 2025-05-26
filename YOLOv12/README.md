# 🦆 YOLOv12n — Высокоскоростной Детектор Объектов Видео

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv12n-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📖 Overview

Этот репозиторий содержит **YOLOv12n** — ультра-лёгкий однокадровый детектор, оптимизированный для реального времени на видео. В основе — **Ultralytics YOLOv12** с предобученными весами, дообученными на вашем датасете.

---

## 🎯 TL;DR — Результаты (Validation Set)

| Метрика        | Значение   |
|----------------|-----------:|
| **mAP @50**    | `0.XX`     |
| **mAP @50-95** | `0.XX`     |
| **FPS (640²)** | ~`YY` fps  |

<sub>*Значения получены при `epochs=150`, `batch=16`, `imgsz=640`, GPU NVIDIA RTX 3090.*</sub>

---

## 📂 Project Structure

```text
├── train-yolo12.ipynb            # Jupyter-ноутбук для обучения YOLOv12n
├── data/
│   ├── data.yaml                 # Конфиг датасета (Ultralytics формат)
│   ├── train/
│   │   ├── images/               # Изображения для обучения
│   │   └── labels/               # Разметка YOLO (*.txt)
│   ├── val/
│   │   ├── images/               # Изображения для валидации
│   │   └── labels/               # Разметка YOLO (*.txt)
│   └── test/
│       └── images/               # Изображения для тестового инференса
├── weights/
│   └── yolov12n.pt               # Предобученные или ваши веса
├── runs/
│   └── detect/
│       ├── train/                # Логи обучения: веса, метрики, графики
│       └── val/                  # Результаты валидации: метрики, примеры
├── requirements.txt              # Зависимости проекта
└── README.md                     # Этот файл
```

## ⚙️ Training — `train-yolo12.ipynb`

### Установка зависимостей

```bash
pip install -U ultralytics
pip install -r requirements.txt
```

### Подготовка данных

Отредактируйте `data/data.yaml`, указав пути к папкам:
- `train/` — изображения и разметка для обучения  
- `val/` — изображения и разметка для валидации  
- `test/` — изображения для тестового инференса  

### Инициализация модели

```python
from ultralytics import YOLO

model = YOLO('weights/yolov12n.pt')
```

### Запуск обучения

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

### Валидация и метрики

```python
metrics = model.val(data='data/data.yaml')
print(f"mAP50-95: {metrics.box.map:.3f}, mAP50: {metrics.box.map50:.3f}")
```

### Инференс на изображениях

```python
results = model.predict(
    source='data/test/images',
    imgsz=640,
    conf=0.25,
    save=True
)
```
