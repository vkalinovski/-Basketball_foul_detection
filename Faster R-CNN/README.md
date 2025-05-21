# ⚽️ Faster R-CNN (ResNet-50 + FPN) — Soccer Foul Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)
![TorchMetrics](https://img.shields.io/badge/TorchMetrics-0.11-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow.svg)
![Kaggle API](https://img.shields.io/badge/Kaggle-API-orange.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-brightgreen.svg)
![Licence](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

<details>
<summary><strong>Table&nbsp;of&nbsp;Contents</strong> 📑</summary>

- [Overview](#-overview)
- [Quick start 🛫](#-quick-start)
- [Installation & Setup 💻](#-installation--setup)
- [Dataset 📦](#-dataset)
- [Notebook Blocks ⚙️](#-notebook-blocks)
- [Model & Training Details 🧠](#-model--training-details)
- [Hyper-parameters ⚙️](#-hyper-parameters)
- [Artifacts 🗂️](#-artifacts)
- [Roadmap 🚧](#-roadmap)
- [License 📝](#-license)
</details>

---

## 📖 Overview
Detects fouls on **single central frames** of football (soccer) videos.  
Architecture — **Faster R-CNN** with **ResNet-50 + Feature Pyramid Network** (pre-trained on COCO, fine-tuned on our data from Label Studio).

---

## 🚀 Quick start

|                          | Colab 1-Click |
|--------------------------|---------------|
| Launch notebook in Colab | <a href="https://colab.research.google.com/github/your-repo/foul-detector/blob/main/MAIN.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

---

## 🔧 Installation & Setup 💻

> **TL;DR** — one copy-paste and you’re up & running:

```bash
pip install -q kaggle torch torchvision torchmetrics opencv-python tqdm scikit-learn

