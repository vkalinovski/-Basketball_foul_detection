## 📂 Repository Structure

| Folder | Purpose |
| ------ | ------- |
| **[data_collection](https://github.com/vkalinovski/-Basketball_foul_detection/tree/main/data_collection)** | 🎞️ Scrape YouTube, find coarse foul intervals, split into clips, upload to cloud storage |
| **labeling** | 🖍️ Annotate clips in **Label Studio** (Video Object Tracking) → `foul` / `no_foul` |
| **dataset_creation** | 📦 Export JSON labels, tidy & push public dataset to **[Kaggle](https://www.kaggle.com/datasets/sesmlhs/foul-detection-test/data?select=M16.mp4)** |
| **model_training** | 🏋️ Fine‑tune **SlowFast R50** with PyTorch Lightning, heavy augments, Mixup & OneCycleLR |
| **model_tuning** | 🎛️ Hyper‑opt via **Optuna**, early stop, checkpoints, gradient clip, selective layer freeze |
| **inference** | 🚀 Single‑clip prediction script → `foul` / `not_foul` with softmax probability |
| **notebooks/** | 📒 Exploratory notebooks & TensorBoard logs |
| **utils/** | 🛠️ Helper scripts: Whisper transcription, clip extraction, dataset sync |

---

### 🔧 Model Training
- **Backbone:** SlowFast R50 from *pytorchvideo* (pre‑trained on Kinetics‑400).  
- **Augmentations:** temporal subsample, crop, flip, jitter, Gaussian noise.  
- **Regularization:** Mixup (α = 0.4), label smoothing (0.1), weight decay (1e‑4).  
- **Scheduler:** OneCycleLR, 8 epochs, automatic mixed precision *(AMP)*.  
- **Infra:** Google Colab GPU (Tesla T4/A100) → checkpoints & TB logs in Google Drive.

### 🛠️ Model Tuning
| Hyper‑param | Search space | Best |
| ----------- | ------------ | ---- |
| Learning rate | log‑uniform 1e‑5 → 1e‑3 | **3e‑4** |
| Dropout | 0 → 0.5 | **0.4** |
| Batch size | 4, 8, 16 | **8** |
| Freeze depth | none / last 1 / last 2 blocks | **none** |

- Optuna 📈 maximised *val_accuracy* with 50 trials.  
- **Early stopping** (patience = 3) & **ModelCheckpoint** keep only the top run.  
- `seen_videos.txt` prevents data leakage when new clips appear.

---

## 🚀 Quick Start
```bash
git clone https://github.com/your‑name/Basketball_foul_detection.git
cd Basketball_foul_detection

# 1. Install deps (CUDA 11.8 stack)
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_kaggle.py  # requires Kaggle API token

# 3. Train or resume
python train.py --epochs 8 --lr 3e-4


