# Basketball_foul_detection
---

<h3 align="center">This project is currently under development🙂</h3>

<div align="center">
  <img height="400" width="840" src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif"  />
</div>

---

## 🏀 Project Overview
This project focuses on detecting fouls in basketball games and validating the decisions made by referees. The goal is to use computer vision and machine learning techniques to analyze game footage, identify fouls, and provide insights into the accuracy of referee calls.

---

## 📂 Repository Structure

### 📁 **[Data collection](https://github.com/vkalinovski/-Basketball_foul_detection/tree/main/data_collection)**
- Used for collecting videos from YouTube, defining foul moment intervals, spliting them into video intervals and uploading to our cloud storage

### 📁 Labeling
- Using **Label Studio** with **Video Object Tracking** to manually annotate short video clips, labeling each segment explicitly as `foul` or `no_foul` for precise ground truth creation.

### 📁 Dataset Creation
- Exported annotated labels as JSON files from Label Studio, structured them systematically, and uploaded the final labeled dataset to [**Kaggle**](https://www.kaggle.com/datasets/sesmlhs/foul-detection-test/data?select=M16.mp4), facilitating efficient training of our model.

### 📁 Model Training
- Fine-tuned a **SlowFast R50 model** from PyTorchVideo using **PyTorch Lightning** with full AMP support.
- Applied advanced video augmentations: temporal subsampling, crop, flip, jitter, and Gaussian noise.
- Integrated **Mixup augmentation**, **label smoothing**, and **OneCycleLR** scheduling.
- Used **Google Colab GPU**, stored checkpoints and logs in **Google Drive**.
- Metrics and training logs visualized via **TensorBoard**.

### 📁 Model Tuning
- Tuned hyperparameters including learning rate and dropout using **Optuna**.
- Leveraged **early stopping** and **checkpointing** based on validation accuracy.
- Separated newly added videos via a `seen_videos.txt` file to avoid data leakage.
- Improved model generalization with **gradient clipping** and selective layer freezing.

### 📁 


