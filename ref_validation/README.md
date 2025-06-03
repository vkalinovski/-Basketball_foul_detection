## 🔄 Validation and Verdict Comparison

Below is the full pipeline that:
1. Loads the video  
2. Transcribes speech  
3. Asks the LLM “is there a foul?”  
4. Simultaneously annotates the video using the detector  
5. Compares both verdicts and computes metrics  

---

### 1. 📥 Video Loading  
- **Source:** `data/raw/…`  
- **Actions:**  
  - Decode the video stream into frames (e.g., `cv2.VideoCapture`)  
  - Extract the audio track (e.g., `ffmpeg` → WAV)  

---

### 2. 🗣️ Automatic Transcription  
- **Tool:** Whisper or another ASR  
- **Steps:**  
  1. Pass the audio file to the ASR model  
  2. Receive text with timestamps:  
     ```json
     [
       { "start": 0.00, "end": 2.35, "text": "Play is stopped for a foul" },
       { "start": 2.35, "end": 4.80, "text": "Player 23 commits the foul" }
     ]
     ```

---

### 3. 🤖 LLM Analysis
- **Prompt:**
  ```text
  Based on the transcript and game description,
  is there a foul in this segment? Reply "foul" or "no_foul".
  {
    "verdict": "foul",
    "confidence": 0.92
  }
  ```
  
---

---
## 4. 🦆 Model-Based Video Annotation

**Model:** YOLOv12n / RF-DETR

**Steps:**
1. Process each frame  
2. Detect and track players and the ball  
3. Apply foul-detection logic (e.g., based on zone overlap)  
4. Generate the final verdict  

**Output Format:**
```json
{
  "verdict": "foul",
  "confidence": 0.87
}
```

---

## 5. 📊 Verdict Comparison and Metric Calculation

| Metric         | Formula                                           | Description                                         |
| -------------- | ------------------------------------------------- | --------------------------------------------------- |
| Accuracy       | (TP + TN) / (P + N)                               | Proportion of matching predictions                   |
| Precision      | TP / (TP + FP)                                    | Precision of “foul” predictions                      |
| Recall         | TP / (TP + FN)                                    | Recall of “foul” predictions                         |
| F1-score       | 2 · (Precision · Recall) / (Precision + Recall)   | Balance between Precision and Recall                  |
| Agreement Rate | Matching verdicts / Total number of cases         | Frequency of matching verdicts between LLM and model |

**Where:**
- **TP** — true positive (both predicted “foul” and it is actually a foul)  
- **TN** — true negative (both predicted “no_foul” and this is correct)  
- **FP** — false positive (verdict “foul” is incorrect)  
- **FN** — false negative (missed foul)  

---

## Tools and Dependencies

- **Language:** Python 3.x  
- **Libraries:**
  - OpenCV  
  - FFmpeg  
  - Whisper (or alternative ASR)  
  - PyTorch / TensorFlow (for detection models)  
  - numpy, pandas, etc.  


