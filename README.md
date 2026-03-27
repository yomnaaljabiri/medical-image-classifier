# 🫁 Medical Image Classifier
https://yomnaaljabiri.github.io/medical-image-classifier/frontend/
> X-Ray classifier for COVID-19, Pneumonia & Normal using EfficientNet-B4 — 96.6% accuracy

**Developed by:** Yomna Aljabari

---

## 📌 Overview

A full-stack medical imaging application that uses Transfer Learning to classify chest X-Ray images into three categories. The system features a clinical-style web interface with a built-in image viewer, real-time inference via a REST API, and drag-and-drop image upload.

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pre-trained) |
| Fine-tuned layers | Last 30 layers |
| Classifier head | Dropout(0.3) → Dense(512) → ReLU → Dropout(0.2) → Softmax(3) |
| Input size | 224 × 224 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Training platform | Google Colab (T4 GPU) |

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | **96.61%** |
| Best Epoch | 17 / 20 |
| Training Images | 12,879 |
| Validation Images | 2,274 |

---

## 🗂️ Dataset

[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) — Kaggle

| Class | Images |
|---|---|
| Normal | 10,192 |
| COVID-19 | 3,616 |
| Viral Pneumonia | 1,345 |

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, TorchVision
- **Model:** EfficientNet-B4 (Transfer Learning)
- **Data Augmentation:** RandomHorizontalFlip, RandomRotation, ColorJitter
- **Backend API:** Flask, Flask-CORS
- **Frontend:** HTML, CSS, JavaScript
- **Training:** Google Colab (T4 GPU)

---

## 📁 Project Structure

```
medical-image-classifier/
├── backend/
│   ├── model/
│   │   ├── model.py        ← EfficientNet-B4 architecture
│   │   ├── dataset.py      ← XRayDataset + Data Augmentation
│   │   └── train.py        ← Training loop with checkpointing
│   ├── api/
│   │   └── app.py          ← Flask REST API
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Clinical-style web dashboard
└── notebooks/
    └── training_notebook.ipynb
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yomnaaljabari/medical-image-classifier.git
cd medical-image-classifier
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Download the dataset
Download [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place it inside `backend/`.

Then run the data preparation script:
```bash
cd backend
python prepare_data.py
```

### 4. Train the model
```bash
cd backend/model
python train.py
```

### 5. Run the API
```bash
cd backend/api
python app.py
```

### 6. Open the interface
Open `frontend/index.html` in your browser.

---

## 🌐 API Endpoint

**POST** `/predict`

| Field | Type | Description |
|---|---|---|
| `file` | Image | Chest X-Ray image (PNG/JPG) |

**Response:**
```json
{
  "diagnosis": "طبيعي",
  "confidence": 97.3,
  "probabilities": {
    "طبيعي": 97.3,
    "التهاب رئوي": 1.8,
    "COVID-19": 0.9
  }
}
```

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified physician for medical advice.

---
