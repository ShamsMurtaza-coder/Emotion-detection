markdown# 😊 Facial Emotion Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red)
![Accuracy](https://img.shields.io/badge/Accuracy-68%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time facial emotion detection system built using a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset. Detects 7 universal human emotions live from webcam or uploaded images with confidence scores.

---

## 🎯 Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😄 |
| Sad | 😢 |
| Surprise | 😮 |
| Neutral | 😐 |

---

## 🚀 Live Demo

👉 **[Click here to try the live web app](https://huggingface.co/spaces/ShamsMurtaza/emotion-detector)**

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | FER-2013 (35,887 images) |
| Model Type | Custom 3-block CNN |
| Training Platform | Kaggle Tesla T4 x2 GPU |
| Training Accuracy | ~72% |
| Validation Accuracy | ~68% |
| Total Parameters | 5,281,479 |
| Training Time | ~12 minutes (GPU) |
| Input Size | 48 × 48 grayscale |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core programming language |
| TensorFlow 2.13 / Keras | Model building and training |
| OpenCV 4.13 | Face detection and webcam access |
| Streamlit | Interactive web application |
| NumPy / Pandas | Data processing |
| Matplotlib | Training visualization |
| Kaggle | Free GPU training |
| Hugging Face Spaces | Free cloud deployment |

---

## 📁 Project Structure
Emotion-detection/
├── app.py                  # Streamlit web application
├── realtime_emotion.py     # Live webcam detection script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── model/
└── emotion_model.h5    # Trained CNN model (60MB)

---

## 🧠 Model Architecture
Input (48 × 48 × 1 grayscale)
↓
┌─────────────────────────────────────┐
│ Block 1                             │
│ Conv2D(64) → BatchNorm              │
│ Conv2D(64) → BatchNorm              │
│ MaxPooling2D → Dropout(25%)         │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Block 2                             │
│ Conv2D(128) → BatchNorm             │
│ Conv2D(128) → BatchNorm             │
│ MaxPooling2D → Dropout(25%)         │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Block 3                             │
│ Conv2D(256) → BatchNorm             │
│ MaxPooling2D → Dropout(25%)         │
└─────────────────────────────────────┘
↓
Flatten
↓
Dense(512) → BatchNorm → Dropout(50%)
↓
Dense(7) + Softmax
↓
Output: [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]

---

## ▶️ Run Locally

### Prerequisites
- Python 3.11
- Webcam (for real-time detection)

### Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/ShamsMurtaza-coder/Emotion-detection.git
cd Emotion-detection

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3A: Run the web app
streamlit run app.py

# Step 3B: OR run real-time webcam detection
python realtime_emotion.py
```

### Controls
- Web app opens at `http://localhost:8501`
- Upload any face image OR use webcam tab
- Press `Q` to quit the webcam detection window

---

## 🏋️ Train the Model Yourself

1. Go to [Kaggle](https://www.kaggle.com) and create a free account
2. Create a new Notebook
3. Add the **FER-2013** dataset as input
4. Enable **GPU T4 x2** under Settings → Accelerator
5. Copy and run the training code
6. Download `emotion_model.h5` from the output section
7. Place it in the `model/` folder

Expected training time: **~12 minutes on GPU**

---

## 📚 Dataset

**FER-2013** — Facial Expression Recognition Challenge

| Property | Details |
|----------|---------|
| Total images | 35,887 |
| Image size | 48 × 48 pixels |
| Format | Grayscale |
| Classes | 7 emotions |
| Training samples | 28,709 |
| Validation samples | 3,589 |
| Test samples | 3,589 |

---

## 🔮 Future Improvements

- [ ] Transfer learning with VGGFace or ResNet50
- [ ] Multi-face simultaneous detection
- [ ] Mobile deployment with TensorFlow Lite
- [ ] Emotion tracking over time with graphs
- [ ] Gender and age detection alongside emotion
- [ ] Real-time emotion analytics dashboard

---

## 👨‍💻 Author

**Shams Murtaza**
- 🎓 BTech CSE (AI & ML) — 3rd Year
- 💻 GitHub: [@ShamsMurtaza-coder](https://github.com/ShamsMurtaza-coder)
- 🔗 LinkedIn: [Add your LinkedIn URL here]

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Kaggle
- [TensorFlow](https://tensorflow.org) — Model training framework
- [OpenCV](https://opencv.org) — Computer vision library
- [Streamlit](https://streamlit.io) — Web app framework
- [Hugging Face](https://huggingface.co) — Free deployment platform
- [Kaggle](https://kaggle.com) — Free GPU training

---

⭐ If you found this project helpful, please give it a star!
