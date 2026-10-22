# 🌿 PlantPulse: Cotton Disease Detection & Agronomy Intelligence

**PlantPulse** is an automated deep learning web application that diagnoses cotton crop foliar diseases from leaf photographs, provides real-time confidence metrics, offers curated organic and chemical remedies with purchasing options, and includes an empirical scientific model benchmark suite.

---

## 🎯 What It Does
1. **Instant Crop Disease Diagnosis**: Upload a cotton leaf photo via drag-and-drop or file browser to detect disease class across 7 categories (*Aphids, Army worm, Bacterial blight, Healthy leaf, Powdery mildew, Target spot, and Out-of-domain Unknown*).
2. **Dual-Model Inference**: Switch seamlessly between **MobileNetV2 (Transfer Learning)** and **Sequential 3-Block CNN (From Scratch)**.
3. **Agronomy Treatment Protocols**: Displays biological and chemical remedies with verified formulation links for detected diseases.
4. **Interactive Performance Comparison**: Side-by-side benchmark table with live metric count-ups, loss/accuracy curves, and architectural explainers.
5. **Modern User Experience**: Dark/Light mode theme memory, instant image zoom lightbox, animated confidence gauges, and mobile responsiveness.

---

## 🧠 The 2 Deep Learning Algorithms & Accuracies

Both algorithms were trained and evaluated on an **identical, stratified 70% Train / 15% Validation / 15% Held-Out Test split** (384 unseen test images, seed `42`):

| Algorithm | Architecture Type | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV2** (`algo2`) | Transfer Learning (ImageNet pre-trained + Inverted Residuals) | **88.54%** | **89.57%** | **88.54%** | **88.48%** | 🏆 **Winner & Default** |
| **Sequential 3-Block CNN** (`algo1`) | Built From Scratch (3 Conv Blocks: 32 → 64 → 128 + Dropout) | **74.74%** | **76.85%** | **74.74%** | **74.70%** | 🧪 **Scratch Baseline** |

---

## 📁 Project Directory Structure

```text
cotton_disease_detection/
├── app.py                     # Main Flask web application & model registry
├── requirements.txt           # Verified Python runtime dependencies
├── README.md                  # Complete documentation and user guide
├── .gitignore                 # Caches, environment, uploads, and quarantine filters
├── data/                      # Standardized cotton disease leaf dataset (7 classes)
├── models/                    # Trained neural network weights (.h5)
│   ├── algo1.h5               # Sequential 3-Block CNN weights (74.74%)
│   └── algo2.h5               # MobileNetV2 Transfer Learning weights (88.54%)
├── src/                       # Core ML training and evaluation scripts
│   ├── algo1_custom_cnn.py    # Sequential 3-Block CNN architecture & training
│   ├── algo2_mobilenetv2.py   # MobileNetV2 fine-tuning & training routine
│   ├── compare_accuracy.py    # Side-by-side benchmarking & chart generator
│   └── dataset_utils.py       # Stratified 70/15/15 generator and metric reporter
├── results/                   # Metric JSONs, confusion matrices & curves
│   ├── algo1_metrics.json
│   ├── algo1_confusion_matrix.png
│   ├── algo1_training_curves.png
│   ├── algo2_metrics.json
│   ├── algo2_confusion_matrix.png
│   ├── algo2_training_curves.png
│   └── comparison.png         # 2-model comparative bar chart
├── static/                    # Front-end static assets
│   ├── css/
│   │   └── custom_ui.css      # Design system, dark mode tokens, interactive styles
│   ├── js/                    # Client-side helper scripts
│   ├── images/                # Product photos, cotton banner, and logo
│   ├── uploads/               # Isolated storage for runtime user uploads
│   └── comparison.png         # Publicly served benchmark chart
├── templates/                 # Jinja2 HTML views
│   ├── index.html             # Homepage: drag-and-drop & model cards
│   ├── result.html            # Diagnostic result, confidence gauge & solutions
│   ├── comparison.html        # Benchmark table, explainer cards & chart zoom
│   ├── weather.html           # Live agricultural weather forecasts
│   └── al.html                # Virtual farming assistant
└── _to_delete/                # Quarantine directory containing retired legacy files
```

---

## ⚡ Setup & Installation

### 1. Clone & Enter Project Directory
```bash
cd c:\Users\ASUS\Desktop\cotton_disease_detection\cotton_disease_detection\chaitanya
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Web Application

Start the Flask application server:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser:
* **Upload Leaf Image**: Drag and drop or browse to select a picture.
* **Select Algorithm**: Choose between **MobileNetV2 (88.54%)** or **Sequential 3-Block CNN (74.74%)**.
* **Detect**: Click **Detect Disease & Get Solutions** for immediate diagnosis.

---

## 📊 How to Run the Accuracy Comparison

To evaluate both models side-by-side, print terminal summaries, and update `results/comparison.png` and `static/comparison.png`:
```bash
python src/compare_accuracy.py
```
View the interactive graphical table on the web app at:
**[http://127.0.0.1:5000/comparison](http://127.0.0.1:5000/comparison)**.

---

## 🏋️ How to Train Each Algorithm

Both training routines automatically pull from `data/`, use fixed seed `42`, and log metrics to `results/`:

* **Train MobileNetV2 (Transfer Learning)**:
  ```bash
  python src/algo2_mobilenetv2.py
  ```
* **Train Custom CNN (From Scratch)**:
  ```bash
  python src/algo1_custom_cnn.py
  ```

---

## 🗑️ Safe Quarantine (`_to_delete/`)
Unused legacy files (such as `cotton_disease_model.h5`, `train_baseline_fair.py`, `training.py`, `baseline_fair.h5`, unused static photos, and bytecode caches) were placed in `_to_delete/` rather than being deleted immediately, allowing safe manual review.
