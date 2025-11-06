# 🦏 RhinoGuardians ML

**YOLOv5/YOLOv8 model training & evaluation for black rhino & threat detection**

Built for **AI Genesis Hackathon 2025** (Nov 14-19) | Lead: Rania & Shrusti

---

## 📋 Overview

This repo handles all machine learning aspects:
- **Dataset Preparation** – Collecting, labeling, augmenting rhino/threat images
- **Model Training** – Fine-tuning YOLOv5/v8 on detection tasks
- **Evaluation** – Computing mAP, precision, recall, F1 scores
- **Optimization** – Quantization, pruning for edge deployment
- **Inference** – Exporting models for production backend

---

## 🤖 Tech Stack

- **Framework:** PyTorch
- **Model:** YOLOv5/YOLOv8 (Ultralytics)
- **Dataset Tool:** Roboflow (optional for dataset management)
- **Evaluation:** scikit-learn, pycocotools
- **Optimization:** TorchScript, ONNX

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (optional but recommended for training)
- 8GB+ RAM

### Installation

```bash
# Clone repo
git clone https://github.com/RhinoGuardians/rhinoguardians-ml.git
cd rhinoguardians-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv5 repo
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

---

## 📂 Project Structure

```
ml/
├── data/
│   ├── raw/                     # Original images (before processing)
│   │   ├── rhino/
│   │   ├── poacher/
│   │   └── vehicle/
│   ├── processed/               # Preprocessed images
│   └── annotations/             # YOLO format labels (.txt files)
├── datasets/
│   └── rhino_detection/         # Complete dataset in YOLO format
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml            # Dataset config for YOLOv5
├── models/
│   ├── yolov5s.pt               # Pretrained baseline
│   ├── yolov5_custom_best.pt    # Fine-tuned model (best weights)
│   └── yolov5_custom_last.pt    # Last checkpoint
├── scripts/
│   ├── prepare_dataset.py       # Download & organize data
│   ├── augment_images.py        # Data augmentation
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Model evaluation
│   └── export_model.py          # Export to ONNX/TorchScript
├── notebooks/
│   ├── exploratory_analysis.ipynb    # Data exploration
│   ├── training_log.ipynb            # Training visualization
│   └── performance_analysis.ipynb    # Results & metrics
├── config/
│   ├── data.yaml                # Dataset paths
│   ├── hyp.yaml                 # Hyperparameters
│   └── model_config.yaml        # Model architecture
├── results/
│   ├── confusion_matrix.png
│   ├── precision_recall.png
│   └── training_results.csv
├── requirements.txt
└── README.md                    # This file
```

---

## 📊 Dataset Preparation

### Data Format (YOLO)

Images in `images/` folder with corresponding labels in `labels/` folder:

```
images/
  train/
    img001.jpg
    img002.jpg
  val/
    img101.jpg
  test/
    img201.jpg

labels/
  train/
    img001.txt  # Format: <class_id> <x_center> <y_center> <width> <height>
    img002.txt  #         (all normalized 0-1)
  val/
    img101.txt
  test/
    img201.txt
```

Example label (img001.txt):
```
0 0.5 0.5 0.3 0.4    # class 0 (rhino), center at (0.5, 0.5), 30% width, 40% height
```

### Dataset Config (data.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3  # Number of classes
names: ['rhino', 'poacher', 'vehicle']
```

---

## 🎓 Training

### Quick Start Training

```bash
# Train YOLOv5 on rhino dataset
python scripts/train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data config/data.yaml \
  --weights yolov5s.pt \
  --device 0  # GPU 0

# Output: runs/detect/train/
```

### Training Configuration (config/hyp.yaml)

```yaml
# Training hyperparameters
lr0: 0.001              # Initial learning rate
lrf: 0.1                # Final learning rate ratio
momentum: 0.937         # SGD momentum
weight_decay: 0.0005    # Weight decay
warmup_epochs: 3        # Warmup epochs
warmup_momentum: 0.8    # Warmup momentum
box: 0.05               # Box loss gain
cls: 0.5                # Class loss gain
cls_pw: 1.0             # Class positive weight
```

### Advanced Training

```bash
# Resume from checkpoint
python scripts/train.py --resume runs/detect/train/weights/last.pt

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node 2 scripts/train.py \
  --data config/data.yaml --device 0,1

# Hyperparameter sweep
python scripts/train.py --data config/data.yaml --hyp config/hyp_sweep.yaml
```

---

## 📈 Evaluation

### Compute Metrics

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --weights models/yolov5_custom_best.pt \
  --data config/data.yaml \
  --task test

# Output metrics:
# - mAP (mean Average Precision)
# - Precision, Recall, F1
# - Per-class metrics
```

### Results Interpretation

```
Class    Images    Labels    P     R  mAP@.5  mAP@.5:.95
all        150      450  0.92  0.87    0.89      0.65
rhino      150      180  0.95  0.91    0.93      0.71
poacher    150      150  0.89  0.85    0.87      0.61
vehicle    150      120  0.92  0.85    0.88      0.63
```

**Target Metrics:**
- mAP >= 0.85
- Precision >= 0.90
- Recall >= 0.85

---

## 🔬 Data Augmentation

```python
# In scripts/augment_images.py
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate,
    RandomBrightnessContrast, Blur, GaussNoise
)

augmentation = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.2),
    Rotate(limit=45, p=0.5),
    RandomBrightnessContrast(p=0.3),
    GaussNoise(p=0.1),
], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 🧠 Model Variants

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **YOLOv5n** | 2.6M | ~650 FPS | 28.4 mAP | Edge devices (slow) |
| **YOLOv5s** | 7.2M | ~300 FPS | 37.3 mAP | Baseline (fast) |
| **YOLOv5m** | 21.2M | ~100 FPS | 45.4 mAP | Balanced |
| **YOLOv5l** | 46.5M | ~50 FPS | 48.2 mAP | High accuracy |
| **YOLOv5x** | 86.7M | ~20 FPS | 50.7 mAP | Max accuracy |

**For hackathon:** Use **YOLOv5s** (balance of speed & accuracy)

---

## 🚀 Export Models

### Export to ONNX (Cross-platform)

```bash
python scripts/export_model.py \
  --weights models/yolov5_custom_best.pt \
  --format onnx

# Output: models/yolov5_custom_best.onnx
```

### Export to TorchScript (PyTorch)

```bash
python scripts/export_model.py \
  --weights models/yolov5_custom_best.pt \
  --format torchscript

# Output: models/yolov5_custom_best.torchscript
```

### Export for Mobile (TFLite)

```bash
python scripts/export_model.py \
  --weights models/yolov5_custom_best.pt \
  --format tflite

# Output: models/yolov5_custom_best.tflite
```

---

## 🔗 Dataset Sources

Public datasets for training:
- **COCO** – General object detection dataset (14M images)
- **Kaggle** – Animal detection datasets
- **Roboflow** – Open-source computer vision datasets
- **African Wildlife Dataset** – Specific rhino/poacher images
- **Custom** – Collect from drones & camera traps in reserves

### Download Example

```bash
# Using Roboflow (free tier)
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='your_key'); 
           project = rf.workspace().project('rhino-detection'); 
           dataset = project.download('yolov5')"
```

---

## 📊 Monitoring Training

```bash
# View TensorBoard logs
tensorboard --logdir runs/detect/train/

# Or check training CSV
cat runs/detect/train/results.csv
```

---

## 🎯 Optimization Tips

**For Speed:**
- Use smaller model (YOLOv5s vs YOLOv5l)
- Reduce input image size (416 vs 640)
- Enable half-precision (--half flag)

**For Accuracy:**
- Use larger model (YOLOv5l vs YOLOv5s)
- Increase training epochs (200 vs 100)
- Augment dataset more aggressively
- Collect more labeled data

**For Edge Deployment:**
- Quantize model (INT8)
- Prune unnecessary weights
- Use TFLite or ONNX format

---

## 🐛 Troubleshooting

**Issue:** CUDA out of memory
- **Solution:** Reduce batch size: `--batch 8` or `--batch 4`

**Issue:** Training loss not decreasing
- **Solution:** Reduce learning rate: `--lr0 0.0005`

**Issue:** mAP below target
- **Solution:** Collect more training data (need 200+ images per class)

---

## 📚 Resources

- [YOLOv5 Docs](https://github.com/ultralytics/yolov5)
- [YOLOv8 Docs](https://github.com/ultralytics/ultralytics)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Roboflow Docs](https://docs.roboflow.com/)

---

## 👥 Lead: Rania & Shrusti

Dataset issues? Training questions? Ping the ML team!

---

**Built with ❤️ for RhinoGuardians AI Genesis Hackathon 2025**
