# True Face Detector

<!-- header badges -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

True Face Detector is a compact, transfer-learning image classifier that identifies whether a face image is REAL or FAKE. It's built with PyTorch and uses a fine-tuned ResNet-18 backbone together with practical augmentations for reliable performance on modest datasets.

---

## Quick Highlights

- **Project name:** True Face Detector
- **Model:** ResNet-18 (transfer learning — fine-tuned `layer4` + classifier)
- **Saved weights:** `model/real_fake_model.pth`
- **Train script:** `train.py` (AMP, class-weighted loss, scheduler)
- **Predict script:** `predict.py` (returns `REAL` or `FAKE`)
- **Dataset layout:** `real_fake/real/*` and `real_fake/fake/*`

---

## Repo Layout

| Path | Purpose |
|---|---|
| `dataset.py` | `RealFakeDataset` implementation — augmentations & preprocessing |
| `train.py` | Training script: uses AMP, scheduler and saves `model/real_fake_model.pth` |
| `predict.py` | Single-image inference utility — returns `REAL` or `FAKE` |
| `model/` | Checkpoint storage (`real_fake_model.pth`) |
| `real_fake/` | Dataset root: `real/` and `fake/` subfolders required for training |
| `test/` | Example images for quick inference checks |
| `requirements.txt` | Pinned package list for reproducible environment |

Quick tree view:

```text
real_fake/
├─ real/
├─ fake/
model/
requirements.txt
train.py
predict.py
dataset.py
README.md
```

---

## Model & Training Details

| Item | Setting |
|---|---|
| Architecture | ResNet-18 (`torchvision.models`) with final `Linear(512, 2)` |
| Trainable layers | `layer4`, `fc` (rest frozen for transfer learning) |
| Loss | `CrossEntropyLoss` with class weights (inverse to class counts) |
| Optimizer | Adam (`lr=1e-4`) |
| Scheduler | `StepLR(step_size=5, gamma=0.5)` |
| Mixed precision | Enabled (`torch.cuda.amp`: `GradScaler` + `autocast`) |
| Epochs | 15 |
| Batch size | 32 |

### Augmentations (from `dataset.py`)

- Resize to 224×224
- Random Horizontal Flip
- Random Rotation ±10°
- Color Jitter (brightness / contrast / saturation)
- Random Gaussian Blur (p=0.3)
- ImageNet normalization

---

## Dataset

**Dataset used to fine-tune this model:** CIPLAb Real and Fake Face Detection Dataset

- **Source:** Kaggle — https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
- **Description:**
	- **Real Faces** — authentic, unaltered human face images (various poses, lighting and backgrounds).
	- **Fake Faces** — manipulated / forged faces produced using morphing, distortion or other synthetic manipulation techniques.
- **Usage in this project:** the dataset was organized into two folders (`real/` and `fake/`) and loaded via `dataset.py`'s `RealFakeDataset` class for training and evaluation.

Recommended local folder layout for training (used by `train.py`):

```
real_fake/
├─ real/    # all authentic face images
├─ fake/    # all manipulated / forged images
```

Please refer to the Kaggle page for dataset licensing, download instructions and dataset statistics.

---

## Results

- **Final test accuracy:** **68.22%**

Brief notes:
- Evaluation used an 80/20 train/test split generated in `train.py` (`random_split` with seed 42).
- Class-weighted `CrossEntropyLoss` was used to reduce class imbalance effects; augmentations were applied during training.
- This result is a reasonable baseline using ResNet-18 with partial fine-tuning; there is room for improvement with additional data, validation-based checkpointing, and model tuning.

---


## Install

Install dependencies from `requirements.txt` (you already generated this file):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer to install only essentials (CPU/GPU choice):

```powershell
pip install torch torchvision pillow
```

---

## Training 

Make sure your dataset is placed at `real_fake/` with `real/` and `fake/` subfolders. Then run:

```powershell
python train.py
```

Logs:
- Device used (CPU/GPU)
- Per-epoch loss
- Final test accuracy and saved model at `model/real_fake_model.pth`

---

## Inference

To run the bundled example prediction:

```powershell
python predict.py
```

To use the `predict()` function from your code:

```python
from predict import predict
print(predict('test/image.jpg'))
```

The function returns `"REAL"` or `"FAKE"`.

---

## Practical Tips

- Keep `real_fake/` organized and aim for class balance; the training uses class-weighted loss but oversampling can help when imbalance is severe.
- Use `requirements.txt` for reproducible installs and prefer a GPU for training (AMP is enabled in `train.py`).

---

## Future Improvements

- Add a validation split and checkpointing to save the best model by validation accuracy instead of final epoch weights.
- Report per-class metrics (precision, recall, F1) and plot a confusion matrix to identify error modes.
- Experiment with stronger backbones (ResNet-50, EfficientNet) or unfreeze additional layers for deeper fine-tuning.
- Address class imbalance more aggressively (oversampling, focal loss) or ensemble multiple fine-tuned models.

---

## License

MIT — see `LICENSE`.

---

### Author
**Adithya Narayan V S**  

*Built to explore face forgery detection using PyTorch through a real-vs-fake image classification task.*
