# 🦾 True Face Detector

<!-- header badges -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy-68.22%25-yellow](https://img.shields.io/badge/Accuracy-68.22%25-yellow)](#results)

---

## ✨ Overview

True Face Detector is a compact, transfer-learning image classifier that identifies whether a face image is REAL or FAKE. It's built with PyTorch and uses a fine-tuned ResNet-18 backbone together with practical augmentations for reliable performance on modest datasets.

---

## 🎯 Quick Highlights

- **Project name:** True Face Detector
- **Model:** ResNet-18 (transfer learning — fine-tuned `layer4` + classifier)
- **Saved weights:** `model/real_fake_model.pth`
- **Test accuracy (reported):** `68.22%`
- **Train script:** `train.py` (AMP, class-weighted loss, scheduler)
- **Predict script:** `predict.py` (returns `REAL` or `FAKE`)
- **Dataset layout:** `real_fake/real/*` and `real_fake/fake/*`

---

## 📂 Repo Layout

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

## 🧠 Model & Training Details

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

**Reported final test accuracy:** **68.22%**

### Augmentations (from `dataset.py`)

- Resize to 224×224
- Random Horizontal Flip
- Random Rotation ±10°
- Color Jitter (brightness / contrast / saturation)
- Random Gaussian Blur (p=0.3)
- ImageNet normalization


---

## ⚙️ Install (quick)

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

## ▶️ Training (one-liner)

Make sure your dataset is placed at `real_fake/` with `real/` and `fake/` subfolders. Then run:

```powershell
python train.py
```

Logs:
- Device used (CPU/GPU)
- Per-epoch loss
- Final test accuracy and saved model at `model/real_fake_model.pth`

---

## 🧪 Inference

To run the bundled example prediction:

```powershell
python predict.py
```

To use the `predict()` function from your code:

```python
from predict import predict
print(predict('test/real_00015.jpg'))
```

The function returns `"REAL"` or `"FAKE"`.

---

## ✅ Practical Tips

- Keep your `real_fake/` dataset organized and balanced where possible.
- If you have a GPU, training will be substantially faster; AMP is already enabled.
- Use `requirements.txt` for reproducible installs (you've already generated it).

---

## 📜 License

MIT — see `LICENSE`.

---

Made with ❤️ — True Face Detector
