# 🦾 True Face Detector

<!-- header badges -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)](https://pytorch.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

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

- `dataset.py` — `RealFakeDataset` (augmentations & transforms)
- `train.py` — training loop, mixed precision, scheduler, saves `model/real_fake_model.pth`
- `predict.py` — single-image inference
- `model/` — contains model checkpoint
- `real_fake/` — dataset folder (required for training)
- `test/` — sample images to try inference
- `requirements.txt` — pinned environment (use `pip install -r requirements.txt`)

---

## 🧠 Model & Training Details

- **Architecture:** ResNet-18 from `torchvision.models` with final layer `Linear(512, 2)`.
- **Transfer learning strategy:** Freeze early layers; only `layer4` and `fc` are trainable.
- **Loss:** `CrossEntropyLoss` with class weights (inverse of class counts).
- **Optimizer:** `Adam`, `lr=1e-4`.
- **Scheduler:** `StepLR(step_size=5, gamma=0.5)`.
- **Mixed precision:** `torch.cuda.amp` (`GradScaler` + `autocast`) for faster GPU training.
- **Epochs / Batch:** 15 epochs, batch size 32 (from `train.py`).

**Reported final test accuracy:** 68.22% (printed by the training script after evaluation).

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
