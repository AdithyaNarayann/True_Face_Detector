# 🦾 True Face Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![License](https://img.shields.io/badge/License-MIT-green)

True Face Detector is a compact, transfer-learning based image classifier that distinguishes between REAL and FAKE faces. It uses a fine-tuned ResNet-18 backbone and practical image augmentations to achieve robust performance on small-to-medium datasets.

## 🚀 Highlights
- **Model:** ResNet-18 (transfer learning, only last blocks and classifier trained)
- **Saved model:** `model/real_fake_model.pth`
- **Inference:** `predict.py` — single-image predictor that returns `REAL` or `FAKE`
- **Training script:** `train.py` — uses mixed-precision (AMP), class-weighted loss, scheduler, and data augmentations
- **Dataset layout:** `real_fake/real/*` and `real_fake/fake/*`

## 📁 Repository Structure

- `dataset.py` — custom `RealFakeDataset` with augmentations.
- `train.py` — training loop (transfer learning + AMP + scheduler).
- `predict.py` — loads the saved model and performs inference on an image.
- `model/real_fake_model.pth` — trained model weights (ResNet-18 classifier).
- `real_fake/` — expected dataset folder containing `real/` and `fake/` subfolders.
- `test/` — example/test images.

## 🧠 Model & Training Summary

- Architecture: ResNet-18 from `torchvision.models` with the final fully-connected layer replaced by `Linear(512, 2)`.
- Transfer learning: All parameters frozen except `layer4` and `fc` (only these are trainable).
- Loss: `CrossEntropyLoss` with class weights inversely proportional to class counts.
- Optimizer: `Adam` with `lr=1e-4`.
- Scheduler: `StepLR(step_size=5, gamma=0.5)`.
- Mixed precision training: `torch.cuda.amp` (`GradScaler` + `autocast`).
- Epochs: `15` (in `train.py`).
- Batch size: `32`.

### Augmentations (in `dataset.py`)

- Resize to 224x224
- Random Horizontal Flip
- Random Rotation (±10°)
- Color Jitter (brightness, contrast, saturation)
- Random Gaussian Blur (applied with probability 0.3)
- Normalization using ImageNet mean/std

## ⚙️ Installation

Create a virtual environment and install the essentials. On Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision pillow
```

Notes:
- If you have a CUDA-capable GPU, install the matching `torch`/`torchvision` build from https://pytorch.org.
- You can also install any extras you prefer (e.g., `tqdm`, `matplotlib`) for progress bars and visualization.

## ▶️ Training

Ensure your dataset follows the layout:

```
real_fake/
  ├─ real/
  └─ fake/
```

Train the model:

```powershell
python train.py
```

Training output:
- Prints device used (CPU/GPU)
- Progress by epoch (loss)
- Final test accuracy printed and weights saved to `model/real_fake_model.pth`

## 🧪 Inference

Quick single-image prediction with `predict.py`. Example usage:

```powershell
python predict.py
# or call the `predict(img_path)` function from the module in your own script
```

The script prints `REAL` or `FAKE` depending on the model prediction.

## 📌 Example: use `predict()` inside Python

```python
from predict import predict
print(predict('test/real_00015.jpg'))  # -> 'REAL' or 'FAKE'
```

## ✅ Tips & Best Practices

- Balance the dataset: the training uses class weights, but if classes are extremely imbalanced consider oversampling or additional augmentation.
- Fine-tuning: you can unfreeze more layers (e.g., `layer3`) for better accuracy if you have more data.
- Increase epochs and add validation checkpointing for production-quality models.
- Use a dedicated `requirements.txt` if you want reproducible installs.

## 🔭 Next Steps / Improvements

- Add a cleaner CLI for inference (image folder, webcam capture).
- Add evaluation metrics (precision, recall, F1) and confusion matrix plot.
- Add training checkpointing and best-model saving by validation accuracy.
- Convert to ONNX or TorchScript for faster deployment.

## 📜 License

This project is released under the MIT License — see `LICENSE`.

## 🙏 Credits

Built with PyTorch and torchvision. Dataset and model ideas inspired by common transfer-learning practices for small image classifiers.

---
Made with ❤️ — True Face Detector
