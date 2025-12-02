from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random

class RealFakeDataset(Dataset):
    def __init__(self, folder):
        self.real_paths = list(Path(folder, "real").glob("*"))
        self.fake_paths = list(Path(folder, "fake").glob("*"))

        self.paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)

        combined = list(zip(self.paths, self.labels))
        random.shuffle(combined)
        self.paths, self.labels = zip(*combined)

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.tf(img)
        label = self.labels[idx]
        return img, label