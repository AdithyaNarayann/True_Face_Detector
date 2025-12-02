from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class RealFakeDataset(Dataset):
    def __init__(self, folder):
        self.real_paths = list(Path(folder, "real").glob("*"))
        self.fake_paths = list(Path(folder, "fake").glob("*"))
        self.paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths) # this is list multiplication, [0] * 120 = [0,0,0,...,0]  120 times
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img  = self.tf(img)
            label = self.labels[idx]
            return img, label
        except Exception as e:
            print(f"Error loading image {self.paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self.paths))
        