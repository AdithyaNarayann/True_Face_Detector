import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load("model/real_fake_model.pth", map_location=device))
model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    return "REAL" if pred.item() == 0 else "FAKE"

print(predict("test/image.jpg"))