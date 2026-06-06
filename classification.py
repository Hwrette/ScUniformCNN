import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ==========================
# CNN 구조 (학습할 때랑 동일해야 함)
# ==========================

class UniformCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================
# 모델 불러오기
# ==========================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = UniformCNN().to(device)

model.load_state_dict(
    torch.load(
        "uniform_cnn.pth",
        map_location=device
    )
)

model.eval()


# ==========================
# 이미지 전처리
# ==========================

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# ==========================
# 이미지 읽기
# ==========================

img = Image.open("test.jpg").convert("RGB")

img = transform(img)

img = img.unsqueeze(0)

img = img.to(device)


# ==========================
# 예측
# ==========================

with torch.no_grad():

    outputs = model(img)

    predicted = torch.argmax(outputs, dim=1)

    if predicted.item() == 0:
        print("예측: 교복")
    else:
        print("예측: 사복")