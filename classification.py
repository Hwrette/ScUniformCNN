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
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.60837322473526, 0.5809004902839661, 0.5647146105766296],
        std=[0.2510661482810974, 0.24891965091228485, 0.24691884219646454]
    )
])


# ==========================
# 이미지 읽기
# ==========================


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)


    # ==========================
    # 예측
    # ==========================

    with torch.no_grad():

        outputs = model(image)

        predicted = torch.argmax(outputs, dim=1)

        print(f"이미지 {image_path}의 예측 결과:", outputs.tolist()[0], ',',torch.softmax(outputs, dim=1).tolist()[0], ',',predicted.item())

        if predicted.item() == 0:
            print("예측: 교복")
        else:
            print("예측: 사복")

predict_image("image.png")
predict_image("image2.jpg")
predict_image("image3.jpg")
predict_image("image4.jpg")
predict_image("image5.jpg")
predict_image("고척고 교복.JPG")
predict_image("동성 교복.jpg")
predict_image("잠옷.jpg")