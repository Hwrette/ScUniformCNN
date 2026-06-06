# ==========================
# 1. 필요한 라이브러리 불러오기
# ==========================

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


# ==========================
# 2. GPU(MPS) 사용 설정
# ==========================

# M4 Mac이면 mps 사용
# 아니면 cpu 사용

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("사용 장치:", device)


# ==========================
# 3. 이미지 전처리
# ==========================

transform = transforms.Compose([

    # 모든 이미지를 128x128로 맞춤
    transforms.Resize((128, 128)),

    # 이미지를 텐서로 변환
    # (H,W,C) -> (C,H,W)
    # 0~255 -> 0~1
    transforms.ToTensor()

])


# ==========================
# 4. 데이터셋 불러오기
# ==========================

train_dataset = datasets.ImageFolder(
    root="dataset/학습데이터",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/테스트데이터",
    transform=transform
)

print("클래스 정보:")
print(train_dataset.class_to_idx)

# 예시 출력
# {'교복': 0, '사복': 1}


# ==========================
# 5. DataLoader 생성
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================
# 6. CNN 모델 정의
# ==========================

class UniformCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # ------------------
        # 특징 추출 부분
        # ------------------

        self.features = nn.Sequential(

            # 입력
            # 3 x 128 x 128

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # 결과
            # 32 x 64 x 64


            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            # 결과
            # 64 x 32 x 32


            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2, 2)

            # 결과
            # 128 x 16 x 16
        )


        # ------------------
        # 분류기 부분
        # ------------------

        self.classifier = nn.Sequential(

            # 128x16x16
            # ↓
            # 32768

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                256
            ),

            nn.ReLU(),

            # 과적합 방지
            nn.Dropout(0.5),

            # 최종 출력
            # 교복 / 사복

            nn.Linear(
                256,
                2
            )
        )

    def forward(self, x):

        # CNN 통과
        x = self.features(x)

        # 분류기 통과
        x = self.classifier(x)

        return x


# ==========================
# 7. 모델 생성
# ==========================

model = UniformCNN().to(device)

print(model)


# ==========================
# 8. 손실 함수
# ==========================

criterion = nn.CrossEntropyLoss()


# ==========================
# 9. 옵티마이저
# ==========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================
# 10. 학습
# ==========================

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 순전파
        outputs = model(images)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 기울기 초기화
        optimizer.zero_grad()

        # 오차 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {avg_loss:.4f}"
    )


# ==========================
# 11. 테스트
# ==========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # 가장 큰 값의 인덱스 선택
        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(
    f"테스트 정확도: "
    f"{accuracy:.2f}%"
)


# ==========================
# 12. 모델 저장
# ==========================

torch.save(
    model.state_dict(),
    "uniform_cnn.pth"
)

print("모델 저장 완료")