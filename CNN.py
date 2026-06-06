# ==========================
# 1. 필요한 라이브러리 불러오기
# ==========================

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

plt.rcParams["font.family"] = "AppleGothic"

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

def compute_mean_std(loader):
    """학습 데이터 기준 채널별 평균·표준편차 계산"""

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0

    for images, _ in loader:
        batch = images.size(0)
        images = images.view(batch, images.size(1), -1)
        mean += images.mean(2).sum(dim=0)
        std += images.std(2).sum(dim=0)
        total += batch

    mean /= total
    std /= total

    return mean, std


base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_for_stats = datasets.ImageFolder(
    root="dataset/train",
    transform=base_transform
)

stats_loader = DataLoader(
    train_for_stats,
    batch_size=32,
    shuffle=False
)

mean, std = compute_mean_std(stats_loader)

print("정규화 평균:", mean.tolist())
print("정규화 표준편차:", std.tolist())

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean.tolist(),
        std=std.tolist()
    )
])


# ==========================
# 4. 데이터셋 불러오기
# ==========================

train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
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

# 처음부터 학습하는 CNN은 10 epoch면 부족한 경우가 많음
epochs = 25

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
# 11. 테스트 + 혼동 행렬
# ==========================

model.eval()

correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)
        correct += (
            predicted == labels
        ).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

accuracy = 100 * correct / total

print(
    f"테스트 정확도: "
    f"{accuracy:.2f}%"
)

num_classes = len(train_dataset.classes)
confusion = torch.zeros(num_classes, num_classes, dtype=torch.int32)

for true_label, pred_label in zip(all_labels, all_preds):
    confusion[true_label, pred_label] += 1

print("\n혼동 행렬:")
print("클래스 순서:", train_dataset.classes)
print(confusion.numpy())

class_names = train_dataset.classes

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(confusion.numpy(), cmap="Blues")

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j, i,
            int(confusion[i, j]),
            ha="center",
            va="center",
            color="white" if confusion[i, j] > confusion.max() / 2 else "black"
        )

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("혼동 행렬 저장: confusion_matrix.png")

print("\n클래스별 성능:")
for i, name in enumerate(class_names):
    tp = confusion[i, i].item()
    fn = confusion[i, :].sum().item() - tp
    fp = confusion[:, i].sum().item() - tp
    tn = confusion.sum().item() - tp - fn - fp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(
        f"  {name}: "
        f"tp {tp}, "
        f"tn {tn}, "
        f"fp {fp}, "
        f"fn {fn}, "
        f"정밀도 {precision:.2%}, "
        f"재현율 {recall:.2%}"
    )


# ==========================
# 12. 모델 저장
# ==========================

torch.save(
    model.state_dict(),
    "uniform_cnn.pth"
)

print("모델 저장 완료")