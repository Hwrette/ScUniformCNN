import cv2
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
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
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
# 이미지 전처리 (학습 때와 동일)
# ==========================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.60837322473526, 0.5809004902839661, 0.5647146105766296],
        std=[0.2510661482810974, 0.24891965091228485, 0.24691884219646454]
    )
])

LABELS = {0: "uniform", 1: "casual"}


def predict_frame(frame_bgr):
    """OpenCV BGR 프레임 -> (라벨, 확률, 신뢰도) 반환"""
    # OpenCV는 BGR, PIL/torchvision은 RGB를 기대하므로 변환
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted = torch.argmax(outputs, dim=1).item()

    return predicted, probs.tolist()


def main():
    # macOS에서는 보통 0번 인덱스가 내장 웹캠. 안 되면 1, 2로 바꿔보세요.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라 권한(시스템 설정 > 개인정보 보호 > 카메라)을 확인하세요.")
        return

    print("웹캠 실행 중... 'q' 키를 누르면 종료됩니다.")

    frame_count = 0
    last_label = None
    last_probs = [0.0, 0.0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다.")
            break

        # 셀카처럼 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)

        # 매 프레임마다 추론하면 느려질 수 있으니 N프레임마다 한 번씩만 추론
        if frame_count % 5 == 0:
            last_label, last_probs = predict_frame(frame)

        frame_count += 1

        # 결과 텍스트 구성
        label_text = "교복" if last_label == 0 else "사복" if last_label == 1 else "..."
        conf = max(last_probs) * 100 if last_probs else 0.0
        display_text = f"{label_text} ({conf:.1f}%)"

        # 화면에 오버레이
        color = (0, 200, 0) if last_label == 0 else (0, 0, 200)
        cv2.rectangle(frame, (10, 10), (330, 60), (0, 0, 0), -1)
        cv2.putText(
            frame, display_text, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )

        cv2.imshow("Uniform Classifier - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
