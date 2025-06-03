# debuging.py

import os
import cv2
import numpy as np
import torch
from model import Net

# -----------------------------
# 설정 파라미터
# -----------------------------
IMG_SIZE = (34, 26)  # (width, height)
WEIGHTS_PATH = os.path.join('weights', 'classifier_weights_iter_20.pth')
IMAGE_PATH = 'data/image.jpeg'  # 테스트용 이미지 파일 경로

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 모델 로드
# -----------------------------
if not os.path.isfile(WEIGHTS_PATH):
    print(f"Error: 모델 가중치 '{WEIGHTS_PATH}' 파일을 찾을 수 없습니다.")
    print("train.py를 먼저 실행하여 가중치를 생성하세요.")
    exit(1)

model = Net().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

def predict_eye_state(eye_tensor):
    """
    eye_tensor: torch.Tensor, shape = (1,1,26,34)
    Returns: torch.Tensor 예측 클래스 (0=Closed, 1=Open)
    """
    with torch.no_grad():
        eye = eye_tensor.to(device)
        outputs = model(eye)                       
        pred_tag = torch.round(torch.sigmoid(outputs))
    return pred_tag

def main():
    # -----------------------------
    # 1. 이미지 로드 및 전처리
    # -----------------------------
    if not os.path.isfile(IMAGE_PATH):
        print(f"Error: 이미지 파일 '{IMAGE_PATH}'을(를) 찾을 수 없습니다.")
        exit(1)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: '{IMAGE_PATH}' 파일을 열 수 없습니다.")
        exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)  # (width, height) = (34,26)

    # (H, W) → (1, H, W, 1) → NumPy float32 → Tensor 형태로 변환
    eye_np = resized.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_tensor = torch.from_numpy(eye_np).permute(0, 3, 1, 2)  # → (1,1,26,34)

    # -----------------------------
    # 2. 예측
    # -----------------------------
    pred = predict_eye_state(eye_tensor)
    result = pred.item()  # 0 또는 1

    print(f"Prediction (0=Closed, 1=Open): {result}")

    # -----------------------------
    # 3. (선택) 결과 시각화
    # -----------------------------
    # 원본 그레이 이미지를 열고, 결과 텍스트를 오버레이하여 잠깐 보여주고 싶다면 아래 주석 해제
    
    display = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    state_text = 'Open' if result == 1 else 'Closed'
    color = (0, 255, 0) if result == 1 else (0, 0, 255)
    cv2.putText(display, state_text, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow('Debug Eye', cv2.resize(display, (200, 150), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
