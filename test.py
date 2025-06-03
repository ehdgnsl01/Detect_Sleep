# test.py

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net

def visualize_samples(x_data, y_data, num_samples=5):
    """
    x_data: NumPy array, shape = (N, 26, 34, 1)
    y_data: NumPy array, shape = (N, 1)
    num_samples: 시각화할 샘플 개수
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_samples, 1, figsize=(4, num_samples * 2))

    for i in range(min(num_samples, len(x_data))):
        img = x_data[i].reshape(26, 34)
        label = int(y_data[i][0])

        ax = axes[i]
        ax.set_title(f"Label: {label}")
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def accuracy(y_pred, y_true):
    """
    y_pred: 모델 출력 로짓, shape = (batch, 1)
    y_true: 실제 레이블 Tensor, shape = (batch, 1)
    """
    prob = torch.sigmoid(y_pred)
    pred_tag = torch.round(prob)
    correct = (pred_tag == y_true).sum().float()
    return (correct / y_true.size(0)) * 100

def main():
    # -----------------------------
    # 1. 검증 데이터 로드 및 시각화
    # -----------------------------
    x_val = np.load('data/x_val.npy').astype(np.float32)  # (N, 26, 34, 1)
    y_val = np.load('data/y_val.npy').astype(np.float32)  # (N, 1)

    # 검증용 샘플 5개 시각화
    visualize_samples(x_val, y_val, num_samples=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 2. Dataset 및 DataLoader 생성
    # -----------------------------
    test_transform = transforms.Compose([
        transforms.ToTensor()  # NumPy (H, W, C) → Tensor (C, H, W)
    ])
    test_dataset = eyes_dataset(x_val, y_val, transform=test_transform)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  # Windows 환경에서 0으로 설정
    )

    # -----------------------------
    # 3. 모델 로드
    # -----------------------------
    model = Net().to(device)
    weights_path = os.path.join('weights', 'classifier_weights_iter_20.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # -----------------------------
    # 4. 검증 루프
    # -----------------------------
    total_acc = 0.0
    count = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            # images: (batch, 1, 26, 34) 형태 → 바로 모델에 입력
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # (batch, 1)
            batch_acc = accuracy(outputs, labels).item()
            total_acc += batch_acc
            count += 1

    if count > 0:
        print(f"average acc: {total_acc / count:.5f} %")
    else:
        print("No validation samples found.")

    print("test finish!")

if __name__ == "__main__":
    main()
