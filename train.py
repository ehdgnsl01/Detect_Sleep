# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net
import torch.optim as optim

def accuracy(y_pred, y_true):
    """
    y_pred: 모델 출력 로짓, shape = (batch, 1)
    y_true: 실제 레이블 Tensor, shape = (batch, 1)
    """
    prob = torch.sigmoid(y_pred)
    pred_tag = torch.round(prob)
    correct = (pred_tag == y_true).sum().float()
    return (correct / y_true.size(0)) * 100

def visualize_samples(x_data, y_data, num_samples=5):
    """
    x_data: NumPy array, shape = (N, 26, 34, 1)
    y_data: NumPy array, shape = (N, 1)
    num_samples: 시각화할 샘플 개수
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(num_samples, 1, figsize=(4, num_samples * 2))

    for i in range(min(num_samples, len(x_data))):
        img = x_data[i].reshape(26, 34)  # (26, 34)
        label = int(y_data[i][0])

        ax = axes[i]
        ax.set_title(f"Label: {label}")
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # -----------------------------
    # 1. 학습 데이터 로드
    # -----------------------------
    x_train = np.load('data/x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
    y_train = np.load('data/y_train.npy').astype(np.float32)  # (2586, 1)

    # 샘플 5개 시각화
    visualize_samples(x_train, y_train, num_samples=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 2. 데이터 증강 및 Dataset 생성
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.ToTensor(),               # NumPy (H, W, C) → Tensor (C, H, W)
        transforms.RandomRotation(10),       # ±10도 회전
        transforms.RandomHorizontalFlip(),   # 좌우 무작위 뒤집기
    ])
    train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)

    # -----------------------------
    # 3. DataLoader 생성 (num_workers=0 권장)
    # -----------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # -----------------------------
    # 4. 모델·손실함수·옵티마이저 설정
    # -----------------------------
    model = Net().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs('weights', exist_ok=True)
    PATH = 'weights/classifier_weights_iter_20.pth'

    # -----------------------------
    # 5. 학습 루프
    # -----------------------------
    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        batch_counter = 0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            # images: (batch, 1, 26, 34) 형태
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)           # (batch, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels).item()
            batch_counter += 1

            # 80개 배치마다 로그 출력
            if (batch_idx + 1) % 80 == 0:
                avg_loss = running_loss / 80
                avg_acc = running_acc / 80
                print(f"epoch: [{epoch}/{epochs}] train_loss: {avg_loss:.5f} train_acc: {avg_acc:.5f}")
                running_loss = 0.0
                running_acc = 0.0

        # 남은 배치가 있으면 평균 출력
        if batch_counter % 80 != 0:
            rem = batch_counter % 80
            avg_loss = running_loss / rem
            avg_acc = running_acc / rem
            print(f"epoch: [{epoch}/{epochs}] train_loss: {avg_loss:.5f} train_acc: {avg_acc:.5f}")

    print("learning finish")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")

if __name__ == "__main__":
    main()
