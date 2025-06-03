# data_loader.py

from torch.utils.data import Dataset
import torch

class eyes_dataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        """
        Args:
            x_data (np.ndarray): shape = (N, 26, 34, 1)
            y_data (np.ndarray): shape = (N, 1)
            transform (callable, optional): 이미지에 적용할 transform (PIL 또는 numpy → Tensor 변환 포함)
        """
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __getitem__(self, idx):
        # 1) NumPy 배열 상태로 눈 이미지와 레이블 가져오기
        img_np = self.x_data[idx]        # (26, 34, 1) → dtype: numpy.float32
        label_np = self.y_data[idx]      # (1,) → dtype: numpy.float32

        # 2) transform이 정의되어 있으면, NumPy 배열을 바로 transform에 전달
        if self.transform is not None:
            # transforms.ToTensor()를 포함하고 있다면, 여기서 (H,W,C) numpy → (C,H,W) Tensor로 바뀜
            img = self.transform(img_np)
        else:
            # transform이 없으면, 수동으로 NumPy → Tensor 변환
            img = torch.from_numpy(img_np).float()
            # (H,W,1) 형태 → (1,H,W)로 채널 차원만 고정
            img = img.permute(2, 0, 1)  # (1,26,34)

        # 3) 레이블은 NumPy → Tensor 변환
        label = torch.from_numpy(label_np).float()  # shape: (1,)

        return img, label

    def __len__(self):
        return len(self.x_data)
