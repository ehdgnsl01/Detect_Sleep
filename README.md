# 졸음 감지 프로그램 (Drowsiness Detection)
참고자료 : [https://github.com/kairess/eye_blink_detector](https://github.com/kairess/eye_blink_detector)<br/>
참고자료2 : [https://github.com/yunseokddi/Pytorch_dev/tree/master/sleep_detect]( https://github.com/yunseokddi/Pytorch_dev/tree/master/sleep_detect)

## 개요
이 코드는 OpenCV, dlib, PyTorch 기반의 CNN 모델을 이용해 웹캠 또는 저장된 영상에서 ‘눈 감김’과 ‘고개 젖’을 실시간으로 감지함.  
30프레임(약 1초) 이상 연속으로 눈이 감기거나 고개가 정면에서 벗어나면 화면 왼쪽 상단에 “Wake up” 메시지를 출력하고,  
90프레임(약 3초) 이상 연속일 때는 화면 중앙에 “SLEEP WARNING” 경고 텍스트와 깜빡이는 빨간 테두리를 표시함.


## 주요 기능
1. **얼굴과 랜드마크 검출**  
   - OpenCV Haar Cascade (`haarcascade_frontalface_alt.xml`)로 얼굴 영역을 빠르게 검출  
   - dlib 랜드마크 모델 (`shape_predictor_68_face_landmarks.dat`)로 눈과 코, 턱 등 68개 지점을 추출  

2. **눈 감김 판단**  
   - CNN 모델 (`model.py`에 정의된 `Net` 클래스)의 출력으로 눈이 감긴(0) 상태인지 뜬(1) 상태인지 구분  
   - 양쪽 눈이 모두 감긴 경우 해당 프레임을 ‘감김 프레임’으로 처리하여 연속 카운터(`n_eye_closed`) 증가  

3. **고개 기울기(pitch) 계산**  
   - 얼굴 랜드마크 6개 지점(코끝, 턱, 양쪽 눈 모서리, 양쪽 입 모서리)을 3D 좌표에 매핑하여 `cv2.solvePnP`로 회전 벡터 획득  
   - 보정 로직을 통해 “정면(pitch ≈ 0°)”에 가깝게 매핑하고, 일정 범위 이상 기울어지면 ‘고개 off 프레임’으로 처리하여 연속 카운터(`n_head_off`) 증가  

4. **단계별 경고 로직**  
   - **30프레임 초과**: 눈 또는 고개 off 카운터가 30을 넘으면 화면 왼쪽 상단에 “Wake up” 텍스트를 빨간색으로 출력  
   - **90프레임 초과**: 위 상태가 90프레임을 넘으면 화면 중앙에 “SLEEP WARNING” 텍스트(빨간, 굵게)를 띄우고, 0.5초 간격으로 깜빡이는 빨간 테두리를 표시  

5. **실시간 정보 표시**  
   - **좌우 눈 이미지를 별도 창(`Left Eye`, `Right Eye`)에 확대**하여 눈 상태(Open/Closed)와 함께 표시  
   - **오른쪽 상단에 현재 보정된 pitch(° 단위) 값을 실시간으로 출력**  

## 디렉터리 구조
```
project_root/
├─ data/
│ ├─ video1.mp4 # 입력 영상 파일
│ ├─ video2.mp4 # 입력 영상 파일
│ ├─ x_train.npy
│ ├─ x_test.npy
│ ├─ y_train.npy
│ ├─ y_test.npy
│ └─ haarcascade_frontalface_alt.xml
├─ weights/
│ └─ classifier_weights_iter_20.pth
├─ data_loader.py
├─ model.py # CNN 모델 정의
├─ train.py
├─ test.py
├─ detect_sleep.py # 메인 실행 스크립트
├─ shape_predictor_68_face_landmarks.dat
└─ README.md
```

## 파일 간단 설명
### 1. model.py  
- **목적**: 눈 이미지(CNN 입력)를 받아 “눈이 감겼는지(Open/Closed)”를 판단하는 신경망(Convolutional Neural Network) 정의  
- **주요 클래스**  
  - `Net(nn.Module)`  
    - 입력: `(batch, 1, 26, 34)` 크기의 흑백 눈 이미지  
    - 구조:  
      1. `conv1(1→32, 3×3) → ReLU → MaxPool(2×2)` → 출력 크기 `(batch, 32, 13, 17)`  
      2. `conv2(32→64, 3×3) → ReLU → MaxPool(2×2)` → 출력 크기 `(batch, 64, 6, 8)`  
      3. `conv3(64→128, 3×3) → ReLU → MaxPool(2×2)` → 출력 크기 `(batch, 128, 3, 4)`  
      4. Flatten → Fully-connected `fc1(128×3×4=1536 → 512) → ReLU` → `fc2(512 → 1)`  
    - 출력: `(batch, 1)` 크기의 로짓 값, 시그모이드(sigmoid) 후 반올림(round)하여 0(Closed) 또는 1(Open)으로 분류  
- **테스트 코드** (`if __name__ == "__main__"`):  
  - `torchsummary.summary(model, (1, 26, 34))`를 실행해 레이어별 출력 크기와 파라미터 개수 확인  

### 2. data_loader.py  
- **목적**: NumPy로 저장된 눈 데이터(`.npy`)와 레이블을 PyTorch `Dataset` 형태로 감싸는 클래스 제공  
- **주요 클래스**  
  - `eyes_dataset(Dataset)`  
    - 생성자 인자  
      - `x_data`: NumPy 배열, shape = `(N, 26, 34, 1)` (눈 이미지 데이터)  
      - `y_data`: NumPy 배열, shape = `(N, 1)` (0 또는 1 레이블)  
      - `transform`: `torchvision.transforms`에서 제공하는 optional transform (예: `ToTensor`, `RandomRotation` 등)  
    - `__getitem__(self, idx)`  
      1. NumPy로 저장된 `(26, 34, 1)` 이미지를 불러와 `transform`이 있으면 적용 → `(1, 26, 34)` Tensor로 변환  
      2. 레이블 `(1,)` NumPy → Tensor(float) 변환  
      3. `(img_tensor, label_tensor)` 반환  
    - `__len__(self)`: 전체 데이터 개수(`len(x_data)`) 반환  


### 3. train.py  
- **목적**: `data_loader.py`의 `eyes_dataset`과 `model.py`의 `Net`을 이용해 눈 감김 분류기를 학습하고 가중치를 저장  
- **주요 기능**  
  1. **데이터 불러오기**  
     - `data/x_train.npy`, `data/y_train.npy` (NumPy float32)  
  2. **데이터 증강 (transform)**  
     - `transforms.ToTensor()` → `(26, 34, 1)` NumPy → `(1, 26, 34)` Tensor  
     - `RandomRotation(10)`, `RandomHorizontalFlip()` 추가  
  3. **Dataset & DataLoader 생성**  
     - `train_dataset = eyes_dataset(x_train, y_train, transform=…)`  
     - `train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)`  
  4. **모델·손실함수·옵티마이저 설정**  
     - `model = Net().to(device)`  
     - `criterion = BCEWithLogitsLoss()` (시그모이드+이진 크로스엔트로피)  
     - `optimizer = Adam(model.parameters(), lr=1e-4)`  
  5. **학습 루프** (`epochs = 50`)  
     - 매 배치마다  
       - `images, labels` → GPU/CPU 이동  
       - `optimizer.zero_grad()` → `outputs = model(images)` → `loss = criterion(outputs, labels)` → `loss.backward()` → `optimizer.step()`  
       - 누적 손실(`running_loss`)과 정확도(`running_acc`) 계산 (`accuracy()` 함수 사용)  
       - 80개 배치마다 평균 손실 및 평균 정확도 출력  
  6. **학습 완료 후 가중치 저장**  
     - `torch.save(model.state_dict(), "weights/classifier_weights_iter_20.pth")`  

- **accuracy 함수**  
  ```python
  def accuracy(y_pred, y_true):
      prob = torch.sigmoid(y_pred)
      pred_tag = torch.round(prob)
      correct = (pred_tag == y_true).sum().float()
      return (correct / y_true.size(0)) * 100
```
#train.py 실행결과
epoch: [1/50] train_loss: 0.30973 train_acc: 88.43750
epoch: [1/50] train_loss: 0.43647 train_acc: 84.61539
epoch: [2/50] train_loss: 0.15871 train_acc: 94.68750
epoch: [2/50] train_loss: 0.05007 train_acc: 100.00000
epoch: [3/50] train_loss: 0.10742 train_acc: 96.79688
epoch: [3/50] train_loss: 0.19546 train_acc: 96.15385
.
.
.
epoch: [48/50] train_loss: 0.00837 train_acc: 99.68750
epoch: [48/50] train_loss: 0.00016 train_acc: 100.00000
epoch: [49/50] train_loss: 0.00982 train_acc: 99.60938
epoch: [49/50] train_loss: 0.00060 train_acc: 100.00000
epoch: [50/50] train_loss: 0.01316 train_acc: 99.60938
epoch: [50/50] train_loss: 0.00691 train_acc: 100.00000
learning finish
```
<img src="https://github.com/user-attachments/assets/dba3c931-1420-4a31-a1ea-753a9366723b" height=500>

### 4. test.py  
- **목적**: 학습된 가중치를 로드해 검증(테스트) 데이터에 대한 정확도를 평가하고, 샘플 이미지를 시각화
- **주요 기능**  
  1. **검증 데이터 불러오기**  
     - `data/x_val.npy`, `data/y_val.npy` (NumPy float32)  
  2. **Dataset & DataLoader**  
     - `test_transform = transforms.Compose([ToTensor()])`  
     - `test_dataset = eyes_dataset(x_val, y_val, transform=test_transform)`
     - `test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)`(Windows 환경이라 `num_workers=0`)
  3. **모델 로드 및 평가 모드**  
     - `model = Net().to(device)`  
     - `model.load_state_dict(torch.load(weights_path))`
     - `model.eval()`
  4. **검증 루프** (`with torch.no_grad():`)  
     - 매 배치마다  
       - `outputs = model(images)` → `accuracy(outputs, labels)` 누  
       - 전체 배치 평균 정확도 출력
```
#test.py 실행결과
average acc: 99.65278 %
test finish!
```
<img src="https://github.com/user-attachments/assets/11132449-0271-4fc9-9772-d1e16ebf37b5" height=500>

### 5. detect_sleep.py  
- **목적**: 웹캠(또는 저장된 영상)에서 **눈 감김**과 **고개 기울기**(피치)를 실시간으로 감지, 일정 프레임 이상 연속 감지되면 단계별로 경고 메시지 출력
- **주요 기능**  
  1. **얼굴 검출 & 랜드마크 추출**
     - OpenCV Haar Cascade로 얼굴 영역 검출  
     - dlib 랜드마크 모델로 68개 포인트(눈, 코, 턱, 입 등) 추출  
  2. **눈 감김 판단**
     - CNN 모델(`model.py`)에 `(1×1×26×34)` 크기 눈 이미지를 입력  
     - 모델 출력(0=Closed, 1=Open)으로 양쪽 눈 모두 감긴 경우 감김 프레임 카운터 증가
  3. **고개 기울기(pitch) 계산**
     - 6개 랜드마크(코끝, 턱, 양쪽 눈 모서리, 양쪽 입 모서리)로 3D PnP 수행  
     - `cv2.solvePnP` → Euler 각도(pitch, yaw, roll) 추출 → ±90° 보정  
     - 보정된 pitch가 “정면 범위(−10° ~ +7°)”를 벗어나면 고개 off 프레임 카운터 증가
  4. **단계별 경고 로직**
     - **30프레임(약 1초) 연속** 눈 감김 또는 고개 off → 화면 왼쪽 상단에 **“Wake up”**  
     - **90프레임(약 3초) 연속** 눈 감김 또는 고개 off → 화면 중앙에 **“SLEEP WARNING”**  
        - 빨간색 굵은 텍스트  
        - 0.5초 간격으로 깜빡이는 빨간 테두리
  5. **실시간 정보 표시**
     - 왼쪽/오른쪽 눈을 각각 확대된 창(200×150)으로 표시하고, 상태(Open/Closed) 텍스트 출력  
     - 화면 오른쪽 상단에 보정된 **Pitch: xx.x°** 실시간 출력
- **동작 흐름**
  1. **영상 열기**
     - `VIDEO_PATH`가 존재하면 동영상 재생, 아니면 웹캠(0번 장치) 사용  
  2. **프레임 처리**
     1. (웹캠일 경우) 좌우 반전  
     2. 절반 크기로 축소  
     3. 그레이스케일 변환  
     4. 얼굴 검출 → 각 얼굴마다:
        - 랜드마크 추출 → PnP로 pitch 계산 → 보정  
        - 눈 영역 Crop → CNN 예측 → 감김 여부 결정  
        - 화면에 랜드마크 점, 눈 사각형, 상태 텍스트 그리기  
        - 좌우 눈을 별도 창에 확대 표시  
     5. **연속 카운터 업데이트**
        - 눈 감김 연속 프레임(`n_eye_closed`), 고개 off 연속 프레임(`n_head_off`)  
     6. **경고 조건 검사**
        - `n_eye_closed > 30` 또는 `n_head_off > 30` → “Wake up” 출력  
        - `n_eye_closed > 90` 또는 `n_head_off > 90` → 중앙 “SLEEP WARNING” + 깜빡이는 빨간 테두리  
     7. 보정된 pitch 값을 화면 오른쪽 상단(메시지 아래)에 출력  
     8. ESC 키(27) 입력 시 반복 종료 → 자원 해제
- **주요 변수**
  - **`EYE_CLOSED_FRAME_THRESHOLD = 30`**  
     - 눈 감김 연속 30프레임 초과 시 첫 단계 경고
  - **`EYE_WARNING_FRAME_THRESHOLD = 90`**  
     - 눈 감김 연속 90프레임 초과 시 최종 경고
  - **`HEAD_PITCH_DOWN_THRESHOLD = -10`, `HEAD_PITCH_UP_THRESHOLD = 7`**  
     - Pitch 보정 후 “정면” 범위
  - **`HEAD_OFF_FRAME_THRESHOLD = 30`**, **`HEAD_WARNING_FRAME_THRESHOLD = 90`**  
     - 고개 off 연속 30/90프레임 임계
   
- **실행 사진**
<img src="https://github.com/user-attachments/assets/33fa00ff-a450-4974-9366-604b33d1dc37" width=400>
<img src="https://github.com/user-attachments/assets/4f8e8369-c845-479f-ab7c-9d7dde087dc1" width=400><br/>
왼 : 평소 (눈이 안감기고, 고개도 안젖힐때) , 오 : 졸음 감지 (눈을 감은지 1초가 지났을 때)<br/>

<img src="https://github.com/user-attachments/assets/ab22edad-6666-4624-989c-7ebeb514bb9e" width=400>
<img src="https://github.com/user-attachments/assets/05845430-1e44-4f2c-af91-89928361de6a" width=400><br/>
왼 : 평소 (눈이 안감기고, 고개도 안젖힐때) , 오 : 졸음 감지 (눈을 감은지 1초가 지났을 때)<br/>

<img src="https://github.com/user-attachments/assets/79e83875-a6c7-4737-a393-5fec565a1c62" width=400>
<img src="https://github.com/user-attachments/assets/f5880454-29ca-424d-8e24-f2bd23699d5c" width=400><br/>
왼 : 졸음 감지 (고개를 젖힌지 1초가 지났을 때) , 오 : 졸음 위험 (고개를 젖힌지 3초가 지났을 때)<br/>

<img src="https://github.com/user-attachments/assets/92a5c80c-dc38-4037-9d18-cbd2143d072c" width=400><br/>
좌우 눈 별도 창에 확대 표시<br/>
