# 졸음 감지 프로그램 (Drowsiness Detection)
참고자료 : [https://github.com/kairess/eye_blink_detector](https://github.com/kairess/eye_blink_detector)

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

project_root/
├─ data/
│ ├─ video1.mp4 # 입력 영상 파일
│ ├─ video2.mp4 # 입력 영상 파일
│ ├─ x_train.npy
│ ├─ x_test.npy
│ ├─ y_train.npy
│ ├─ y_test.npy
│ └─ haarcascade_frontalface_alt.xml
├─ model.py # CNN 모델 정의
├─ detect_drowsiness.py # 메인 실행 스크립트 (아래 코드)
├─ shape_predictor_68_face_landmarks.dat
└─ weights/
└─ classifier_weights_iter_20.pth
