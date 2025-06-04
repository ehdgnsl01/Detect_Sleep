import os
import cv2
import dlib
import numpy as np
import torch
import time
from imutils import face_utils
from model import Net

# -----------------------------
# 설정 파라미터
# -----------------------------
IMG_SIZE = (34, 26)  # 모델 입력용 눈 크기 (width, height)
WEIGHTS_PATH = os.path.join('weights', 'classifier_weights_iter_20.pth')
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_CASCADE_PATH = 'data/haarcascade_frontalface_alt.xml'  # Haar Cascade 경로
VIDEO_PATH = 'data/video2.mp4'  # 저장된 영상 경로 (없으면 웹캠 사용)

# 눈 감김 연속 판단 임계
EYE_CLOSED_FRAME_THRESHOLD = 30
# 90프레임 이상 지속 시 SLEEP WARNING
EYE_WARNING_FRAME_THRESHOLD = 90

# 고개 정면이 아닐 때 판정 기준(사용하지 않음, 오직 pitch 표시용)
HEAD_PITCH_DOWN_THRESHOLD = -10
HEAD_PITCH_UP_THRESHOLD = 7
HEAD_OFF_FRAME_THRESHOLD = 30
HEAD_WARNING_FRAME_THRESHOLD = 90

# -----------------------------
# 1. 모델 및 검출기 초기화
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 모델 로드
model = Net().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# dlib 랜드마크 예측기
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# OpenCV Haar Cascade 얼굴 검출기
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(FACE_CASCADE_PATH)):
    print(f"Error loading Haar Cascade from {FACE_CASCADE_PATH}")
    exit(0)

# 연속 눈 감김 카운터
n_eye_closed = 0
# 연속 고개 숙임 카운터
n_head_off = 0

# Facial 3D 모델 포인트 (mm 단위)
MODEL_POINTS = np.array([
    (0.0,    0.0,   0.0),     # 코끝 (index 30)
    (0.0,  -330.0, -65.0),    # 턱 (index 8)
    (-225.0, 170.0, -135.0),  # 왼쪽 눈 왼쪽 모서리 (36)
    (225.0,  170.0, -135.0),  # 오른쪽 눈 오른쪽 모서리 (45)
    (-150.0, -150.0, -125.0), # 왼쪽 입 모서리 (48)
    (150.0,  -150.0, -125.0)  # 오른쪽 입 모서리 (54)
], dtype=np.float32)

# -----------------------------
# 2. 헬퍼 함수들
# -----------------------------
def get_head_pose(shape, img_shape):
    image_points = np.array([
        tuple(shape[30]),  # 코끝
        tuple(shape[8]),   # 턱
        tuple(shape[36]),  # 왼눈 왼쪽 모서리
        tuple(shape[45]),  # 오른눈 오른쪽 모서리
        tuple(shape[48]),  # 왼쪽 입 모서리
        tuple(shape[54])   # 오른쪽 입 모서리
    ], dtype=np.float32)

    h, w = img_shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll

def crop_eye(gray_img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * (IMG_SIZE[1] / IMG_SIZE[0])
    margin_x, margin_y = w / 2, h / 2

    min_x = int(cx - margin_x)
    min_y = int(cy - margin_y)
    max_x = int(cx + margin_x)
    max_y = int(cy + margin_y)

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, gray_img.shape[1])
    max_y = min(max_y, gray_img.shape[0])

    eye_rect = [min_x, min_y, max_x, max_y]
    eye_img = gray_img[min_y:max_y, min_x:max_x]
    return eye_img, eye_rect

def predict_eye_state(eye_tensor):
    with torch.no_grad():
        eye = eye_tensor.to(device)
        outputs = model(eye)
        pred = torch.round(torch.sigmoid(outputs))
    return int(pred.item())

# -----------------------------
# 3. 메인 루프
# -----------------------------
def main():
    global n_eye_closed, n_head_off

    # 저장 비디오가 존재하면 파일 재생, 아니면 웹캠
    if os.path.isfile(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        is_webcam = False
    else:
        cap = cv2.VideoCapture(0)
        is_webcam = True

    if not cap.isOpened():
        print("영상 소스를 열 수 없습니다.")
        return

    cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drowsiness Detection', 800, 600)
    cv2.namedWindow('Left Eye', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Left Eye', 200, 150)
    cv2.namedWindow('Right Eye', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Right Eye', 200, 150)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 웹캠일 때만 좌우 미러
        if is_webcam:
            frame = cv2.flip(frame, 1)
        # 공통: 절반 크기로 축소
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)

        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        any_closed_this_frame = False
        any_head_off_this_frame = False
        current_pitch = 0.0

        for (x, y, w, h) in faces:
            # 얼굴 바운딩 박스 (기본 흰색)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # dlib용 rectangle 객체
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shapes = predictor(gray, rect)
            shapes = face_utils.shape_to_np(shapes)

            # 1) Head pose 계산 (pitch 값 저장)
            pitch_raw, yaw, roll = get_head_pose(shapes, gray.shape)
            if pitch_raw > 90:
                pitch = pitch_raw - 180
            elif pitch_raw < -90:
                pitch = pitch_raw + 180
            else:
                pitch = pitch_raw
            current_pitch = pitch  # 가장 마지막 얼굴의 pitch이지만, 얼굴이 하나만 있다고 가정
            # 정면 기준: -15 ≤ pitch ≤ +15, 아닐 경우 n_head_off 카운터 증가
            if pitch < HEAD_PITCH_DOWN_THRESHOLD or pitch > HEAD_PITCH_UP_THRESHOLD:
                any_head_off_this_frame = True

            # 2) 눈 감김 판단
            left_eye_pts  = shapes[36:42]
            right_eye_pts = shapes[42:48]

            # 눈 영역 시각화(landmark 점 찍기)
            for pt in np.concatenate((left_eye_pts, right_eye_pts), axis=0):
                cv2.circle(img, tuple(pt), 1, (0, 255, 255), -1)

            # 왼쪽 눈 크롭 → CNN 예측
            eye_img_l, eye_rect_l = crop_eye(gray, left_eye_pts)
            eye_img_l = cv2.resize(eye_img_l, IMG_SIZE)
            eye_tensor_l = torch.from_numpy(
                eye_img_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            ).permute(0, 3, 1, 2)
            pred_l = predict_eye_state(eye_tensor_l)

            # 오른쪽 눈 크롭 → CNN 예측 (좌우 반전)
            eye_img_r, eye_rect_r = crop_eye(gray, right_eye_pts)
            eye_img_r = cv2.resize(eye_img_r, IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, 1)
            eye_tensor_r = torch.from_numpy(
                eye_img_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            ).permute(0, 3, 1, 2)
            pred_r = predict_eye_state(eye_tensor_r)

            # 이 얼굴의 눈이 둘 다 Closed면 any_closed_this_frame = True
            if pred_l == 0 and pred_r == 0:
                any_closed_this_frame = True

            # 눈 영역 사각형 + 상태 텍스트
            state_l = 'Open' if pred_l == 1 else 'Closed'
            state_r = 'Open' if pred_r == 1 else 'Closed'
            cv2.rectangle(img, (eye_rect_l[0], eye_rect_l[1]),
                          (eye_rect_l[2], eye_rect_l[3]), (255, 255, 255), 1)
            cv2.rectangle(img, (eye_rect_r[0], eye_rect_r[1]),
                          (eye_rect_r[2], eye_rect_r[3]), (255, 255, 255), 1)
            cv2.putText(img, state_l, (eye_rect_l[0], eye_rect_l[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, state_r, (eye_rect_r[0], eye_rect_r[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 작은 창: 확대된 눈 + 상태 표시
            eye_small_l = cv2.resize(eye_img_l, (200, 150), interpolation=cv2.INTER_NEAREST)
            eye_small_l = cv2.cvtColor(eye_small_l, cv2.COLOR_GRAY2BGR)
            cv2.putText(eye_small_l, state_l, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if pred_l == 1 else (0, 0, 255), 1)
            cv2.imshow('Left Eye', eye_small_l)

            eye_small_r = cv2.resize(eye_img_r, (200, 150), interpolation=cv2.INTER_NEAREST)
            eye_small_r = cv2.cvtColor(eye_small_r, cv2.COLOR_GRAY2BGR)
            cv2.putText(eye_small_r, state_r, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if pred_r == 1 else (0, 0, 255), 1)
            cv2.imshow('Right Eye', eye_small_r)

        # 프레임 내 얼굴 중 하나라도 눈 감김이면 연속 카운터 증가, 아니면 리셋
        if any_closed_this_frame:
            n_eye_closed += 1
        else:
            n_eye_closed = 0

        # 프레임 내 얼굴 중 하나라도 정면이 아닐 경우 카운터 증가, 아니면 리셋
        if any_head_off_this_frame:
            n_head_off += 1
        else:
            n_head_off = 0

        h, w = img.shape[:2]

        # “눈이나 고개”가 30프레임 초과 시
        if n_eye_closed > EYE_CLOSED_FRAME_THRESHOLD or n_head_off > HEAD_OFF_FRAME_THRESHOLD:
            # “눈이나 고개”가 90프레임 초과 시 → 중앙 “SLEEP WARNING” + 깜빡이는 빨간 테두리
            if n_eye_closed > EYE_WARNING_FRAME_THRESHOLD or n_head_off > EYE_WARNING_FRAME_THRESHOLD:
                # 중앙 텍스트
                warning_text = "SLEEP WARNING"
                txt_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                x_txt = (w - txt_size[0]) // 2
                y_txt = (h + txt_size[1]) // 2
                cv2.putText(img, warning_text, (x_txt, y_txt),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # 깜빡이는 테두리: 0.5초 간격으로 토글
                if int(time.time() * 2) % 2 == 0:
                    thickness = 8
                    color = (0, 0, 255)
                    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)
            else:
                # 30프레임 초과이지만 90프레임 이하는 → “Wake up” 메시지
                cv2.putText(img, "Wake up", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 보정된 pitch를 오른쪽 상단(메시지 바로 아래)에 실시간 표시
        pitch_text = f"Pitch: {current_pitch:.1f}"
        txt_sz = cv2.getTextSize(pitch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x_txt2 = w - txt_sz[0] - 10
        cv2.putText(img, pitch_text, (x_txt2, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 결과 창 표시
        cv2.imshow('Drowsiness Detection', img)

        # ESC(27) 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
