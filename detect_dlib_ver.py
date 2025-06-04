# detect_dlib_ver.py

import os
import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils

# -----------------------------
# 설정 파라미터
# -----------------------------
IMG_SIZE = (34, 26)  # (width, height) for model input
WEIGHTS_PATH = os.path.join('weights', 'classifier_weights_iter_20.pth')
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

# head down 판정 기준 (pitch 각도의 절댓값 기준)
HEAD_PITCH_THRESHOLD = 15.0  # degree, 고개가 이만큼 아래로 숙여질 때
HEAD_FRAME_LIMIT = 60       # 프레임 수: 60프레임(약 3초) 연속이면 졸음

# -----------------------------
# 1. 디바이스 및 모델 초기화
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

model = Net().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

n_eye_closed = 0    # 연속 눈 감김 프레임 카운터
n_head_down = 0     # 연속 고개 숙임 프레임 카운터

# Facial 3D 모델 포인트 (mm 단위 임의 값, 일반적으로 사용되는 값)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip (index 30)
    (0.0, -330.0, -65.0),      # Chin (index 8)
    (-225.0, 170.0, -135.0),   # Left eye left corner (index 36)
    (225.0, 170.0, -135.0),    # Right eye right corner (index 45)
    (-150.0, -150.0, -125.0),  # Left Mouth corner (index 48)
    (150.0, -150.0, -125.0)    # Right mouth corner (index 54)
], dtype=np.float32)

def get_head_pose(shape, size):
    """
    dlib shape(np.ndarray 68x2) 와 이미지 크기(size=(h, w))를 받아
    head pose의 pitch, yaw, roll 각도를 구함.
    """
    image_points = np.array([
        tuple(shape[30]),    # Nose tip
        tuple(shape[8]),     # Chin
        tuple(shape[36]),    # Left eye left corner
        tuple(shape[45]),    # Right eye right corner
        tuple(shape[48]),    # Left Mouth corner
        tuple(shape[54])     # Right mouth corner
    ], dtype=np.float32)

    h, w = size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # 왜곡 계수: 0으로 가정

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    # rotation_vector -> rotation_matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    # euler_angles: [pitch, yaw, roll] in degrees
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll

def crop_eye(gray_img, eye_points):
    """
    눈 랜드마크(6개 포인트)로부터 눈 영역을 크롭
    Args:
        gray_img (np.ndarray): 그레이스케일 이미지
        eye_points (ndarray): shape (6,2) 눈 랜드마크 좌표
    Returns:
        eye_img (np.ndarray): 크롭된 눈 이미지
        eye_rect (list[int]): [min_x, min_y, max_x, max_y] 좌표
    """
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
    """
    eye_tensor: torch.Tensor, shape (1,1,26,34)
    Returns: torch.Tensor 예측 클래스 (0=Closed, 1=Open)
    """
    with torch.no_grad():
        eye = eye_tensor.to(device)
        outputs = model(eye)
        pred_tag = torch.round(torch.sigmoid(outputs))
    return pred_tag

def main():
    global n_eye_closed, n_head_down

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam을 열 수 없습니다.")
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

        # 좌우 반전(미러)
        frame = cv2.flip(frame, 1)
        # 프레임 절반 크기로 축소
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)

        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            # 얼굴 바운딩 박스 그리기(초기에는 흰색)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_color = (255, 255, 255)

            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            # 1) Head pose 계산 (pitch, yaw, roll)
            pitch, yaw, roll = get_head_pose(shapes, gray.shape)
            # pitch가 threshold 이상 아래로 숙여지면 head down 카운터 증가
            if pitch > HEAD_PITCH_THRESHOLD:
                n_head_down += 1
            else:
                n_head_down = 0

            # head down 상태라면 face bounding box 색 빨강으로 변경
            if n_head_down > HEAD_FRAME_LIMIT:
                face_color = (0, 0, 255)
                cv2.putText(img, "Wake up (Head)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 얼굴 bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), face_color, 2)

            # --- 기존 eye detection & drowsiness logic ---
            left_eye_pts = shapes[36:42]
            right_eye_pts = shapes[42:48]

            eye_img_l, eye_rect_l = crop_eye(gray, left_eye_pts)
            eye_img_r, eye_rect_r = crop_eye(gray, right_eye_pts)

            eye_img_l = cv2.resize(eye_img_l, IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, 1)

            eye_np_l = eye_img_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            eye_np_r = eye_img_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            eye_tensor_l = torch.from_numpy(eye_np_l).permute(0, 3, 1, 2)
            eye_tensor_r = torch.from_numpy(eye_np_r).permute(0, 3, 1, 2)

            pred_l = predict_eye_state(eye_tensor_l)
            pred_r = predict_eye_state(eye_tensor_r)

            # 연속 눈 감김 카운터
            if pred_l.item() == 0.0 and pred_r.item() == 0.0:
                n_eye_closed += 1
            else:
                n_eye_closed = 0

            # 눈 감김 기준으로 졸음 판단
            if n_eye_closed > 100:
                cv2.putText(img, "Wake up (Eye)", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 눈 사각형 & 상태 표시
            state_l = 'Open' if pred_l.item() == 1 else 'Closed'
            state_r = 'Open' if pred_r.item() == 1 else 'Closed'

            cv2.rectangle(img, (eye_rect_l[0], eye_rect_l[1]),
                          (eye_rect_l[2], eye_rect_l[3]), (255, 255, 255), 1)
            cv2.rectangle(img, (eye_rect_r[0], eye_rect_r[1]),
                          (eye_rect_r[2], eye_rect_r[3]), (255, 255, 255), 1)
            cv2.putText(img, state_l, (eye_rect_l[0], eye_rect_l[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, state_r, (eye_rect_r[0], eye_rect_r[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 작은 창: 확대된 눈 이미지 + 상태 표시
            eye_small_l = cv2.resize(eye_img_l, (200, 150), interpolation=cv2.INTER_NEAREST)
            eye_small_l = cv2.cvtColor(eye_small_l, cv2.COLOR_GRAY2BGR)
            cv2.putText(eye_small_l, state_l, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if pred_l.item() == 1 else (0, 0, 255), 1)
            cv2.imshow('Left Eye', eye_small_l)

            eye_small_r = cv2.resize(eye_img_r, (200, 150), interpolation=cv2.INTER_NEAREST)
            eye_small_r = cv2.cvtColor(eye_small_r, cv2.COLOR_GRAY2BGR)
            cv2.putText(eye_small_r, state_r, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if pred_r.item() == 1 else (0, 0, 255), 1)
            cv2.imshow('Right Eye', eye_small_r)

        # 메인 창 표시
        cv2.imshow('Drowsiness Detection', img)

        # ESC 키(27) 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
