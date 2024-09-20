import cv2
import mediapipe as mp

# MediaPipe 포즈 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 포즈 추정
    results = pose.process(image)
    
    # 이미지 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 신체 랜드마크 그리기
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 결과 화면 출력
    cv2.imshow('Pose Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
