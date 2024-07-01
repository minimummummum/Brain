import cv2

# 노트북 카메라를 사용하여 VideoCapture 객체 생성
cap = cv2.VideoCapture(0)  # 0은 노트북 내장 카메라를 의미, 다른 카메라일 경우 1, 2 등으로 변경 가능

# SIFT 생성
sift = cv2.SIFT_create()

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("no frame")
            break
        
        # 노이즈 제거 (가우시안 블러)
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 그레이스케일로 변환
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

        # SIFT로 특징점 검출 및 기술자 계산
        keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

        # 특징점 그리기
        frame_with_keypoints = cv2.drawKeypoints(blurred_frame, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 화면에 표시
        cv2.imshow('frame with keypoints', frame_with_keypoints)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("no camera")

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
