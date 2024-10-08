import cv2

cap = cv2.VideoCapture("memory/memory_piece2.avi")

# SIFT 생성
sift = cv2.SIFT_create()

if cap.isOpened():
    fps = 25.40
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

        if cv2.waitKey(int(1000/fps)) != -1:
            break
else:
    print("no camera")

cap.release()
cv2.destroyAllWindows()
