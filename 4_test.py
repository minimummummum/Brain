import cv2
import numpy as np

# SIFT 설정
sift = cv2.SIFT_create()

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 물체 이미지 초기화
object_image = None
kp1, des1 = None, None

def capture_object_image():
    global object_image, kp1, des1
    ret, frame = cap.read()
    if ret:
        object_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(object_image, None)
        cv2.imshow('Captured Object', object_image)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 스페이스바를 눌러 물체 이미지 캡처
    if cv2.waitKey(1) & 0xFF == ord(' '):
        capture_object_image()
    
    # 물체 이미지가 지정된 경우 매칭 수행
    if object_image is not None:
        # 현재 프레임에서 키포인트와 디스크립터 찾기
        kp2, des2 = sift.detectAndCompute(gray_frame, None)
        
        if des1 is not None and des2 is not None:
            # FLANN 기반 매칭 설정
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # 매칭 수행
            matches = flann.knnMatch(des1, des2, k=2)
            
            # 좋은 매칭만을 사용
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # 매칭 결과를 그리기
            frame_matches = cv2.drawMatches(object_image, kp1, gray_frame, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Object Detection', frame_matches)
    
    cv2.imshow('Webcam', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()