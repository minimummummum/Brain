import cv2

cap = cv2.VideoCapture("memory/test.avi")
if cap.isOpened():
    fps = 25.40
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('cam', img)
            if cv2.waitKey(int(1000/fps)) != -1:
                break
        else:
            print("no frame")
            break
else:
    print("no camera")
cap.release()
cv2.destroyAllWindows

