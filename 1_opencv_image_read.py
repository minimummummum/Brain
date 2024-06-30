import cv2
memory_count = 1

cap = cv2.VideoCapture(0)
if cap.isOpened():
    file_path = f'memory/memory_piece{memory_count}.avi'
    fps = 25.40
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    out = cv2.VideoWriter(file_path, fourcc, fps, size)
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('cam', img)
            out.write(img)
            if cv2.waitKey(int(1000/fps)) != -1:
                break
        else:
            print("no frame")
            break
    out.release()
else:
    print("no camera")
cap.release()
cv2.destroyAllWindows

