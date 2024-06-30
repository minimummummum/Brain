##############영상 읽어 오는 거를 함수로 해서 임시 버튼 누르면 실행되도록, 모든 영상을 다 저장할 거야?에 대한 해결책도.
import cv2

memory_count = 1
cap = cv2.VideoCapture(0)

if cap.isOpened():
    file_path = f'memory/memory_piece{memory_count}.avi'
    fps = 25.40
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    out = None
    recording = False

    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('cam', img)
            # ESC 키를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            # 스페이스바를 눌러 녹화 시작/종료
            elif key == 32:  # 스페이스바
                if not recording:
                    file_path = f'memory/memory_piece{memory_count}.avi'
                    out = cv2.VideoWriter(file_path, fourcc, fps, size)
                    recording = True
                    print(f"Recording started: {file_path}")
                else:
                    out.release()
                    recording = False
                    memory_count += 1
                    print(f"Recording stopped: {file_path}")

            # 녹화 중일 때 프레임 저장
            if recording:
                out.write(img)

        else:
            print("No frame")
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("No camera")
