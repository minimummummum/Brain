import numpy as np
import cv2

# 저장된 파일 불러오기
loaded_data = np.load('videos.npz', allow_pickle=True)
min_size = 100000000
videos = []
for file in loaded_data.files:
    videos.append(loaded_data[file])
    min_size = min(min_size, len(loaded_data[file]))

# 두 개의 비디오를 동시에 재생
num_frames = min_size
for i in range(num_frames):
    frame = []
    for video in videos:
        frame.append(video[i])
    # 두 프레임을 나란히 붙이기 (수평 스택)
    combined_frame = np.hstack(frame)

    # 화면에 표시
    cv2.imshow('Combined Video', combined_frame)

    if cv2.waitKey(int(1000 / 25.40)) != -1:  # 25.40 FPS 가정
        break

cv2.destroyAllWindows()