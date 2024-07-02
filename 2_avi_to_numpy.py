import numpy as np
import cv2
import os
def video_to_numpy(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)

# memory 폴더 경로 설정
memory_folder = "memory/"

# memory 폴더의 모든 영상 파일에 대해 매칭 수행
video_files = []
for filename in os.listdir(memory_folder):
    if filename.endswith(".avi"):
        video_files.append(os.path.join(memory_folder, filename))


# 영상 파일 경로
video1_path = "memory/memory_piece2.avi"
video2_path = "memory/test.avi"

# 영상 파일을 numpy 배열로 변환
video1_data = video_to_numpy(video1_path)
video2_data = video_to_numpy(video2_path)

if video1_data is not None and video2_data is not None:
    # numpy 배열을 하나의 파일에 저장
    np.savez_compressed('videos.npz', video1=video1_data, video2=video2_data)

    # 저장된 파일 불러오기
    loaded_data = np.load('videos.npz', allow_pickle=True)
    video1_loaded = loaded_data['video1']
    video2_loaded = loaded_data['video2']

    # 데이터 사용 예시
    print("Loaded video 1 shape:", video1_loaded.shape)
    print("Loaded video 2 shape:", video2_loaded.shape)

    # 두 개의 비디오를 동시에 재생
    num_frames = min(len(video1_loaded), len(video2_loaded))

    for i in range(num_frames):
        frame1 = video1_loaded[i]
        frame2 = video2_loaded[i]

        # 두 프레임을 나란히 붙이기 (수평 스택)
        combined_frame = np.hstack((frame1, frame2))

        # 화면에 표시
        cv2.imshow('Combined Video', combined_frame)

        if cv2.waitKey(int(1000 / 25.40)) != -1:  # 25.40 FPS 가정
            break

    cv2.destroyAllWindows()
else:
    print("Failed to convert videos to numpy arrays.")
