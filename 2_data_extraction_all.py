import cv2
import numpy as np
import os
####지금 해야 할 거는 이 영상들을 numpy 배열로 변환 후 데이터베이스에 저장하기, 그 데이터베이스 불러오기, 데이터베이스에서 검색하기
#### 영상을 numpy 배열로 변환할지 특징점 같은 거를 numpy 배열로 저장할지??? 뭘로 할까
# 함수 정의: 영상 파일에서 SIFT 특징점 검출 후 반환
def detect_sift_features(video_file):
    cap = cv2.VideoCapture(video_file)
    sift = cv2.SIFT_create()
    keypoints_list = []

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 노이즈 제거 (가우시안 블러)
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # 그레이스케일로 변환
            gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

            # SIFT로 특징점 검출 및 기술자 계산
            keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

            # 검출된 특징점 저장
            keypoints_list.append((keypoints, descriptors))

        cap.release()
    else:
        print(f"Failed to open video file: {video_file}")

    return keypoints_list

# 함수 정의: 두 개의 영상에서 특징점 매칭 후 가장 유사도가 높은 두 개의 영상 반환
def match_and_play_videos(video_files):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # 영상 파일들에서 특징점 검출 및 기술자 계산
    keypoints_descriptors = []
    for video_file in video_files:
        keypoints_list = detect_sift_features(video_file)
        keypoints_descriptors.append(keypoints_list)

    # 매칭할 영상 인덱스 초기화
    best_match_indices = (-1, -1)
    best_match_ratio = 0

    # 모든 영상들 간의 특징점 매칭 비교
    for i in range(len(keypoints_descriptors)):
        for j in range(i + 1, len(keypoints_descriptors)):
            keypoints1, descriptors1 = keypoints_descriptors[i][-1]  # 마지막 프레임의 특징점 사용
            keypoints2, descriptors2 = keypoints_descriptors[j][-1]  # 마지막 프레임의 특징점 사용

            if descriptors1 is None or descriptors2 is None:
                continue

            # 기술자 매칭
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # 좋은 매칭 포인트 선택
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 매칭 비율 계산
            match_ratio = len(good_matches) / len(matches)
            print(f"Match ratio between video {i + 1} and video {j + 1}: {match_ratio}")

            # 가장 높은 매칭 비율인 영상 쌍 선택
            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_match_indices = (i, j)

    if best_match_indices == (-1, -1):
        print("No good matches found.")
        return

    # 가장 높은 매칭 비율을 보인 두 개의 영상 선택
    best_video1_index, best_video2_index = best_match_indices
    best_video1 = video_files[best_video1_index]
    best_video2 = video_files[best_video2_index]
    print(f"Best matched videos: {best_video1}, {best_video2}")

    
    # 선택된 두 개의 영상을 순서대로 재생
    for best_video in [best_video1, best_video2]:
        cap = cv2.VideoCapture(best_video)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Best Matched Videos", frame)
                if cv2.waitKey(int(1000 / fps)) != -1:
                    break
            cap.release()
        else:
            print(f"Failed to open video file: {best_video}")

    cv2.destroyAllWindows()

# memory 폴더 경로 설정
memory_folder = "memory/"

# memory 폴더의 모든 영상 파일에 대해 매칭 수행
video_files = []
for filename in os.listdir(memory_folder):
    if filename.endswith(".avi"):
        video_files.append(os.path.join(memory_folder, filename))

if len(video_files) < 2:
    print("At least two video files are required for matching.")
else:
    match_and_play_videos(video_files)
