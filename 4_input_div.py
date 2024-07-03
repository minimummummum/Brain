'''
1. 캠으로 이미지를 인풋으로 받고, 이미지에 대한 Keypoints 및 Descriptor를 구한다.
2. 이미지에 대한 name을 교육자에게 받는다. 그렇지 않을 경우 auto_name1, 2, 3... 으로 지정
(나중에 auto_name에 대한 교육을 할 시 이름 수정)
3. name_info에서 name이 있는지 검색한 후, 없을 경우 memory 폴더에 새로운 name폴더를 만든다.
4. name 폴더에 이미지를 name으로 저장하고, Keypoints, Descriptro 정보 파일도 저장한다.
5. 만약 4에서 name이 있을 경우, name1, name2 등으로 1씩 추가해서 저장한다.
6. name_info 딕셔너리에 정보를 없으면 저장, 있으면 추가한다. 저장일 경우 keypoints, descriptor를 저장한다.
(인풋 값을 매칭 비교할 때 사용)
name_info = {
    "name": {
        "memory/name": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/name/name1": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/name/name2": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        }
    },
    "soccer": {
        "memory/soccer": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/soccer/pass": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/soccer/shoot": {
            "keypoints": [],
            "descriptor": [],
            "actions": [],
            "reward": 0
        }
    }
}
# Pickle 파일에서 객체 읽어오기
with open('name_info.pkl', 'rb') as f:
    name_info = pickle.load(f)

# Pickle 파일로 저장
with open('name_info.pkl', 'wb') as f:
    pickle.dump(name_info, f)

'''
########################################################################
import numpy as np
import cv2
import pickle
auto_mode = False
actions = {}
agent = {}
name_info = {}
bf = cv2.BFMatcher()

# configuration 폴더 내 정보 모두 불러오기
def reboot_set():
    global actions, agent, name_info
    with open('configuration/name_info.pkl', 'rb') as f:
        name_info = pickle.load(f)
    with open('configuration/agent.pkl', 'rb') as f:
        agent = pickle.load(f)
    with open('configuration/actions.pkl', 'rb') as f:
        actions = pickle.load(f)

# SIFT로 특징점 검출, 기술자 계산
def detect_sift(img):
    sift = cv2.SIFT_create()
    if img:
        while True:
            blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
            gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
            keypoint, descriptor = sift.detectAndCompute(gray_frame, None)
    else:
        print(f"{img} file open error")
    return descriptor
# 특징점 매칭 비교
def match_ratio(descriptor):
    match_list = []
    for name in name_info:
        descriptor_memroy = name_info[name]["memory/"+name]["descriptor"]
        if descriptor is None or descriptor_memroy is None:
            continue
        matches = bf.knnMatch(descriptor, descriptor_memroy, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(matches) > 0:
            match_ratio = len(good_matches) / len(matches)
        else:
            match_ratio = 0.0
        match_list.append([name, match_ratio])
    return sorted(match_list, key=lambda x: x[1])
#####2024-07-03 여기까지 함






# 해당 데이터 이름 지정
def naming(img, best_match_ratio): # 유사율? 매개변수 받아야함
    if auto_mode:
        pass # 구현할 때 naming()에 매개변수 받고, 그걸로 name_info 돌려서 비슷한 거 일정 퍼센트 넘으면 그 이름으로
            # 만약, soccer과 유사한데, 그 중에서 pass와 유사할 경우 pass/pass1, 2, 3 ...
    else:
        name = input("어떤 이름으로 지정하겠습니까?") # 재정의, 새정의?
    return name
# name_info에 name 저장
def name_info_save(name, keypoints_list):
    pass









########################################################################
"""
Actions
actions_info = {
    #servo motor
    "왼발목": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    "오른발목": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    ...

    #package
    "work": {
        "configuration/action/work1": [[왼발목, 30], [딜레이, 5], [골반,  30], [딜레이, 5], [왼발목, 0]],
        "configuration/action/work2": [],
        "configuration/action/work3": []
    },
    "turn_left": {
        "configuration/action/turn_left1": [],
        "configuration/action/turn_left2": [],
        "configuration/action/turn_left3": []
    },
}

"""

########################################################################