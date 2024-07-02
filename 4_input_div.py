import numpy as np
import cv2
import pickle

########################################################################
'''
1. 캠으로 이미지를 인풋으로 받고, 이미지에 대한 Keypoints 및 Descriptor를 구한다.
2. 이미지에 대한 name을 교육자에게 받는다. 그렇지 않을 경우 랜덤 값으로 지정한다.
(랜덤일 경우 name_path에서 값이 있으면 다시 랜덤)
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