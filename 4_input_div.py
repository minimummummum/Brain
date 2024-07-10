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
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/name/name1": {
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/name/name2": {
            "descriptor": [],
            "actions": [],
            "reward": 0
        }
    },
    "soccer": {
        "memory/soccer": {
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/soccer/pass": {
            "descriptor": [],
            "actions": [],
            "reward": 0
        },
        "memory/soccer/shoot": {
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
import os
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
    if img is not None:
        blurred_frame = cv2.GaussianBlur(img, (5, 5), 0)
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
        return descriptors
    else:
        print(f"{img} file open error")
        return None
# 특징점 매칭 비교 메인 키만 검색 ex) 축구, 농구
def match_ratio_main(descriptors):
    match_list_main = []
    for main_key in name_info:
        descriptor_memroy = name_info[main_key]["memory/"+main_key]["descriptor"]
        if descriptors is None or descriptor_memroy is None:
            continue
        matches = bf.knnMatch(descriptors, descriptor_memroy, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(matches) > 0:
            match_ratio = len(good_matches) / len(matches)
        else:
            match_ratio = 0.0
        match_list_main.append([main_key, match_ratio])
    return sorted(match_list_main, key=lambda x: x[1], reverse=True)
# 특징점 매칭 비교 세부 키 검색 ex) 축구-> 패스, 슛, 드리블
def match_ratio_details(descriptors, match_list_main_select):
    match_list_details = []
    main_key = match_list_main_select[0]
    for detail_key in name_info[main_key]:
        descriptor_memroy = name_info[main_key][detail_key]["descriptor"]
        if descriptors is None or descriptor_memroy is None:
            continue
        matches = bf.knnMatch(descriptors, descriptor_memroy, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(matches) > 0:
            match_ratio = len(good_matches) / len(matches)
        else:
            match_ratio = 0.0
        match_list_details.append([detail_key, match_ratio])
    return sorted(match_list_details, key=lambda x: x[1], reverse=True)
def best_match_ratio(descriptors):
    best_match_list = []
    # 메인 키 중 매치율 제일 높은 이름(나중에 E-greedy로 2위, 3위 등 쓰기)
    match_list_main = match_ratio_main(descriptors)
    for best_match_list_main in match_list_main:
        if best_match_list_main is not None:  # None 체크 추가
            match_list_detail = match_ratio_details(descriptors, best_match_list_main)
            for j in range(min(len(match_list_detail), 10)):
                best_match_list_detail = match_list_detail[j]
                if best_match_list_detail is not None:  # None 체크 추가
                    best_match_list.append(best_match_list_detail)
    return sorted(best_match_list, key=lambda x: x[1], reverse=True)

# 해당 데이터 이름 지정
def naming(best_match_list):
    if auto_mode:
        pass # name_info 돌려서 비슷한 거 일정 퍼센트 넘으면 그 이름으로
            # 메인 키 검색해서 특정 일치율 넘는 게 없을 경우 & 세부 키에서도 특정 일치율 넘는 게 없을 경우
            # 새로운 랜덤 메인 키 생성 -> 랜덤 했는데 만약 name_info에 있을 경우 다시 랜덤 생성
            # 랜덤 모드로 지어진 이름을 나중에 교육으로 수정 -> 유사율이 특정 숫자보다 높을 경우 이름 수정 모드 여부 확인
    else:
        print(name_info.keys())
        print(f"현재 가장 유사한 카테고리: {best_match_list}") 
        main_key = input("어떤 이름으로 지정하겠습니까?") # 재정의, 새정의?
        if main_key in name_info:
            print("지정하신 이름은 현재 존재하는 이름입니다.")
            print(name_info[main_key].keys())
            detail_key = input(f"세부 이름을 지정해주세요.(메인 이름과 동일하게 가능) {main_key}/")
        else:
            print("미존재")
            detail_key = None
    return main_key, detail_key

# 이름 정보를 파일에 저장하는 함수
def save_data_to_file(dir_path, filename, data, img):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    cv2.imwrite(os.path.join(dir_path, f"{filename}.jpg"), img)
    with open(os.path.join(dir_path, f"{filename}.pkl"), 'wb') as f:
        pickle.dump(data, f)
    print(f"경로: {dir_path}, 이름: {filename}으로 저장되었습니다.")

# name_info에 name 저장
def name_info_save(img, descriptors, main_key, detail_key, actions=[], reward=0):
    global name_info
    # main_key가 name_info에 없을 경우
    if main_key not in name_info:
        name_info[main_key] = {}
    if main_key == detail_key or detail_key is "":
        if main_key in name_info and name_info[main_key]:
            index = 1
            while f"memory/{main_key}/{main_key}{index}" in name_info[main_key]:
                index += 1
            memory_key = f"memory/{main_key}/{main_key}{index}"
        else:
            memory_key = f"memory/{main_key}"
    else:
        if f"memory/{main_key}/{detail_key}" in name_info[main_key]:
            index = 1
            while f"memory/{main_key}/{detail_key}{index}" in name_info[main_key]:
                index += 1
            memory_key = f"memory/{main_key}/{detail_key}{index}"
        else:
            memory_key = f"memory/{main_key}/{detail_key}"
    name_info[main_key][memory_key] = {
        "descriptor": descriptors,
        "actions": actions,
        "reward": reward
    }
    # 디렉토리 생성 및 이미지와 데이터 저장
    save_data_to_file(memory_key, detail_key if detail_key else main_key, name_info[main_key][memory_key], img)
    # name_info 업데이트
    with open('configuration/name_info.pkl', 'wb') as f:
        pickle.dump(name_info, f)

#main_test, 나중에 함수에 넣기
reboot_set()
print("현재 name_info")
print(name_info)
img = cv2.imread('input.jpg')
descriptors = detect_sift(img)
best_match_list = best_match_ratio(descriptors)
main_key, detail_key = naming(best_match_list)
name_info_save(img, descriptors, main_key, detail_key, actions=[], reward=0)
print("수정 name_info")
print(name_info)
# 지금 main match -> detail match인데 수정할 필요가 있음
# 조금 손 봐서 main match의 detail match를 전부 도는 걸로 되어 있음
# 1. main match 함수를 main 돌면서 detail의 min(길이, 10)으로 개수 뽑고, 랜덤으로 개수만큼 뽑기
# 2. 랜덤으로 뽑은 것 중 가장 높은 detail의 main을 best_match_main 리스트에 저장
# 3. main 다 돌았으면 best_match_main 높은 순으로 정렬 후 리턴
# 4. best_match_main 중 min(len, 5)개 정도만 높은 순으로 컷하여서 main의 detail 전부 돌면서 가장 높은 거 하나 뽑기
# 5. 뽑은 걸 best_match_detail 리스트에 저장, 높은 순으로 정렬 
# 6. 그걸로 naming 함수 실행

# 그리고 추가해야 할 게 액션을 했을 경우 그 img와 데이터, 어떤 액션 했는지까지 저장
# 즉, name_info_save는 액션했을 때는 무조건 실행 후 저장 but 액션 안 했을 때 실행 안 되는 건 아님
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