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
import random
import string
import re
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
        keypoints, descriptor = sift.detectAndCompute(gray_frame, None)
        return descriptor
    else:
        print(f"{img} file open error")
        return None
    
def match_ratio_main(descriptor):
    match_list_main = []
    for main_key in name_info:
        random_detail_keys = random.sample(list(name_info[main_key].keys()), min(len(name_info[main_key]), 10))
        max_ratio = 0
        for random_detail_key in random_detail_keys:
            descriptor_memory = name_info[main_key][random_detail_key]["descriptor"]
            if descriptor is None or descriptor_memory is None:
                continue
            match_ratio = compute_match_ratio(descriptor, descriptor_memory)
            max_ratio = max(max_ratio, match_ratio)
        match_list_main.append([main_key, max_ratio])
    return sorted(match_list_main, key=lambda x: x[1], reverse=True)

def match_ratio_details(descriptor, match_list_main):
    match_list_best = []
    for match_main in range(min(len(match_list_main), 5)):
        main_key = match_list_main[match_main][0]
        match_list_details = []
        for detail_key in name_info[main_key]:
            descriptor_memory = name_info[main_key][detail_key]["descriptor"]
            if descriptor is None or descriptor_memory is None:
                continue
            match_ratio = compute_match_ratio(descriptor, descriptor_memory)
            match_list_details.append([main_key, detail_key, match_ratio])
        if match_list_details:
            best_detail = max(match_list_details, key=lambda x: x[2])
            match_list_best.append(best_detail)
    return sorted(match_list_best, key=lambda x: x[2], reverse=True)

def match_ratios(descriptor):
    match_list_main = match_ratio_main(descriptor)
    match_list_details = match_ratio_details(descriptor, match_list_main)
    return match_list_details

def compute_match_ratio(descriptor1, descriptor2):
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_ratio = len(good_matches) / len(matches) if matches else 0.0
    return match_ratio

def random_string(len):
    characters = string.ascii_lowercase
    random_string = ''.join(random.choice(characters) for _ in range(len))
    return random_string

def remove_trailing_numbers(s):
    return re.sub(r'\d+$', '', s)

def remove_first_two_parts(path):
    parts = path.split('/')
    remaining_parts = parts[2:]
    return '/'.join(remaining_parts)

# 해당 데이터 이름 지정
def naming(best_match_list):
    auto_mode = False
    if auto_mode:
        if best_match_list[0][2] > 0.4:
            main_key = best_match_list[0][0]
            detail_key = best_match_list[0][1]
            detail_key = remove_trailing_numbers(detail_key)
            detail_key = remove_first_two_parts(detail_key)
        else:
            random_name_flag = True
            while random_name_flag:
                random_name = random_string(15)
                random_name = remove_trailing_numbers(random_name)
                if random_name not in name_info:
                    random_name_flag = False
                    main_key = random_name
                    detail_key = None
        # name_info 돌려서 비슷한 거 일정 퍼센트 넘으면 그 이름으로
        # 메인 키 검색해서 특정 일치율 넘는 게 없을 경우 & 세부 키에서도 특정 일치율 넘는 게 없을 경우
        # 새로운 랜덤 메인 키 생성 -> 랜덤 했는데 만약 name_info에 있을 경우 다시 랜덤 생성 # 2024-07-17 여기까지 완
        # 랜덤 모드로 지어진 이름을 나중에 교육으로 수정 -> 유사율이 특정 숫자보다 높을 경우 이름 수정 모드 여부 확인
        # 새로운 키 생성 후 지울 키 pop
    else:
        print(f"현재 가장 유사한 카테고리: {best_match_list}") 
        main_key = input("어떤 이름으로 지정하겠습니까?, 맨 끝 숫자 불가능") # 재정의, 새정의?
        if main_key in name_info:
            print("세부 이름을 지정해주세요.")
            print(name_info[main_key].keys())
            detail_key = input(f"메인 이름과 동일하게 가능, 맨 끝 숫자 불가능 {main_key}/")
        else:
            print("생성")
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
def name_info_save(img, descriptor, main_key, detail_key, actions=[], reward=0):
    global name_info
    # main_key가 name_info에 없을 경우
    if main_key not in name_info:
        name_info[main_key] = {}
    if main_key == detail_key or detail_key == None or detail_key == "":
        if main_key in name_info and name_info[main_key]:
            index = 1
            while f"memory/{main_key}/{main_key}{index}" in name_info[main_key]:
                index += 1
            memory_key = f"memory/{main_key}/{main_key}{index}"
            filename = f"{main_key}{index}"
        else:
            memory_key = f"memory/{main_key}"
            filename = f"{main_key}"
    else:
        if f"memory/{main_key}/{detail_key}" in name_info[main_key]:
            index = 1
            while f"memory/{main_key}/{detail_key}{index}" in name_info[main_key]:
                index += 1
            memory_key = f"memory/{main_key}/{detail_key}{index}"
            filename = f"{detail_key}{index}"
        else:
            memory_key = f"memory/{main_key}/{detail_key}"
            filename = f"{detail_key}"
    name_info[main_key][memory_key] = {
        "descriptor": descriptor,
        "actions": actions,
        "reward": reward
    }
    # 디렉토리 생성 및 이미지와 데이터 저장
    save_data_to_file(memory_key, filename, name_info[main_key][memory_key], img)
    # name_info 업데이트
    with open('configuration/name_info.pkl', 'wb') as f:
        pickle.dump(name_info, f)
def main_key_edu():#07-18 여기 개발중
    name_flag = True
    img = cv2.imread('input.jpg')
    descriptor = detect_sift(img)
    best_match_list = match_ratios(descriptor)
    print("이 이미지와 유사한 이름 리스트")
    print(best_match_list)
    while name_flag:
        main_key = input("수정할 이름 선택(main_key)")
        if main_key not in name_info:
            print("해당 이름은 존재하지 않는 이름입니다.")
        else:
            while True:
                rename_main_key = input("어떤 이름으로 수정하실 건가요? 맨 뒤 숫자 불가능")
                if rename_main_key not in name_info:
                    name_flag = False
                    memory_name_change(main_key, rename_main_key)
                    name_info_change(main_key, rename_main_key)
                    print("이름 수정 완료.")
                    break
                else:
                    print("해당 이름은 이미 존재합니다.")

def memory_name_change(main_key, rename_main_key):
    parent_dir = "memory"  # 부모 디렉토리 이름
    old_dir_name = main_key  # 기존 디렉토리 이름
    new_dir_name = rename_main_key  # 새 디렉토리 이름
    old_dir = os.path.join(parent_dir, old_dir_name)  # 기존 디렉토리의 전체 경로
    new_dir = os.path.join(parent_dir, new_dir_name)  # 새 디렉토리의 전체 경로
    if os.path.exists(old_dir) and os.path.isdir(old_dir):
        # 새 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        # 기존 디렉토리의 모든 파일과 폴더를 탐색
        for root, dirs, files in os.walk(old_dir):
            for name in dirs + files:
                old_item_path = os.path.join(root, name)  # 기존 파일 또는 폴더의 전체 경로
                # 새 경로를 계산
                relative_path = os.path.relpath(old_item_path, old_dir)  # 기존 디렉토리를 기준으로 상대 경로
                new_item_name = relative_path.replace(old_dir_name, new_dir_name)  # 새 디렉토리 이름으로 변환된 경로
                new_item_path = os.path.join(new_dir, new_item_name)  # 새 디렉토리 내의 새 전체 경로
                # 필요한 경우 새 디렉토리 구조를 생성
                if not os.path.exists(os.path.dirname(new_item_path)):
                    os.makedirs(os.path.dirname(new_item_path))
                # 파일 또는 폴더를 새 경로로 이동
                os.rename(old_item_path, new_item_path)
                print(f"Renamed {old_item_path} -> {new_item_path}")  # 변경된 경로 출력
        # 기존 디렉토리가 비어 있으면 삭제
        if not os.listdir(old_dir):
            os.rmdir(old_dir)
            print(f"Removed empty directory {old_dir}")  # 삭제된 디렉토리 출력
    else:
        print(f"{old_dir} 존재하지 않습니다.")  # 기존 디렉토리가 존재하지 않을 때의 메시지

def name_info_change(main_key, rename_main_key): # 07-19 이름 수정 기능 완
    global name_info
    if main_key in name_info:
        name_info[rename_main_key] = name_info.pop(main_key)
        new_details = {}
        for detail_key in list(name_info[rename_main_key].keys()):
            new_detail_key = detail_key.replace(main_key, rename_main_key)
            new_details[new_detail_key] = name_info[rename_main_key].pop(detail_key)
        name_info[rename_main_key].update(new_details)
        with open('configuration/name_info.pkl', 'wb') as f:
            pickle.dump(name_info, f)

# test
# reboot_set()
# print("현재 name_info")
# print(name_info)
# img = cv2.imread('input.jpg')
# descriptor = detect_sift(img)
# best_match_list = match_ratios(descriptor)
# main_key, detail_key = naming(best_match_list)
# name_info_save(img, descriptor, main_key, detail_key, actions=[], reward=0)
# print("수정 name_info")
# print(name_info)
# main_key_edu()





# 지금 main match -> detail match인데 수정할 필요가 있음
# 조금 손 봐서 main match의 detail match를 전부 도는 걸로 되어 있음
# 1. main match 함수를 main 돌면서 detail의 min(길이, 10)으로 개수 뽑고, 랜덤으로 개수만큼 뽑기
# 2. 랜덤으로 뽑은 것 중 가장 높은 detail의 main을 best_match_main 리스트에 저장
# 3. main 다 돌았으면 best_match_main 높은 순으로 정렬 후 리턴 2024-07-11 여기까지 완
# 4. best_match_main 중 min(len, 5)개 정도만 높은 순으로 컷하여서 main의 detail 전부 돌면서 가장 높은 거 하나 뽑기
# 5. 뽑은 걸 best_match_detail 리스트에 저장, 높은 순으로 정렬 
# 6. 그걸로 naming 함수 실행 2024-07-16 완
# 오토모드 개발하기 # 2024-07-19 완 -> 액션 개발하면 됨 -> 리워드 추가, 리워드 부터 개발


########################################################################
# 액션
# 그리고 추가해야 할 게 액션을 했을 경우 그 img와 데이터, 어떤 액션 했는지까지 저장
# 즉, name_info_save는 액션했을 때는 무조건 실행 후 저장 but 액션 안 했을 때 실행 안 되는 건 아님
# 에피소드? 리플레이? 폴더 생성
# start -> action -> end(reward) 과정 img, descriptor, action, reward 주기적으로 저장
# 그 폴더 참고하여 액션 선택
"""
Actions
# 로봇 관절 인덱스
shoulder_right = 4 # -10~190
shoulder_left = 49 # -190~10
hand_right = 7 # -60~50
hand_left = 52 # -50~60
leg_right = 15 # -20~20
leg_left = 31 # -20~20
foot_right = 20 # -90~90
foot_left = 36 # -90~90
max_velocity = 8 # 모터 속도
actions_info = {
    "motor_control":{
        "shoulder_right": [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "shoulder_left": [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        "hand_right": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "hand_left": [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50],
        "leg_right": [-20, -10, 0, 10, 20],
        "leg_left": [-20, -10, 0, 10, 20],
        "foot_right": [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "foot_left": [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "delay": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    #package
    "work": [[foot_left, 7], [foot_right, 7], [delay, 1] ...],
    "turn_left": [],
    "stop": [[delay, 10]]
}
2024-07-21
1. 어떤 액션을 할 것인가? package에서 액션 선택하는 기능
2. 액션 시작할 때 img, 액션 끝나고 img, 어떤 액션 했는지, reward 저장(episode? replay?), or name_info에 action, reward 추가?
3. 액션 학습 모드. work에서 수정, 추가, 삭제를 통해 보상 확인 후 work2, work3 등 생성, 저장


"""


########################################################################
# 리워드
# 어떤 보상을 주는 기관
# ex) 기울기 센서가 많이 기울었을 경우 마이너스 보상
# mini_game 네모 박스 안에 있을 경우 플러스 보상 -> 검정 공 물체 따라가기 follow ball
# def add(a,b):
    # return a+b
# def multiply(a,b):
    # return a*b
# always_functions = { 여기는 넘어짐, 부딪힘 등
# 'add': add,
# 'multiply': multiply
# }
# with open('functions.pkl', 'wb') as file:
#     pickle.dump(functions, file)

# temp_functions = { 여기는 minigame 같은 거
# }

# 저장된 함수 객체들을 파일에서 불러오기
# with open('functions.pkl', 'rb') as file:
#     loaded_functions = pickle.load(file)

# loaded_functions['add'](3, 5)
# loaded_functions['multiply'](4, 6)

def mini_game_follow_ball():#2024-07-22 여기하는중
    # ball 이미지 등록 후 cam에서 ball이 감지될 경우 게임 시작
    # cam에서 ball이 있으면 +보상, cam에서 ball이 없으면 -보상
    # 처음에는 랜덤으로 액션 선택 stay, work, turn_left. turn_right
    # short_memory 만들어서 단기 기억 생성, -+ 보상 얻을 때마다 img 저장, 액션 저장
    # 현재 이미지를 short_memory 참고해서 비슷한 순 
    # short_memory에서 각 비슷한 상황일 때 보상 높은 걸로 리플레이? 에피소드?에 저장
########################################################################