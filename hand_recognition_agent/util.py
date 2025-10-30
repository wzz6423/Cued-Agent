import numpy as np


def load_npy(npy_path):
    npy_file = np.load(npy_path)
    # print(hand_position.shape)
    return npy_file


def group_elements(lst):
    if not lst:
        return []

    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] <= 2:
            current_group.append(lst[i])
        else:
            result.append(int(np.mean(current_group)))
            # result.append(current_group)
            current_group = [lst[i]]

    if current_group not in result:
        result.append(int(np.mean(current_group)))
    return result

def Keyframe_filter(hand_position):
    slow_index = []
    for i in range(1, len(hand_position)):
        # print(type(hand_position[i]))
        if np.linalg.norm(hand_position[i] - hand_position[i - 1]) <= 6:
            slow_index.append(i)
    final_slow_index = group_elements(slow_index)

    return final_slow_index

def get_keyframes(position_path):
    hand_position = load_npy(position_path)
    # print(len(hand_position))
    slow_frame_index = Keyframe_filter(hand_position)
    return slow_frame_index


def get_keyframe_groups(hand_position):
    # print(len(hand_position))
    slow_frame_groups = screen_slow_motion_group(hand_position)
    return slow_frame_groups

def screen_slow_motion_group(hand_position):
    slow_index = []
    for i in range(1, len(hand_position)):
        # print(type(hand_position[i]))
        if np.linalg.norm(hand_position[i] - hand_position[i - 1]) <= 6:
            slow_index.append(i)
    lip_keyframes_list = group_elements2(slow_index)
    return lip_keyframes_list


def group_elements2(lst):
    if not lst:
        return []

    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] <= 2:
            current_group.append(lst[i])
        else:
            result.append(current_group[:])
            # result.append(current_group)
            current_group = [lst[i]]

    if current_group not in result:
        result.append(current_group[:])
    return result