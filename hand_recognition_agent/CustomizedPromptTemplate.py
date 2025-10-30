import cv2  # used for video processing
# import moviepy.editor as mpe  # used for composing video and audio
import base64  # used for image encoding
import os
import json
from openai import OpenAI
from tqdm import tqdm
from typing import List
from pydantic import BaseModel

from supportprompt_shape import build_shape_support_set
from supportprompt_position import build_position_support_set
from util import get_keyframes

# Setting for OpenAI API
OPENAI_API_KEY = ''
client = OpenAI(api_key=OPENAI_API_KEY)


# Define the data structure for the recognition results
class Frame(BaseModel):
    frame_id: int
    hand_position: int
    hand_shape: int
    reasoning_process: str


class HandRecognition(BaseModel):
    recog_results: List[Frame]


def generate_recognition_single(hand_frames, seed=3702, support_set_path=""):
    Background_prompt = f"""
    You will be asked to recognize the hand positions and shapes for keyframes in video of mandarin cued speech. 
    the speaker is saying a sentence with hand prompt using cued speech in mandarin in the video.
    """

    # In addition to being able to easily determine whether the thumb is straight (similar to the vertical position of other fingers),
    # you can determine whether the other fingers are bent by judging whether their nails are visible. However, please note that in some cases,
    # the nails of the little finger and ring finger may not be clear enough due to the angle problem.
    # It is recommended to only use this method to determine whether the index finger and middle finger are bent or not.

    position_text_incontext_prompt = f"""
    To be more specific, the first task is to select one of the hand position label according the speaker right hand position within hand zoom frames with the reference of support set. 
    Different hand positions are distinguished by fingers pointing location. 
    The explanation corresponding to each position label shows the finger pointing area corresponding to the hand position
    There are 5 hand positions in mandarin cued speech, They are
    0: 'Eye', 
    1: 'Right side of the head',
    2: 'Cheek,
    3: 'Chin',   
    4: 'Below the head'.
"""

    shape_text_incontext_prompt = f"""
    The second task is to select one of the hand shape label according the speaker right hand shape within hand zoom frames with the reference of support set. 
    Different shapes are distinguished by different finger states, i.e. each of five fingers can be straight or bent. 
    The explanation corresponding to each shape label shows the finger status corresponding to the hand shape
    There are 8 hand shapes in mandarin cued speech, They are
    0: 'Index finger straight only. Thumb, middle, ring and pinky fingers are bent',
    1: 'Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent',
    2: 'Middle, ring and pinky fingers are straight. Thumb and index fingers are bent', 
    3: 'Index, middle, ring and pinky fingers are straight. Thumb is bent', 
    4: 'All fingers are straight', 
    5: 'Thumb and index fingers are straight. Middle, ring and pinky fingers are bent ', 
    6: 'Thumb, index and middle fingers are straight. Ring and pinky fingers are bent ',
    7: 'Index and middle fingers are straight and apart. Thumb, ring and pinky fingers are bent',
    """

    COT_prompt = f"""
    Now, you will be asked to first give the hand_position label based on test hand zoom frames. 
    For each test input zoom frame, please use a reasonable reasoning process to complete the prediction
    in the test sample and give its correct position label and reasoning process. The reference reasoning process should 
    be focused on the hand within the test image and find the most similar position category in the support set. 
    Then check again whether the hand position in the test image is consistent with the predicted label based on the label and text description corresponding to the category.

    Please note for the hand_position (refers to straight fingers' pointing location), you will choose from the following five options:
    0: 'Eye', 
    1: 'Right side of the head',
    2: 'Cheek,
    3: 'Chin',   
    4: 'Below the head'.
    
    Then, you will be asked to give the hand_shape label based on test hand zoom frames. 
    For each test input zoom frame, please use a reasonable reasoning process to complete the prediction
    in the test sample and give its correct shape label and reasoning process. The reference reasoning process should 
    be focused on the hand within the test image and find the most similar shape category in the support set. 
    Then check again whether the hand shape in the test image is consistent with the predicted label based on the label and text description corresponding to the category.
     
    Please note for the hand_shape (refers to five fingers' status), you will choose from the following eight options:
    0: 'Index finger straight only. Thumb, middle, ring and pinky fingers are bent',
    1: 'Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent',
    2: 'Middle, ring and pinky fingers are straight. Thumb and index fingers are bent', 
    3: 'Index, middle, ring and pinky fingers are straight. Thumb is bent', 
    4: 'All fingers are straight', 
    5: 'Thumb and index fingers are straight. Middle, ring and pinky fingers are bent', 
    6: 'Thumb, index and middle fingers are straight. Ring and pinky fingers are bent',
    7: 'Index and middle fingers are straight and Apart. Thumb, ring and pinky fingers are bent',
    """

    position_visual_incontext_prompt = build_position_support_set(support_set_path)
    shape_visual_incontext_prompt = build_shape_support_set(support_set_path)

    messages = [
        {"role": "system",
         "content": "You are a helpful assistant designed to recognise the hand position in mandarin cued speech, then output JSON."},
        {"role": "user", "content": Background_prompt},
        {"role": "user", "content": position_text_incontext_prompt},
        {"role": "user", "content": position_visual_incontext_prompt},
        {"role": "user", "content": shape_text_incontext_prompt},
        {"role": "user", "content": shape_visual_incontext_prompt},
        {"role": "user", "content": COT_prompt}
    ]
    token_list = []

    for i in range(len(hand_frames)):
        messages.append({"role": "user",
                         "content": "Here is the {} test sample of the hand zoom image".format(
                             i + 1)})
        messages.append({"role": "user", "content": [{"image": hand_frames[i], "resize": 512}]})
    messages.append({"role": "user",
                     "content": "Please provide the predictions of hand position and shape for each test sample with detailed reasoning process about how you get these labels. Please make good use of support sets and make sure to provide the correct label of position and shape based on each frame."})

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        seed=seed,
        messages=messages,
        response_format=HandRecognition,
    )
    response_content = response.choices[0].message.parsed

    recog_list = response_content.dict()['recog_results']
    token_list.append(response.usage.total_tokens)
    return recog_list, token_list


def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        # convert numpy buffer to bytes before base64 encoding
        base64Frames.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


if __name__ == '__main__':
    hand_frames = process_video(r".\HS-0001.mp4")
    key_frame_index = get_keyframes(r".\HS-0001.npy")
    support_set_path = r".\support_set"

    print('Key frame index: ', key_frame_index)
    hand_frames = [hand_frames[i] for i in key_frame_index]
    print(len(hand_frames), "key frames read.")
    phoneme_label = []
    phoneme_frame_range_list = []
    phoneme_time_range_list = []

    recogn_dict = {}
    recogn_dict['total_frames'] = len(hand_frames)
    recogn_dict['frame_index'] = key_frame_index
    recogn_dict['total_tokens'] = 0
    recognition_list, token_use = generate_recognition_single(hand_frames, seed=3702, support_set_path=support_set_path)
    recogn_dict['recog_results'] = recognition_list
    recogn_dict['total_tokens'] = sum(token_use)

    with open('./output/cued_caption_support.json', 'w') as f:
        json.dump(recogn_dict, f, indent=4)
