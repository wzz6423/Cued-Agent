import os

import requests
import json

from tqdm import tqdm

from CuedseqSamples import build_cuedseq_list
from openai import OpenAI

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

output_prompt_2 = """Please output your final result"""

system_prompt = """
The user will provide instructions of how to post-process a cued speech sequence. After post-process please parse the "Processed_Cued_Speech_Sequence" and corresponding 
"Pinyin_Sequence" and "Mandarin_Sequence" output them in JSON format. 

EXAMPLE JSON OUTPUT:
{   
    "Processed_Cued_Speech_Sequence": "n i - j y ao - sh en - m e - m y eng - z i",
    "Pinyin_Sequence": "ni jiao shen me ming zi",
    "Mandarin_Sequence": "你叫什么名字"
    "Reasoning_Process":
}
"""

Background = """
    You will be asked to postprocess a sequence of mandarin cued speech. Cued speech is a system that use hand movements to assistant lip reading for deaf people communication.
    To be more specific, for each cued speech video, speakers are using cued speech talking a whole mandarin sentence. The sentences you will deal with include conversations that may appear in daily scenarios, news content, idioms, and poems.
    We have previously used other models to recognize cued speech videos as cued speech sequences. Due to the limitation of recognition accuracy, there may be some errors in the existing sequences. What you need to do now is to use your Chinese language ability 
    and the grammar of cued speech to complete the post-processing of the cued speech sequence and output the final cued speech sequence, the corresponding pinyin, and the corresponding Chinese sentence.
"""

cued_rules = """
In order to help you better understand the grammar of mandarin cued speech, I will introduce its rules to interchange with pinyin in detail. 
First, a Chinese word usually has its corresponding pinyin, and its pinyin can be converted into a combination of cued speech phonemes according to its pronunciation rules.
Second, the cued speech combination corresponding to a Chinese word consists of 0 to 2 consonant phonemes and only one vowel phoneme. In the mandarin cued speech rules, the consonant phonemes are:
['p','d','zh','k','q','z','s','r','h','b','n','yu','m','t','f','l','x','w','g','j','ch','y','c','sh'].
The vowels are:
['an', 'e', 'o', 'a', 'ou', 'er', 'en', 'i', 'v', 'ang', 'ai', 'u', 'ao', 'eng', 'ong', 'ei']

note the 'v' in the vowels is a special case, it corresponds to the 'ü' in pinyin.

Third, most of the time, the pinyin and cued speech combination are the same. But there are some special cases that need to be noted:

In a word combination without consonant phonemes, the corresponding combination of the word is composed by one vowel phonemes, examples include (Mandarin -> Cued speech):
'安' -> 'an', '饿' -> 'e', '哦' -> 'o', '啊' -> 'a', '欧' -> 'ou', '而' -> 'er', '恩' -> 'en', '昂' -> 'ang', '爱' -> 'ai', '奥' -> 'ao'.

Please note there are also certain special circumstances. 
if a word's pinyin is 'yi' like '衣' and '一', the corresponding cued speech combination is 'i'.
Also if a a word's pinyin is 'yu', like '雨' the corresponding cued speech combination is 'v'. 
If a word's pinyin is 'wu', like '午', the corresponding cued speech combination is 'u'.

Fourth, in a combination with consonant phonemes, the corresponding combination of the word is composed of one or two consonant phonemes and one vowel phoneme.
But there are also some special cases: If the pinyin of a word contains the following compound vowels, 
the compound vowels will be converted into a double syllable of the medial consonant and the simple vowel (Pinyin compound vowels)->(medial consonant simple vowel): 
'ia' -> 'y a', 'iao' -> 'y ao', 'ian' -> 'y an', 'iang' -> 'y ang', 'ie' -> 'y e', 'in' -> 'y en', 'ing' -> 'y eng', 'iu' -> 'y ou', 'iong' -> 'y ong', 
'ua' -> 'w a', 'uai' -> 'w ai', 'uan' -> 'w an', 'uang' -> 'w ang', 'ui' -> 'w ei', 'un' -> 'w en','uo' -> 'w o'.
some examples include (Mandarin -> Pinyin -> Cued speech):
'下' -> 'xia' -> 'x y a', '跳' -> 'tiao'-> t y ao', '片' -> 'pian' -> 'p y an', '想' -> 'xiang' -> 'x y ang', 
'谢' -> 'xie' -> 'x y e', '林' -> 'lin' -> 'l y en', '经' -> 'jing' -> 'j y eng', '秋' -> 'qiu' -> 'q y ou', '熊' -> 'xiong' -> 'x y ong', 
'瓜' -> 'gua' -> 'g w a', '怀' -> 'huai' -> 'h w ai', '宽' -> 'kuan' -> 'k w an', '光' -> 'guang' -> 'g w ang', 
'会' -> 'hui' -> 'h w ei', '准' -> 'zhun' -> 'zh w en', '活' -> 'huo' -> 'h w o'. '情' -> 'qing' -> 'q y eng'.

Because y is occupied, there are also two special cases (Mandarin -> Pinyin -> Cued speech):
'音' -> 'yin' -> 'y en', '英' -> 'ying' -> 'y eng'


Also, if a word's first consonant is one of 'j', 'q', 'x', 'y' and the following compound vowels is one of 'uan', 'ue', 'un',  which like 'xuan','que', 'xun', 'yun', 'yuan'
the following compound vowels is actually 'van(üan)', 've(üe)', 'vn(ün)', and the corresponding cued speech is
'van' -> 'yu an', 've' -> 'yu e', 'vn' -> 'yu en'.
some examples include (Mandarin -> Pinyin -> Cued speech):
'捐' -> 'juan' -> 'j yu an', '宣' -> 'xuan' -> 'x yu an', '元' -> 'yuan' -> 'yu an' '却' -> 'que' -> 'q yu e', '云' -> 'yun' ->'yu en'.

After understanding the interchange rules of cued speech sequence, pinyin and mandarin, we provide some examples of cued speech sequence and its corresponding Mandarin sentence.
Please note that in general, in a cued speech sequence, phonemes belonging to the same Chinese word are separated by ' ', and words are separated by '-'.
"""

question_prompt = """Now, you will post-process the following cued speech sequence. 
It is worth noting that for the current cued speech sequence you are going to post-process, 
more than 90% on average of the phonemes are correct, with only a few errors. 

So your first task is to check whether there are obvious grammatical errors in the cued speech sequence, mainly whether the phoneme combination of a word is grammatical.

The second task is to use your Chinese ability and interchange rules try to interchange the cued speech sequence to pinyin and convert it into a corresponding fluent Chinese sentence as much as possible. You may need to make multiple attempts in this process to ensure that the corresponding Chinese sentence is semantically fluent.

The third task, combined with the difficulties you encountered in converting to a fluent Chinese sentence, try to modify as few phonemes as possible and get the final post-processing result. Once again, please be cautious in modifying the phonemes in the cued sequence, and be more cautious in deleting or adding phonemes.

Here are some tips when you try to modify the phonemes in the cued speech sequence:

the phoneme 'a' is may misidentified as 'er', the phonemes 'b', 'p', and 'm' are easily confused, the phonemes 't' and 'n','r' are easily confused, 
the phonemes 'j', 'q', and 'x' are easily confused, the phonemes 'g', 'k', and 'h' are easily confused
the phonemes 'z', 'c', 's', are easily confused, 

Also, the phonemes 's', 'sh', are easily confused, the phonemes 'c' and 'ch' are easily confused, and the phonemes 'z' and 'zh' are easily confused. 
If the phoneme 'y' is the first in a word combination, consider whether there is another consonant phoneme missing before it. 
Vowel phonemes are less likely to be wrong.

Make as few changes as possible to ensure that the sentence semantic smoothly, replace a whole word combination or add or delete phonemes with caution!!!!
Do not try to swap the order or split of words!!! 

Also, please note the complete sentence may also contain some numbers, dates, medical dialogue, news content, Chinese idioms, two-part allegorical sayings, and poems, etc. The sentence does not contain words combination with unclear meanings. The target sentence must be one that appears in daily communication.

Here is the predicted cued speech sequence: 
"""


"""please to try to convert it into pinyin and translate it into Chinese sentences. 
Use your Chinese ability to check whether the Chinese sentence is smooth, has grammatical errors, and is semantically fluent.
If there are any of the above problems, please first only check the following tips for modification"""

output_prompt = """Please output your final result with format in following sample:
                    "Processed_Cued_Speech_Sequence": "n i - j y ao - sh en - m e - m y eng - z i",
                    "Pinyin_Sequence": "ni jiao shen me ming zi",
                    "Mandarin_Sequence": "你叫什么名字"
                    "Reasoning_Process":
                    """


def single_process(predicted_cued_seq):
    train_label_path = r"F:\dataset\chinese_cued_speech_data\HS_train_labels.txt"
    mandarin_path = r"F:\dataset\chinese_cued_speech_data\normal-1000.txt"
    sentence_num = 20
    content_list = build_cuedseq_list(train_label_path, mandarin_path, sentence_num)

    messages = Background + cued_rules
    for content in content_list:
        messages += content

    messages += question_prompt
    messages += predicted_cued_seq
    messages += output_prompt

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user", "content": messages}],
        temperature=0.6

    )
    # print(response.choices[0].message.content)
    return response


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    return lines


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    return lines


def file_loop(predicted_txt, label_csv, output_folder):
    text_lines = read_txt(predicted_txt)
    csv_lines = read_csv(label_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if len(text_lines) != len(csv_lines):
        print("The number of lines in the two files is not equal.")
        quit()

    for i in tqdm(range(len(text_lines))):
        try:
            predicted_cued_seq = text_lines[i].strip()
            filename = csv_lines[i].split(',')[1].split('/')[-1].replace('.mp4', '')
            output_file = os.path.join(output_folder, f"{filename}.txt")

            if os.path.exists(output_file):
                print(f"{output_file} already exists.")
                continue
            else:
                response = single_process(predicted_cued_seq)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(response.choices[0].message.content)
                f.close()
                print(f"Write to {output_file} successfully.")
        except:
            print(f"Error in processing {filename}.")
            continue



def file_loop_reverse(predicted_txt, label_csv, output_folder):
    text_lines = read_txt(predicted_txt)
    csv_lines = read_csv(label_csv)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if len(text_lines) != len(csv_lines):
        print("The number of lines in the two files is not equal.")
        quit()

    for i in tqdm(range(len(text_lines)-1,0,-1)):
        try:
            predicted_cued_seq = text_lines[i].strip()
            filename = csv_lines[i].split(',')[1].split('/')[-1].replace('.mp4', '')
            output_file = os.path.join(output_folder, f"{filename}.txt")

            if os.path.exists(output_file):
                print(f"{output_file} already exists.")
                continue
            else:
                response = single_process(predicted_cued_seq)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(response.choices[0].message.content)
                f.close()
                print(f"Write to {output_file} successfully.")
        except:
            print(f"Error in processing {filename}.")
            continue


if __name__ == '__main__':
    predicted_cued_seq = 'n i - d ei - ch v - k an - f u - g e - y an - ch a - r w an - ch ao - g ong - n eng '
    'n i - d ei - ch v - k an - f u - g e - y an - ch a - r w an - ch ao - g ong - n eng '

    response = single_process(predicted_cued_seq)
    print(response.choices[0].message.content)
    print(response.choices[0].message.reasoning_content)
