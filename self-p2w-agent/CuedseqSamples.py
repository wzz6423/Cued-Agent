import os


def build_cuedseq_list(train_label_path,mandarin_path,sentence_num):
    #read the train_label file
    with open(train_label_path, 'r') as f:
        label_lines = f.readlines()
        f.close()
    #read the mandarin file
    with open(mandarin_path, 'r',encoding='utf-8') as f:
        mandarin_lines = f.readlines()
        f.close()




    content_list = ["Here are some example of mandarin cued speech sequences and the corresponding mandarin sentences."]

    for i in range(sentence_num):
        label_temp_list = label_lines[i].split(',')
        cuedseq=label_temp_list[-1].strip()
        video_num = int(label_temp_list[1].split('-')[1].replace('.mp4',''))
        mandarin_line = mandarin_lines[video_num-1].split("）")[1].strip()
        content_list.append("Cued speech Sequence: {}, the corresponding mandarin sentence: {}".format(cuedseq,mandarin_line))



    return content_list

if __name__ == '__main__':
    train_label_path = r"F:\dataset\chinese_cued_speech_data\HS_train_labels.txt"
    mandarin_path = r"F:\dataset\chinese_cued_speech_data\normal-1000.txt"
    sentence_num = 10
    content_list = build_cuedseq_list(train_label_path,mandarin_path,sentence_num)
    print(content_list)