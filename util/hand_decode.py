hashmap = { "<blank>" : 0,
            "<unk>":1,"b": 2, "p": 3, "m": 4, "f": 5, "d": 6, "t": 7, "n": 8, "l": 9, "g": 10, "k": 11, "h": 12, "j": 13,
            "q": 14, "x": 15, "zh": 16, "ch": 17, "sh": 18, "r": 19, "z": 20, "c": 21, "s": 22, "y": 23, "w": 24,
            "yu": 25, "a": 26, "o": 27, "e": 28, "i": 29, "u": 30, "v": 31, "ai": 32, "ei": 33, "ao": 34, "ou": 35, "er": 36,
            "an": 37, "en": 38, "ang": 39, "eng": 40, "ong": 41, "-": 42}

def label2phone(hand_position, hand_gesture):
    if hand_position == 0:
        vowel_options = ['an', 'e', 'o']
    elif hand_position == 1:
        vowel_options = ['a', 'ou', 'er', 'en']
    elif hand_position == 2:
        vowel_options = ['i', 'v', 'ang']
    elif hand_position == 3:
        vowel_options = ['ai', 'u', 'ao']
    elif hand_position == 4:
        vowel_options = ['eng', 'ong', 'ei']
    else:
        vowel_options = ['<blank>']

    if hand_gesture == 0:
        consonant_options = ['p', 'd', 'zh']
    elif hand_gesture == 1:
        consonant_options = ['k', 'q', 'z']
    elif hand_gesture == 2:
        consonant_options = ['s', 'r', 'h']
    elif hand_gesture == 3:
        consonant_options = ['b', 'n', 'yu']
    elif hand_gesture == 4:
        consonant_options = ['m', 't', 'f']
    elif hand_gesture == 5:
        consonant_options = ['l', 'x', 'w']
    elif hand_gesture == 6:
        consonant_options = ['g', 'j', 'ch']
    elif hand_gesture == 7:
        consonant_options = ['y', 'c', 'sh']
    else:
        consonant_options = ['<blank>']

    return vowel_options, consonant_options