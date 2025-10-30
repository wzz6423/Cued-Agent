"""
Original Author: Liu Lei
Modified by Guanjie Huang
Date: 2025 March 12
"""

import torch
import torchaudio


def compute_cer(prediction, target):
    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """
    distance = torchaudio.functional.edit_distance(prediction.strip().lower().split(" "), target.strip().lower().split(" "))
    return distance / len(target.lower().split(" "))


#
def compute_wer(prediction, target):
    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    prediction_list = prediction.lower().strip().replace(" ", '').split('-')
    target_list = target.lower().strip().replace(" ", '').split('-')
    totalEdits = torchaudio.functional.edit_distance(prediction_list, target_list)

    return totalEdits / len(target_list)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    a="b ai - sh a - z ai - n y e - v - zh i - j v - h ei"
    b="b ai - sh a - z ai - n y e - v - zh i - j v - h ei"
    print(compute_cer(a,b))