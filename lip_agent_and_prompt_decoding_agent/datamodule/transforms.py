#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import random
from g2p_en import G2p
import re


import numpy
import sentencepiece
import torch
import torchaudio
import torchvision
from torch import tensor

NOISE_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "babble_noise.wav"
)

SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class SpeedPerturb(torch.nn.Module):
    """
    Speed Perturbation for Video (Idea 2: Dynamic Changes)
    Randomly resamples the video along the time dimension.
    """
    def __init__(self, speeds=[0.9, 1.0, 1.1]):
        super().__init__()
        self.speeds = speeds

    def forward(self, x):
        # x: [T, C, H, W] or [C, T, H, W] depending on pipeline
        # In VideoTransform, input is [T, C, H, W] (from load_video permute) -> Grayscale -> [T, 1, H, W]
        
        speed = random.choice(self.speeds)
        if speed == 1.0:
            return x
            
        # Interpolate expects [Batch, Channel, Time, Height, Width]
        # or [Batch, Channel, Depth, Height, Width]
        # Current x is [T, C, H, W]
        
        x_permuted = x.permute(1, 0, 2, 3).unsqueeze(0) # [1, C, T, H, W]
        
        new_time = int(x.shape[0] / speed)
        
        x_resampled = torch.nn.functional.interpolate(
            x_permuted, 
            size=(new_time, x.shape[2], x.shape[3]), 
            mode='trilinear', 
            align_corners=False
        )
        
        # Back to [T, C, H, W]
        return x_resampled.squeeze(0).permute(1, 0, 2, 3)


class VideoRandomErasing(torch.nn.Module):
    """
    Random Erasing for Video (Spatial Augmentation)
    Randomly erases a rectangle region in the video frames.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def forward(self, img):
        # img: [T, C, H, W]
        if random.random() < self.p:
            # Apply same mask to all frames to simulate static occlusion (e.g. microphone, hand)
            # Or different mask? Static occlusion is harder and more realistic for "blocking".
            # Let's generate one mask parameters
            t, c, h, w = img.shape
            area = h * w

            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h_mask = int(round((target_area * aspect_ratio) ** 0.5))
            w_mask = int(round((target_area / aspect_ratio) ** 0.5))

            if h_mask < h and w_mask < w:
                y = random.randint(0, h - h_mask)
                x = random.randint(0, w - w_mask)

                img[:, :, y:y + h_mask, x:x + w_mask] = self.value

        return img


class TemporalDropout(torch.nn.Module):
    """
    Frame-level Dropout for temporal robustness
    Randomly drops frames to simulate missing/corrupted data
    """
    def __init__(self, p=0.1, contiguous=True, max_drop_ratio=0.2):
        super().__init__()
        self.p = p  # Probability of applying dropout
        self.contiguous = contiguous  # Whether to drop contiguous frames
        self.max_drop_ratio = max_drop_ratio  # Max ratio of frames to drop

    def forward(self, x):
        # x: [T, C, H, W]
        if random.random() > self.p:
            return x

        t = x.shape[0]
        if t < 5:
            return x

        num_drop = max(1, int(t * random.uniform(0.05, self.max_drop_ratio)))

        if self.contiguous:
            # Drop contiguous block
            start_idx = random.randint(0, t - num_drop)
            drop_indices = set(range(start_idx, start_idx + num_drop))
        else:
            # Drop random frames
            drop_indices = set(random.sample(range(t), num_drop))

        keep_indices = [i for i in range(t) if i not in drop_indices]
        if len(keep_indices) < 3:
            return x

        return x[keep_indices]


class RandomRotation(torch.nn.Module):
    """
    Random rotation for simulating head tilt variations
    """
    def __init__(self, degrees=15, p=0.5):
        super().__init__()
        self.degrees = degrees
        self.p = p

    def forward(self, x):
        # x: [T, C, H, W]
        if random.random() > self.p:
            return x

        angle = random.uniform(-self.degrees, self.degrees)
        return torchvision.transforms.functional.rotate(x, angle)


class RandomHorizontalFlip(torch.nn.Module):
    """
    Random horizontal flip for data augmentation
    Note: Lip reading may be symmetric, so flipping can help generalization
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: [T, C, H, W]
        if random.random() < self.p:
            return torch.flip(x, dims=[-1])  # Flip along width
        return x


class GaussianNoise(torch.nn.Module):
    """
    Add Gaussian noise for robustness to sensor noise
    """
    def __init__(self, std=0.05, p=0.3):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return x + noise


class ColorJitter(torch.nn.Module):
    """
    Apply brightness/contrast jitter for lighting robustness
    Works on grayscale by adjusting intensity
    """
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        # Random brightness
        if self.brightness > 0:
            brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
            x = x * brightness_factor

        # Random contrast
        if self.contrast > 0:
            contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = x.mean()
            x = (x - mean) * contrast_factor + mean

        return torch.clamp(x, 0, 1)


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=NOISE_FILENAME,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if os.path.exists(noise_filename):
            self.noise, sample_rate = torchaudio.load(noise_filename)
        else:
            print(f"  ⚠ Warning: Noise file {noise_filename} not found. AddNoise disabled.")
            self.noise = None

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None: return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoTransform:
    def __init__(self, subset, augment_level="standard"):
        """
        Video transform pipeline with configurable augmentation levels

        Args:
            subset: "train", "val", or "test"
            augment_level: "minimal", "standard", or "aggressive"
        """
        if subset == "train":
            # Base transforms always applied
            base_transforms = [
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
            ]

            # Augmentation based on level
            if augment_level == "minimal":
                aug_transforms = [
                    SpeedPerturb(speeds=[0.9, 1.0, 1.1]),
                    AdaptiveTimeMask(10, 25),
                ]
            elif augment_level == "aggressive":
                aug_transforms = [
                    SpeedPerturb(speeds=[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]),
                    RandomHorizontalFlip(p=0.5),
                    RandomRotation(degrees=10, p=0.3),
                    ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                    VideoRandomErasing(p=0.5, scale=(0.02, 0.25)),
                    TemporalDropout(p=0.2, contiguous=True, max_drop_ratio=0.15),
                    GaussianNoise(std=0.03, p=0.3),
                    AdaptiveTimeMask(10, 25),
                ]
            else:  # "standard" - default
                aug_transforms = [
                    SpeedPerturb(speeds=[0.9, 1.0, 1.1]),
                    RandomHorizontalFlip(p=0.3),
                    RandomRotation(degrees=8, p=0.2),
                    ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
                    VideoRandomErasing(p=0.5, scale=(0.02, 0.2)),
                    TemporalDropout(p=0.1, contiguous=True, max_drop_ratio=0.1),
                    AdaptiveTimeMask(10, 25),
                ]

            # Final normalization
            final_transforms = [
                torchvision.transforms.Normalize(0.421, 0.165),
            ]

            self.video_pipeline = torch.nn.Sequential(
                *base_transforms,
                *aug_transforms,
                *final_transforms
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, snr_target=None):
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddNoise(),
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                FunctionalModule(
                    lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                ),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)


class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
        target_vocab_size=None,  # Set to 5049 to match pretrained model
    ):

        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]

        # Pad to target vocab size if specified (for pretrained model compatibility)
        if target_vocab_size and len(self.token_list) < target_vocab_size:
            pad_count = target_vocab_size - len(self.token_list)
            for i in range(pad_count):
                self.token_list.append(f"<pad_{i}>")

        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")


class TextTransform_CCS:
    def __init__(self, units_file=None):
        # If units_file provided (e.g., multilingual), load units from it, else use built-in CCS units
        if units_file is not None and os.path.exists(units_file):
            units = open(units_file, encoding='utf8').read().splitlines()
            self.hashmap = {u: i + 1 for i, u in enumerate(units)}
        else:
            # Load default CCS mapping
            self.hashmap = {
                "<unk>":1,"b": 2, "p": 3, "m": 4, "f": 5, "d": 6, "t": 7, "n": 8, "l": 9, "g": 10, "k": 11, "h": 12, "j": 13,
                "q": 14, "x": 15, "zh": 16, "ch": 17, "sh": 18, "r": 19, "z": 20, "c": 21, "s": 22, "y": 23, "w": 24,
                "yu": 25, "a": 26, "o": 27, "e": 28, "i": 29, "u": 30, "v": 31, "ai": 32, "ei": 33, "ao": 34, "ou": 35, "er": 36,
                "an": 37, "en": 38, "ang": 39, "eng": 40, "ong": 41, "-": 42,
            }
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = text.strip().split(" ")
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return " ".join(token_as_list).replace("<space>", " ")

if __name__ == '__main__':
    ccs_text=TextTransform_CCS()
    #print(ccs_text.tokenize("ao er en"))
    token_ids=tensor([ 2, 32, 42, 18, 26, 42, 20, 32, 42, 8, 23, 28, 42, 31, 42, 16, 29, 42,
        13, 31, 42, 12, 33], device='cuda:0')
    print(ccs_text.post_process(token_ids))
