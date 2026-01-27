import os
import pickle
import warnings

import torch
import torchaudio
import torchvision
import av

# 修复 av.AVError 兼容性问题
if not hasattr(av, 'AVError'):
    av.AVError = Exception

from .hand_util import load_hand_recog


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data


def load_video(path):
    """
    Load video from .mp4 or .pkl file.
    rtype: torch, T x C x H x W
    """
    if path.endswith('.pkl'):
        # Load preprocessed pickle file
        with open(path, 'rb') as f:
            vid = pickle.load(f)
        if isinstance(vid, torch.Tensor):
            return vid
        else:
            return torch.from_numpy(vid)
    else:
        # Load video file (mp4, etc.)
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset_CCS(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            label_path,
            subset,
            modality,
            audio_transform,
            video_transform,
            rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id, hand_recog_path, hand_position_path = path_count_label.split(",")
            paths_counts_labels.append(
                (
                    dataset_name,
                    rel_path,
                    int(input_length),
                    torch.tensor([int(_) for _ in token_id.split()]),
                    hand_recog_path,
                    hand_position_path
                )
            )
        return paths_counts_labels

    def __getitem__(self, idx):
        # Robustly skip corrupted samples (e.g., broken MP4 headers or tiny frames)
        max_attempts = 5
        for attempt in range(max_attempts):
            curr_idx = (idx + attempt) % len(self.list)
            dataset_name, rel_path, input_length, token_id, hand_recog_path, hand_position_path = self.list[curr_idx]
            path = os.path.join(self.root_dir, dataset_name, rel_path)

            if os.path.exists(path) is False:
                warnings.warn(f"{path} does not exist, skipping…")
                continue

            hand_available = bool(hand_recog_path and os.path.exists(hand_recog_path))

            try:
                if self.modality == "video":
                    video = load_video(path)
                    # Skip tiny/corrupted frames before transform
                    if video.numel() == 0 or video.shape[-2] < 88 or video.shape[-1] < 88:
                        raise ValueError(f"Invalid video size {video.shape}")
                    video = self.video_transform(video)
                    hand_recog_matrix = load_hand_recog(hand_recog_path, hand_position_path, input_length) if hand_available else torch.zeros((input_length, 44))
                    return {"input": video, "target": token_id, "hand_matrix": hand_recog_matrix}

                elif self.modality == "audio":
                    audio = load_audio(path)
                    audio = self.audio_transform(audio)
                    return {"input": audio, "target": token_id}

                elif self.modality == "audiovisual":
                    video = load_video(path)
                    if video.numel() == 0 or video.shape[-2] < 88 or video.shape[-1] < 88:
                        raise ValueError(f"Invalid video size {video.shape}")
                    audio = load_audio(path)
                    audio = cut_or_pad(audio, len(video) * self.rate_ratio)
                    video = self.video_transform(video)
                    audio = self.audio_transform(audio)
                    return {"video": video, "audio": audio, "target": token_id}

            except (ValueError, RuntimeError, OSError, av.AVError) as e:
                warnings.warn(f"Skipping corrupted sample {path} (idx={curr_idx}): {e}")
                continue

        raise RuntimeError(f"Failed to load a valid sample after {max_attempts} attempts starting from idx {idx}")

    def __len__(self):
        return len(self.list)
