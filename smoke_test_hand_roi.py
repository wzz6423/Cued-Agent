import os
import sys
import cv2
import numpy as np

# Ensure project root is on sys.path
proj_root = os.path.dirname(__file__)
if proj_root not in sys.path:
    sys.path.append(proj_root)

# Import the Inference module
import importlib
Inference = importlib.import_module('Inference')

# Monkeypatch _init_lip_and_decoding_agent to avoid heavy model initialization
def _init_lip_and_decoding_agent_noop(self, cfg):
    # Minimal attributes required by preprocess_video
    self.modelmodule = None
    import torch
    self.device = torch.device('cpu')

Inference.CuedAgentInference._init_lip_and_decoding_agent = _init_lip_and_decoding_agent_noop

# Build a simple config object similar to the example in Inference.py
class SimpleConfig:
    class Data:
        modality = 'video'
    class Model:
        class VisualBackbone:
            pass
        visual_backbone = VisualBackbone()
    data = Data()
    model = Model()
    pretrained_model_path = ''

# Create inference pipeline
cfg = SimpleConfig()
agent = Inference.CuedAgentInference(cfg, detector='mediapipe')

# Sample video path from repo
video_path = os.path.join(proj_root, 'HS-0001.mp4')
if not os.path.exists(video_path):
    print('Sample video not found:', video_path)
    sys.exit(2)

print('Running preprocess_video on:', video_path)
lip_video, hand_frames, hand_positions,_ = agent.preprocess_video(video_path)

print('Lip video shape:', getattr(lip_video, 'shape', None))
print('Number of hand frames extracted:', len(hand_frames))
print('Hand positions array shape:', np.array(hand_positions).shape)

# Save a few hand frames for visual inspection
out_dir = os.path.join(proj_root, 'tmp_hand_frames')
os.makedirs(out_dir, exist_ok=True)
num_save = min(20, len(hand_frames))
for i in range(num_save):
    frame = hand_frames[i]
    # Ensure frame is BGR for OpenCV write (it already is)
    out_path = os.path.join(out_dir, f'hand_frame_{i:03d}.jpg')
    cv2.imwrite(out_path, frame)

print(f'Saved {num_save} frames to', out_dir)
print('Sample hand positions (first 10):')
print(np.array(hand_positions)[:10])

