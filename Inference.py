import hydra
import os
import sys
import numpy as np
import cv2
import json
import torch
import torchvision

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lip_agent_and_prompt_decoding_agent'))

from lip_agent_and_prompt_decoding_agent.datamodule.transforms import VideoTransform
from lip_agent_and_prompt_decoding_agent.lightning_CCS import ModelModule_CCS, get_beam_search_decoder

class CuedAgentInference:
    """
    Lip Reading Inference Pipeline
    Input: Video file path
    Output: Cued Speech phoneme sequence
    """
    def __init__(self, cfg, detector="mediapipe", ctc_weight=0.5):
        self.cfg = cfg
        self.modality = cfg.data.modality
        self.ctc_weight = ctc_weight

        # Step 1: Initialize video preprocessing module (lip ROI extraction)
        print("[1/2] Initializing video preprocessing module...")
        self._init_video_preprocessor(detector)

        # Step 2: Initialize lip recognition agent
        print("[2/2] Initializing lip recognition agent...")
        self._init_lip_agent(cfg)

        print("✓ Lip Reading Agent initialization complete!\n")

    def _init_video_preprocessor(self, detector):
        """Initialize video preprocessor"""
        try:
            from util.mediapipe.detector import LandmarksDetector
            from util.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        except Exception:
            print("  ⚠ Warning: project-specific detector/video_process not found, using fallback implementations.")
            self.landmarks_detector = lambda video: None
            self.video_process = lambda video, landmarks: video

        self.video_transform = VideoTransform(subset="test")

    def _init_lip_agent(self, cfg):
        """Initialize lip recognition model"""
        self.modelmodule = ModelModule_CCS(cfg)

        # Load pretrained model
        if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
            ckpt = torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
            if 'state_dict' in ckpt:
                self.modelmodule.load_state_dict(ckpt['state_dict'])
            else:
                self.modelmodule.load_state_dict(ckpt)
            print(f"  ✓ Loaded pretrained model: {cfg.ckpt_path}")

        self.modelmodule.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelmodule.to(self.device)

    def preprocess_video(self, video_path):
        """
        Video preprocessing: extract lip ROI
        """
        print("\n=== Step 1: Video Preprocessing ===")
        print(f"Processing video: {video_path}")

        # Load original video
        video_data = self.load_video(video_path)
        print(f"  Video shape: {video_data.shape}")

        # Detect facial landmarks
        print("  Detecting facial landmarks...")
        landmarks = self.landmarks_detector(video_data)

        # Extract lip ROI
        print("  Extracting lip ROI...")
        lip_video = self.video_process(video_data, landmarks)
        lip_video = torch.tensor(lip_video)
        lip_video = lip_video.permute((0, 3, 1, 2))  # [T, H, W, C] -> [T, C, H, W]
        lip_video = self.video_transform(lip_video)
        print(f"  Lip ROI shape: {lip_video.shape}")

        return lip_video

    def lip_decoding(self, lip_video):
        """
        Lip decoding
        """
        print("\n=== Step 2: Lip Decoding ===")

        with torch.no_grad():
            # Prepare input data
            lip_video = lip_video.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

            # Encoder feature extraction
            print("  Encoding features...")
            enc_feat, _ = self.modelmodule.model.encoder(lip_video, None)
            enc_feat = enc_feat.squeeze(0)

            # Initialize Beam Search decoder
            print("  Initializing Beam Search decoder...")
            beam_search = get_beam_search_decoder(
                self.modelmodule.model,
                self.modelmodule.token_list,
                ctc_weight=self.ctc_weight
            )

            # Decoding
            print("  Performing decoding...")
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]

            # Extract prediction result
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            cued_sequence = self.modelmodule.text_transform.post_process(predicted_token_id)
            cued_sequence = cued_sequence.replace("<eos>", "").strip()

            print(f"  ✓ Predicted sequence: {cued_sequence}")

        return cued_sequence

    def __call__(self, video_path):
        print(f"\n{'='*60}")
        print(f"Starting Lip Reading Inference")
        print(f"{'='*60}")

        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Step 1: Video preprocessing
        lip_video = self.preprocess_video(video_path)

        # Step 2: Lip decoding
        cued_sequence = self.lip_decoding(lip_video)

        result = {
            "Processed_Cued_Speech_Sequence": cued_sequence,
        }

        print(f"\n{'='*60}")
        print(f"Inference Complete!")
        print(f"{'='*60}")
        print(f"Final Result:")
        print(f"  Sequence: {result['Processed_Cued_Speech_Sequence']}")
        print(f"{'='*60}\n")

        return result

    def load_video(self, video_path):
        """Load video file"""
        return torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()

@hydra.main(version_base="1.3", config_path="lip_agent_and_prompt_decoding_agent/configs", config_name="config_CCS_hand_infer")
def main(cfg):
    """Main function for testing"""
    video_path = cfg.file_path
    try:
        inference_pipeline = CuedAgentInference(
            cfg,
            detector="mediapipe",
            ctc_weight=0.5
        )

        if os.path.exists(video_path):
            result = inference_pipeline(video_path)

            # Save result
            output_path = "inference_result.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to: {output_path}")
        else:
            print(f"⚠ Video file not found: {video_path}")
            print("Please set correct video path in the code")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
   main()
