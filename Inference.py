import hydra

from lip_hand_seg_CS_latest import get_hand_margin, single_video_segment
from util.hand_decode import label2phone, hashmap

video_path = ""

import os
import sys
import numpy as np
import cv2
import base64
import json

import torch
import torchvision

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lip_agent_and_prompt_decoding_agent'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'hand_recognition_agent'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'self-p2w-agent'))

from lip_agent_and_prompt_decoding_agent.datamodule.transforms import VideoTransform
from lip_agent_and_prompt_decoding_agent.lightning_CCS_hand_prompt_decoding import ModelModule_CCS_Hand_prompt_decoding, get_beam_search_decoder
from hand_recognition_agent.CustomizedPromptTemplate import generate_recognition_single
from hand_recognition_agent.util import get_keyframes, Keyframe_filter, get_keyframe_groups

# Delayed import of self-p2w-agent module (path may need dynamic adjustment)
try:
    from PostProcess_deepseek import single_process
except ImportError:
    # If direct import fails, try importing from full path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "PostProcess_deepseek",
        os.path.join(os.path.dirname(__file__), 'self-p2w-agent', 'PostProcess_deepseek.py')
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        single_process = module.single_process
    else:
        def single_process(cued_sequence):
            """Placeholder function: if DeepSeek module cannot be imported"""
            raise ImportError("Cannot import PostProcess_deepseek module")


class CuedAgentInference:
    """
    Complete Cued Speech inference pipeline
    Input: Video file path
    Output: Cued Speech phoneme sequence and corresponding Chinese sentence
    """
    def __init__(self, cfg, detector="mediapipe", hand_weight=4.5, ctc_weight=0.5):
        """
        Initialize Cued Agent inference pipeline

        Args:
            cfg: Configuration object
            detector: Face detector type ("mediapipe" or "retinaface")
            hand_weight: Hand prompt weight
            ctc_weight: CTC loss weight
        """
        self.cfg = cfg
        self.modality = cfg.data.modality
        self.hand_weight = hand_weight
        self.ctc_weight = ctc_weight

        # Step 1: Initialize video preprocessing module (lip ROI extraction)
        print("[1/4] Initializing video preprocessing module...")
        self._init_video_preprocessor(detector)

        # Step 2: Initialize hand recognition agent
        print("[2/4] Initializing hand recognition agent...")
        self.support_set_path = os.path.join(os.path.dirname(__file__),
                                            'hand_recognition_agent', 'support_set')

        # Step 3: Initialize lip recognition and joint decoding agent
        print("[3/4] Initializing lip recognition and joint decoding agent...")
        self._init_lip_and_decoding_agent(cfg)

        # Step 4: Initialize self-correction P2W agent
        print("[4/4] Initializing self-correction agent...")
        # P2W Agent uses DeepSeek API, initialized when called

        print("✓ Cued Agent initialization complete!\n")

        self.total_frame_num = 0

    def _init_video_preprocessor(self, detector):
        """Initialize video preprocessor"""
        # Try to import project-specific wrappers (these may not exist in all setups).
        try:
            #if detector == "mediapipe":
            from util.mediapipe.detector import LandmarksDetector
            from util.mediapipe.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
            #elif detector == "retinaface":
                #from preparation.detectors.retinaface.detector import LandmarksDetector
                #from preparation.detectors.retinaface.video_process import VideoProcess
                #self.landmarks_detector = LandmarksDetector(device="cuda:0")
                #self.video_process = VideoProcess(convert_gray=False)
            #else:
                #raise ValueError(f"Unsupported detector type: {detector}")
        except Exception:
            # If project-specific wrappers are missing, provide lightweight fallbacks so
            # the rest of the pipeline can still execute. These fallbacks are simple
            # and do not perform high-quality landmark detection.
            print("  ⚠ Warning: project-specific detector/video_process not found, using fallback implementations.")

            # Fallback landmarks detector: returns None (no landmarks) when called with video array
            self.landmarks_detector = lambda video: None

            # Fallback video process: returns the original video (no lip cropping/transform)
            # Signature matches expected (video, landmarks) -> lip_video_array
            self.video_process = lambda video, landmarks: video

        # Video transform used by the lip recognition model
        self.video_transform = VideoTransform(subset="test")


    def _init_lip_and_decoding_agent(self, cfg):
        """Initialize lip recognition and joint decoding model"""
        self.modelmodule = ModelModule_CCS_Hand_prompt_decoding(
            cfg,
            hand_weight=self.hand_weight,
            ctc_weight=self.ctc_weight,
            output_results=False
        )

        # Load pretrained model
        if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
            self.modelmodule.load_state_dict(
                torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage).get('state_dict'))
            print(f"  ✓ Loaded pretrained model: {cfg.pretrained_model_path}")

        self.modelmodule.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelmodule.to(self.device)

    def preprocess_video(self, video_path):
        """
        Video preprocessing: extract hand and lip ROI

        Args:
            video_path: Input video path

        Returns:
            lip_video: Lip ROI video tensor
            hand_video_frames: Hand ROI video frames (for hand recognition)
            hand_position_npy: Hand position information (for keyframe extraction)
        """
        print("\n=== Step 1: Video Preprocessing ===")
        print(f"Processing video: {video_path}")

        # Load original video
        video_data = self.load_video(video_path)
        print(f"  Video shape: {video_data.shape}")
        self.total_frame_num = video_data.shape[0]

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

        # Extract hand ROI (assuming hand extraction functionality exists, otherwise use original video)
        # Note: This needs to be adjusted based on actual hand extraction method
        print("  Extracting hand ROI...")
        hand_video_frames, hand_position_npy,seg_frame_indexes = single_video_segment(video_path)
        print(f"  Hand ROI Shape: {hand_video_frames.shape}")



        # Save hand position information for keyframe extraction

        return lip_video, hand_video_frames, hand_position_npy,seg_frame_indexes




    def hand_recognition(self, hand_video_frames, hand_position_npy,seg_frame_indexes):
        """
        Hand recognition: Use GPT-4V to recognize hand shape and position

        Args:
            hand_video_frames: Hand video frame list
            hand_position_npy: Hand position array

        Returns:
            hand_matrix: Hand recognition result matrix [T, 2] (position and shape)
        """
        print("\n=== Step 2: Hand Recognition ===")

        # Extract keyframes
        print("  Extracting keyframes...")

        try:
            keyframe_indices = Keyframe_filter(hand_position_npy)
            print(f"  Keyframe indices: {keyframe_indices}")
        except:
            print("  Keyframe indices error")

        # Convert keyframes to base64 encoding
        print("  Encoding keyframes...")
        hand_frames_base64 = []
        for idx in keyframe_indices:
            if idx < len(hand_video_frames):
                frame = hand_video_frames[idx]
                _, buffer = cv2.imencode(".jpg", frame)
                # cv2.imencode returns a numpy ndarray; convert to bytes before base64 encoding
                base64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
                hand_frames_base64.append(base64_str)

        # Call GPT-4V for hand recognition
        print(f"  Calling GPT-4V to recognize {len(hand_frames_base64)} keyframes...")
        try:
            recog_results, token_usage = generate_recognition_single(
                hand_frames_base64,
                support_set_path=self.support_set_path
            )
            print(f"  ✓ Hand recognition complete, tokens used: {token_usage}")
        except Exception as e:
            print(f"  ⚠ Hand recognition failed: {e}")
            print("  Using default hand prompts...")
            recog_results = [{"frame_id": i, "hand_position": 0, "hand_shape": 0}
                           for i in range(len(keyframe_indices))]

        # Build hand matrix (interpolate to all frames)
        hand_matrix = self._build_hand_matrix(recog_results, keyframe_indices,seg_frame_indexes)
        print(f"  Hand matrix shape: {hand_matrix.shape}")

        return hand_matrix

    def _build_hand_matrix(self, recog_results, keyframe_indices,hand_position_npy):
        """Build complete hand matrix from keyframe recognition results"""

        hand_matrix = torch.zeros(self.total_frame_num, 44)


        # slow_groups = get_nokeyframe_groups(key_frames,frame_num)
        slow_groups = get_keyframe_groups(hand_position_npy)
        print("slow_groups:", slow_groups)
        quit()
        for i in range(len(keyframe_indices)):
            hand_position = recog_results[i]['hand_position']
            hand_gesture = recog_results[i]['hand_gesture']
            vowel_options, consonant_options = label2phone(hand_position, hand_gesture)

            for vowel in vowel_options:
                vowel_index = hashmap[vowel]
                for j in range(slow_groups[i][0], slow_groups[i][-1] + 1):
                    hand_matrix[j][vowel_index] = 1

            for consonant in consonant_options:
                consonant_index = hashmap[consonant]
                for j in range(slow_groups[i][0], slow_groups[i][-1] + 1):
                    hand_matrix[j][consonant_index] = 1

        return hand_matrix





    def lip_hand_joint_decoding(self, lip_video, hand_matrix):
        """
        Lip and hand joint decoding

        Args:
            lip_video: Lip video tensor
            hand_matrix: Hand recognition result matrix

        Returns:
            cued_sequence: Cued Speech phoneme sequence
        """
        print("\n=== Step 3: Lip and Hand Joint Decoding ===")

        with torch.no_grad():
            # Prepare input data
            lip_video = lip_video.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
            hand_matrix = torch.tensor(hand_matrix, dtype=torch.float32).to(self.device)

            print(f"  Lip video shape: {lip_video.shape}")
            print(f"  Hand matrix shape: {hand_matrix.shape}")

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

            # Joint decoding
            print("  Performing joint decoding...")
            nbest_hyps = beam_search(enc_feat, hand_matrix=hand_matrix)
            nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]

            # Extract prediction result
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            cued_sequence = self.modelmodule.text_transform.post_process(predicted_token_id)
            cued_sequence = cued_sequence.replace("<eos>", "").strip()

            print(f"  ✓ Preliminary Cued Speech sequence: {cued_sequence}")

        return cued_sequence

    def self_correction(self, cued_sequence):
        """
        Self-correction: Use DeepSeek model to correct Cued Speech sequence

        Args:
            cued_sequence: Preliminary Cued Speech sequence

        Returns:
            final_result: Dictionary containing corrected sequence, pinyin and Chinese sentence
        """
        print("\n=== Step 4: Self-correction P2W ===")
        print(f"  Input sequence: {cued_sequence}")

        try:
            print("  Calling DeepSeek for post-processing...")
            response = single_process(cued_sequence)

            # Parse response
            content = response.choices[0].message.content
            print(f"  ✓ DeepSeek response received successfully")

            # Try to parse JSON format result
            try:
                # Extract JSON part
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                else:
                    json_str = content

                result = json.loads(json_str)

            except json.JSONDecodeError:
                # If JSON parsing fails, extract manually
                print("  ⚠ JSON parsing failed, using text parsing...")
                result = {
                    "Processed_Cued_Speech_Sequence": cued_sequence,
                    "Pinyin_Sequence": "",
                    "Mandarin_Sequence": "",
                    "Reasoning_Process": content
                }

            print(f"  ✓ Corrected sequence: {result.get('Processed_Cued_Speech_Sequence', cued_sequence)}")
            print(f"  ✓ Pinyin sequence: {result.get('Pinyin_Sequence', 'N/A')}")
            print(f"  ✓ Chinese sentence: {result.get('Mandarin_Sequence', 'N/A')}")

            return result

        except Exception as e:
            print(f"  ⚠ Self-correction failed: {e}")
            print("  Returning uncorrected result...")
            return {
                "Processed_Cued_Speech_Sequence": cued_sequence,
                "Pinyin_Sequence": "",
                "Mandarin_Sequence": "",
                "Reasoning_Process": f"Error: {str(e)}"
            }

    def __call__(self, video_path):
        """
        Complete inference pipeline

        Args:
            video_path: Input video path

        Returns:
            result: Dictionary containing final recognition result
        """
        print(f"\n{'='*60}")
        print(f"Starting Cued Speech Inference")
        print(f"{'='*60}")

        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Step 1: Video preprocessing
        lip_video, hand_video_frames, hand_position_npy,seg_frame_indexes = self.preprocess_video(video_path)

        # Step 2: Hand recognition
        hand_matrix = self.hand_recognition(hand_video_frames, hand_position_npy,seg_frame_indexes)

        # Step 3: Lip and hand joint decoding
        cued_sequence = self.lip_hand_joint_decoding(lip_video, hand_matrix)

        # Step 4: Self-correction
        final_result = self.self_correction(cued_sequence)

        print(f"\n{'='*60}")
        print(f"Inference Complete!")
        print(f"{'='*60}")
        print(f"Final Result:")
        print(f"  Cued Speech Sequence: {final_result['Processed_Cued_Speech_Sequence']}")
        print(f"  Pinyin Sequence: {final_result['Pinyin_Sequence']}")
        print(f"  Chinese Sentence: {final_result['Mandarin_Sequence']}")
        print(f"{'='*60}\n")

        return final_result

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
            hand_weight=4.5,
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
