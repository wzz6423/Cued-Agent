import hydra
import os
import sys
import numpy as np
import cv2
import json
import torch
import torchvision
from pathlib import Path

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lip_agent_and_prompt_decoding_agent'))

from lip_agent_and_prompt_decoding_agent.datamodule.transforms import VideoTransform
from lip_agent_and_prompt_decoding_agent.lightning_CCS import ModelModule_CCS, get_beam_search_decoder
from utils.llm_postprocess import LLMPostProcessor # Import LLM Processor

class CuedAgentInference:
    """
    Lip Reading Inference Pipeline
    Input: Video file path
    Output: Cued Speech phoneme sequence
    """
    def __init__(self, cfg, detector="mediapipe", ctc_weight=0.3, beam_size=20, max_frames=None, skip_errors=False, ctc_threshold=0.08):
        self.cfg = cfg
        self.modality = cfg.data.modality
        # Optimized CTC weight: 0.3 balances attention and CTC better than 0.5
        self.ctc_weight = ctc_weight
        # Optimized beam size: 20 provides good accuracy/speed tradeoff
        # (40 was overkill for most cases, 10 too small for diverse outputs)
        self.beam_size = beam_size
        self.max_frames = max_frames
        self.skip_errors = skip_errors
        # Reduced CTC threshold: 0.08 allows more valid tokens through
        # (0.15 was too aggressive, filtering out real speech)
        self.ctc_threshold = ctc_threshold

        # Initialize LLM Post-processor with language support
        language = getattr(cfg, "language", "mixed")
        self.llm_processor = LLMPostProcessor(language=language)

        # Step 1: Initialize video preprocessing module (lip ROI extraction)
        print("[1/2] Initializing video preprocessing module...")
        self._init_video_preprocessor(detector)

        # Step 2: Initialize lip recognition agent
        print("[2/2] Initializing lip recognition agent...")
        self._init_lip_agent(cfg)

        print("✓ Lip Reading Agent initialization complete!\n")

    def _init_video_preprocessor(self, detector):
        """Initialize video preprocessor with enhanced CLAHE parameters"""
        try:
            from utils.video_processing.detector import LandmarksDetector
            from utils.video_processing.video_process import VideoProcess
            self.landmarks_detector = LandmarksDetector()
            # Enhanced CLAHE parameters for better lip feature extraction
            # clipLimit=2.0: Higher contrast enhancement for lip details
            # tileGridSize=(6,6): Smaller grid for more local adaptation
            # IMPORTANT: Disable CLAHE for pretrained Auto-AVSR model
            # The original model was NOT trained with CLAHE preprocessing
            self.video_process = VideoProcess(
                convert_gray=True,
                use_clahe=False,  # Must be False for pretrained model compatibility
                crop_width=96,
                crop_height=96
            )
        except Exception as e:
            print(f"  ⚠ Warning: project-specific detector/video_process not found: {e}")
            self.landmarks_detector = lambda video: None
            self.video_process = lambda video, landmarks: video

        # Grayscale transform in torchvision as backup
        self.video_transform = VideoTransform(subset="test")

    def _init_lip_agent(self, cfg):
        """Initialize lip recognition model with improved language support"""
        # Pass config object directly - ModelModule_CCS now handles it without save_hyperparameters

        # Determine vocabulary type and size
        vocab_type = getattr(cfg, "vocab_type", "project")
        language = getattr(cfg, "language", "en")
        self.vocab_type = vocab_type

        if vocab_type == "base":
            print(f"  Using BASE vocabulary (BPE 5000) for {language}...")
            print(f"  ✓ Full pretrained weights will be loaded (encoder + decoder)")
            from lip_agent_and_prompt_decoding_agent.datamodule.transforms import TextTransform

            # Check if SPM files exist
            spm_path = getattr(cfg, "spm_model_path", "")
            dict_path = getattr(cfg, "dict_path", "")

            if not os.path.exists(spm_path) or os.path.getsize(spm_path) < 10000:
                print(f"  ⚠ SPM model not found or too small: {spm_path}")
                print(f"  Please run: curl -L -k -o lip_agent_and_prompt_decoding_agent/spm/unigram/unigram5000.model https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model")
                vocab_type = "project"
            else:
                try:
                    # Create TextTransform with BPE vocabulary
                    # Use target_vocab_size=5049 to match pretrained model
                    self.text_transform_bpe = TextTransform(
                        sp_model_path=spm_path,
                        dict_path=dict_path,
                        target_vocab_size=5049  # Match pretrained model vocab size
                    )

                    # Create a custom ModelModule for BPE vocabulary
                    # We need to override the default text_transform
                    self.modelmodule = self._create_bpe_model_module(cfg, self.text_transform_bpe)
                    print(f"  ✓ BPE vocabulary size: {len(self.modelmodule.token_list)}")

                    # Initialize empty token map for base vocab (no language filtering needed for BPE)
                    self.token_language_map = {}

                except Exception as e:
                    import traceback
                    print(f"  ⚠ Failed to load Base SPM files: {e}")
                    traceback.print_exc()
                    vocab_type = "project"

        if vocab_type == "project":
            print(f"  Using PROJECT vocabulary (82 units) for {language}...")
            print(f"  ⚠ Note: Decoder weights won't match pretrained model (shape mismatch)")
            self.modelmodule = ModelModule_CCS(cfg)
            self.vocab_type = "project"

            # Build language token mapping for intelligent filtering
            self._build_language_token_map()

        # Load model weights
        self._load_model_weights(cfg)

    def _create_bpe_model_module(self, cfg, text_transform):
        """Create a ModelModule with BPE vocabulary for full pretrained weight loading"""
        from lip_agent_and_prompt_decoding_agent.espnet.nets.pytorch_backend.e2e_asr_conformer_clean import E2E_Clean

        class BPEModelModule(torch.nn.Module):
            def __init__(self, cfg, text_transform):
                super().__init__()
                self.cfg = cfg
                self.text_transform = text_transform
                self.token_list = text_transform.token_list

                # Get backbone args
                if cfg.data.modality == "audio":
                    self.backbone_args = cfg.model.audio_backbone
                elif cfg.data.modality == "video":
                    self.backbone_args = cfg.model.visual_backbone

                # Initialize E2E_Clean model (no DynamicFeatureModule, no ctc_inter)
                self.model = E2E_Clean(len(self.token_list), self.backbone_args)

        module = BPEModelModule(cfg, text_transform)
        return module

    def _load_model_weights(self, cfg):
        """Load pretrained weights into the model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelmodule.to(self.device)
        self.modelmodule.eval()

        loaded = False
        # 1. Primary Checkpoint
        ckpt_path = cfg.pretrained_model_path
        if ckpt_path and os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1024:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

                # Intelligent mapping for mismatched headers
                model_dict = self.modelmodule.state_dict()
                filtered_dict = {}
                matched_count = 0
                skipped_count = 0

                for k, v in state_dict.items():
                    # Map checkpoint keys to model keys
                    # Checkpoint: frontend.xxx -> Model: model.encoder.frontend.xxx
                    # Checkpoint: encoder.xxx -> Model: model.encoder.xxx
                    # Checkpoint: decoder.xxx -> Model: model.decoder.xxx
                    # Checkpoint: ctc.xxx -> Model: model.ctc.xxx
                    # Checkpoint: proj_encoder.xxx -> Model: model.encoder.embed.0.xxx

                    if k.startswith("frontend."):
                        # frontend.xxx -> model.encoder.frontend.xxx
                        key = "model.encoder." + k
                    elif k.startswith("proj_encoder."):
                        # proj_encoder.weight -> model.encoder.embed.0.weight
                        key = "model.encoder.embed.0." + k.replace("proj_encoder.", "")
                    elif k.startswith("model."):
                        # Already has model. prefix
                        key = k
                    else:
                        # encoder.xxx, decoder.xxx, ctc.xxx
                        key = "model." + k

                    if key in model_dict:
                        if v.shape == model_dict[key].shape:
                            filtered_dict[key] = v
                            matched_count += 1
                        else:
                            skipped_count += 1
                    # Skip keys that don't exist in model (e.g., dynamic_feature_module is new)

                self.modelmodule.load_state_dict(filtered_dict, strict=False)
                print(f"  ✓ Loaded {matched_count} weight tensors from: {ckpt_path}")
                if skipped_count > 0:
                    print(f"  ⚠ Skipped {skipped_count} tensors due to shape mismatch")
                loaded = True
            except Exception as e:
                print(f"  ⚠ Failed to load checkpoint: {e}")

        if not loaded:
            print("  ⚠ WARNING: No weights loaded. Running with random initialization.")

        self.modelmodule.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelmodule.to(self.device)

    def _build_language_token_map(self):
        """Build intelligent token mapping for language filtering"""
        # Chinese Pinyin tokens (based on TextTransform_CCS hashmap)
        chinese_tokens = {
            # Finals/Initials that are distinctly Chinese
            "zh", "ch", "sh", "yu", "ao", "ou", "er", "an", "en", "ang", "eng", "ong",
            # Common Chinese syllables
            "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"
        }

        # English ARPAbet-style tokens (common phonemes)
        english_tokens = {
            # Vowels
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
            # Consonants
            "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"
        }

        # Build reverse lookup: token_id -> language
        self.token_language_map = {}
        token_list = self.modelmodule.token_list

        for idx, token in enumerate(token_list):
            token_clean = token.strip().lower()
            if token_clean in chinese_tokens or token_clean in {"a", "o", "e", "i", "u", "v", "ai", "ei"}:
                self.token_language_map[idx] = "zh"
            elif token_clean.upper() in english_tokens or len(token) > 1 and token[0].isupper():
                self.token_language_map[idx] = "en"
            else:
                # Ambiguous or special tokens
                self.token_language_map[idx] = "both"

        print(f"  ✓ Built language token map: {sum(1 for v in self.token_language_map.values() if v=='zh')} Chinese, "
              f"{sum(1 for v in self.token_language_map.values() if v=='en')} English, "
              f"{sum(1 for v in self.token_language_map.values() if v=='both')} Shared")

    def preprocess_video(self, video_path):
        """
        Video preprocessing: extract lip ROI
        """
        print("\n=== Step 1: Video Preprocessing ===")
        print(f"Processing video: {video_path}")

        # Load original video
        video_data = self.load_video(video_path)
        print(f"  Input frame count: {len(video_data)}")

        # Detect facial landmarks
        print("  Detecting facial landmarks...")
        landmarks = self.landmarks_detector(video_data)
        if landmarks is None or all(lm is None for lm in landmarks):
             print("  ⚠ Failed to detect faces. Falling back to center crop.")
             # Mock landmarks for centercrop fallback
             # VideoProcess expects 4 points: right_eye, left_eye, nose, mouth_center
             t, h, w, c = video_data.shape
             cx, cy = w // 2, h // 2
             # Create approximate face landmark positions relative to center
             landmarks = [np.array([
                 [cx - 30, cy - 30],  # right eye (approximate)
                 [cx + 30, cy - 30],  # left eye (approximate)
                 [cx, cy],            # nose tip
                 [cx, cy + 30]        # mouth center
             ]) for _ in range(t)]

        # Extract lip ROI
        print("  Extracting lip ROI...")
        # Note: VideoProcess(convert_gray=True) already produces grayscale 96x96 or 88x88
        lip_video_np = self.video_process(video_data, landmarks)
        
        # Standardize based on Auto-AVSR expectations (Standard for many lip reading models)
        # Even if VideoProcess did CLAHE, we ensure consistency here
        processed_frames = []
        for frame in lip_video_np:
            # Handle grayscale or RGB
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Standardization: resizing to 88x88 is handled by VideoTransform's CenterCrop if it was 96.
            # But we can force it here for safety.
            processed_frames.append(gray)
        
        # Convert to tensor [T, 1, H, W]
        # VideoTransform("test"): CenterCrop(88) -> Grayscale -> Normalize(0.421, 0.165)
        lip_video = torch.tensor(np.array(processed_frames)).unsqueeze(1) # [T, 1, H, W]
        
        # Apply transformation pipeline (Crop + Normalize)
        # This handles /255.0 and normalization to N(0.421, 0.165)
        lip_video = self.video_transform(lip_video)
        print(f"  Lip ROI shape: {lip_video.shape}")
        
        return lip_video

    def greedy_decode(self, enc_feat):
        """Perform greedy decoding for faster inference when beam_size=1."""
        print("  Using greedy decoding for beam_size=1...")
        
        model = self.modelmodule.model
        sos = model.sos
        eos = model.eos
        
        yseq = torch.tensor([sos], dtype=torch.long, device=enc_feat.device)
        score = 0.0
        maxlen = min(200, enc_feat.shape[0] * 2)
        
        for i in range(maxlen):
            decoder_scores, _ = model.decoder.score(yseq, None, enc_feat)
            probs = torch.softmax(decoder_scores, dim=-1)
            max_prob, next_token = torch.max(probs, dim=-1)
            next_token = next_token.item()

            if i > 5 and max_prob < self.ctc_threshold:
                break
            
            yseq = torch.cat([yseq, torch.tensor([next_token], dtype=torch.long, device=enc_feat.device)])
            score += decoder_scores[next_token].item()
            if next_token == eos:
                break
        
        return [{
            "yseq": yseq.tolist(),
            "score": score,
            "scores": {"decoder": score, "ctc": score},
            "states": {}
        }]

    def _clean_sequence(self, sequence, frame_count=None):
        """
        Enhanced sequence cleaning with multi-stage processing:
        1. Sequential deduplication (CTC-style)
        2. Loop suppression (detects A-B-C-A-B-C patterns)
        3. Noise token filtering
        4. Length-based heuristics
        """
        if not sequence: return ""
        parts = sequence.split()
        if not parts: return ""

        # Stage 1: Basic Sequential Dedup (CTC style)
        collapsed = [parts[0]]
        for i in range(1, len(parts)):
            if parts[i] != parts[i-1]:
                collapsed.append(parts[i])

        # Stage 2: Enhanced Loop Suppression
        # Detects repeating patterns (A-B-C-A-B-C or A-A-A) and removes them
        tokens = collapsed
        max_iterations = 5  # Prevent infinite loops
        for _ in range(max_iterations):
            n = len(tokens)
            if n <= 4: break  # Too short for patterns

            changed = False
            # Try pattern lengths from 1 to 5
            for pattern_len in range(1, min(6, n // 2 + 1)):
                i = 0
                new_tokens = []
                while i < n:
                    # Check if we have a repeating pattern starting at i
                    if i + 2 * pattern_len <= n:
                        pattern = tokens[i:i + pattern_len]
                        next_segment = tokens[i + pattern_len:i + 2 * pattern_len]

                        if pattern == next_segment:
                            # Found a repeat! Skip the duplicate
                            new_tokens.extend(pattern)
                            i += 2 * pattern_len
                            changed = True
                            # Skip additional repeats of the same pattern
                            while i + pattern_len <= n and tokens[i:i + pattern_len] == pattern:
                                i += pattern_len
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                if changed:
                    tokens = new_tokens
                    break

            if not changed:
                break

        # Stage 3: Noise Token Filtering
        # Remove common noise tokens and artifacts
        noise_tokens = {"<blank>", "<unk>", "<pad>", "-", ".", "|", "_", "sil", "sp", "spn", "<s>", "</s>", "<eos>"}
        # Also filter pad tokens like <pad_0>, <pad_1>, etc. and clean embedded special markers
        filtered_tokens = []
        for t in tokens:
            # Skip basic noise tokens
            if t.lower() in noise_tokens:
                continue
            # Skip pad tokens
            if t.startswith("<pad_"):
                continue
            # Clean up embedded special markers (e.g., "MOONARIAN<pad_32>" -> "MOONARIAN")
            for marker in ["<pad_32>", "<pad_46>", "<s>", "</s>", "<eos>", "<unk>"]:
                t = t.replace(marker, "")
            # Remove any remaining angle bracket content
            import re
            t = re.sub(r'<[^>]+>', '', t)
            if t.strip():  # Only add non-empty tokens
                filtered_tokens.append(t.strip())
        tokens = filtered_tokens

        # Stage 4: Length-based Heuristics
        if frame_count is not None:
            # Heuristic: phoneme rate should be roughly 8-15 per second (at 25fps)
            expected_max_tokens = int(frame_count / 25 * 15) + 10  # Add buffer
            if len(tokens) > expected_max_tokens:
                # Sequence too long, likely has repetition noise
                # Truncate with priority to unique tokens
                seen = []
                for t in tokens:
                    if t not in seen[-3:]:  # Allow token if not in last 3
                        seen.append(t)
                    if len(seen) >= expected_max_tokens:
                        break
                tokens = seen

        # Stage 5: Remove isolated single characters (often noise)
        # But keep them if they're part of a longer sequence
        if len(tokens) > 3:
            cleaned = []
            for i, token in enumerate(tokens):
                # Keep token if:
                # - It's multi-character, OR
                # - It's surrounded by similar tokens
                if len(token) > 1:
                    cleaned.append(token)
                else:
                    # Check neighbors
                    has_neighbor = False
                    if i > 0 and len(tokens[i-1]) == 1:
                        has_neighbor = True
                    if i < len(tokens) - 1 and len(tokens[i+1]) == 1:
                        has_neighbor = True
                    if has_neighbor or i == 0 or i == len(tokens) - 1:
                        cleaned.append(token)
            tokens = cleaned if cleaned else tokens

        return " ".join(tokens)

    def lip_decoding(self, lip_video):
        """Lip decoding with enhanced beam search and confidence filtering"""
        print("\n=== Step 2: Lip Decoding ===")

        with torch.no_grad():
            lip_video = lip_video.unsqueeze(0).to(self.device).float()

            if hasattr(self, 'max_frames') and self.max_frames is not None:
                if lip_video.shape[1] > self.max_frames:
                    lip_video = lip_video[:, : self.max_frames, ...]

            # E2E_Clean model doesn't have DynamicFeatureModule - skip it
            # The clean model matches the original Auto-AVSR architecture exactly

            enc_feat, _ = self.modelmodule.model.encoder(lip_video, None)
            enc_feat = enc_feat.squeeze(0)

            print("  Performing decoding...")
            beam_size = getattr(self, 'beam_size', 20)
            ctc_weight = self.ctc_weight

            try:
                if beam_size == 1:
                    nbest_hyps = self.greedy_decode(enc_feat)
                else:
                    # Optimized length penalty: 0.0 for no penalty (let model decide)
                    # Previous 0.6 was biasing toward shorter sequences
                    length_penalty = getattr(self.cfg, "length_penalty", 0.0)
                    beam_search = get_beam_search_decoder(
                        self.modelmodule.model,
                        self.modelmodule.token_list,
                        ctc_weight=ctc_weight,
                        beam_size=beam_size,
                        length_penalty=length_penalty
                    )
                    nbest_hyps = beam_search(enc_feat)
                    # Increased from 10 to min(beam_size, 15) for more candidate diversity
                    nbest_hyps = [h.asdict() for h in nbest_hyps[:min(beam_size, 15)]]

                candidates = []
                candidate_scores = []
                frame_count = lip_video.shape[1]
                language_constraint = getattr(self.cfg, "language", "mixed") # Default to mixed for flexibility

                # Confidence threshold for filtering low-quality hypotheses
                min_confidence = getattr(self.cfg, "min_confidence", -100.0)

                for hyp in nbest_hyps:
                    # Get hypothesis score for confidence filtering
                    hyp_score = hyp.get("score", 0)

                    # Skip low confidence hypotheses
                    if hyp_score < min_confidence:
                        continue

                    token_ids = list(map(int, hyp["yseq"][1:]))

                    # Intelligent language filtering using token_language_map
                    filtered_ids = []
                    for tid in token_ids:
                        # Get language of this token
                        token_lang = self.token_language_map.get(tid, "both")

                        # Apply language constraint
                        if language_constraint == "en" and token_lang == "zh":
                            continue  # Skip Chinese tokens in English mode
                        elif language_constraint == "zh" and token_lang == "en":
                            continue  # Skip English tokens in Chinese mode
                        elif language_constraint == "mixed":
                            # Accept all tokens in mixed mode
                            filtered_ids.append(tid)
                        else:
                            # Accept token if it matches language or is shared
                            if token_lang in [language_constraint, "both"]:
                                filtered_ids.append(tid)

                    if not filtered_ids: continue
                    token_ids_tensor = torch.tensor(filtered_ids)

                    raw = self.modelmodule.text_transform.post_process(token_ids_tensor).replace("<eos>", "").strip()
                    cleaned = self._clean_sequence(raw, frame_count=frame_count)

                    if cleaned and cleaned not in candidates:
                        candidates.append(cleaned)
                        candidate_scores.append(hyp_score)

                # Apply diversity penalty to re-rank candidates
                if len(candidates) > 1:
                    candidates, candidate_scores = self._apply_diversity_penalty(candidates, candidate_scores)

                print(f"  ✓ Candidates: {candidates[:3]}")
                # Pass original video context if possible
                cued_sequence = candidates[0] if candidates else ""

                # LLM post-processing (skip if disabled)
                if self.llm_processor is not None:
                    corrected_sentence = self.llm_processor.correct(candidates)
                    print(f"  ✓ LLM Corrected: {corrected_sentence}")
                else:
                    corrected_sentence = cued_sequence
                    print("  ℹ LLM disabled, using raw sequence")

            except Exception as e:
                import traceback
                print(f"  ⚠ Decoding failed: {e}")
                traceback.print_exc()
                cued_sequence = ""
                corrected_sentence = f"Error: {e}"

        return cued_sequence, corrected_sentence

    def _apply_diversity_penalty(self, candidates, scores, diversity_weight=0.5):
        """
        Re-rank candidates by penalizing those too similar to higher-ranked ones.
        This helps avoid redundant hypotheses and promotes diverse outputs.
        """
        if len(candidates) <= 1:
            return candidates, scores

        # Calculate pairwise similarity based on token overlap
        def token_overlap(s1, s2):
            tokens1 = set(s1.split())
            tokens2 = set(s2.split())
            if not tokens1 or not tokens2:
                return 0.0
            return len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))

        # Adjust scores based on diversity
        adjusted_scores = list(scores)
        for i in range(1, len(candidates)):
            max_overlap = 0
            for j in range(i):
                overlap = token_overlap(candidates[i], candidates[j])
                max_overlap = max(max_overlap, overlap)
            # Penalize high overlap
            adjusted_scores[i] -= diversity_weight * max_overlap * abs(adjusted_scores[i])

        # Sort by adjusted scores
        sorted_pairs = sorted(zip(candidates, adjusted_scores), key=lambda x: -x[1])
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

    def __call__(self, video_path):
        print(f"\n{'='*60}")
        print(f"Starting Lip Reading Inference")
        print(f"{'='*60}")

        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        lip_video = self.preprocess_video(video_path)
        cued_sequence, corrected_sentence = self.lip_decoding(lip_video)

        result = {
            "Processed_Cued_Speech_Sequence": cued_sequence,
            "LLM_Corrected_Sentence": corrected_sentence
        }

        print(f"\n{'='*60}")
        print(f"Inference Complete!")
        print(f"{'='*60}")
        print(f"Final Result:")
        print(f"  Sequence: {result['Processed_Cued_Speech_Sequence']}")
        print(f"{'='*60}\n")
        return result

    def load_video(self, video_path):
        """Load video file using OpenCV for better compatibility"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        video_data = np.array(frames)
        # Ensure we have enough frames (heuristic: at least 25fps)
        if len(video_data) < 10:
            print(f"  ⚠ Warning: Video too short ({len(video_data)} frames)")
        return video_data

@hydra.main(version_base="1.3", config_path="lip_agent_and_prompt_decoding_agent/configs", config_name="config_CCS_hand_infer")
def main(cfg):
    """Main function for testing"""
    video_path = cfg.file_path
    try:
        inference_pipeline = CuedAgentInference(cfg, detector="mediapipe", ctc_weight=0.5)
        if os.path.exists(video_path):
            result = inference_pipeline(video_path)
            output_path = "inference_result.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to: {output_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
   main()
