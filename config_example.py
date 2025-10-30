"""
Simplified configuration example file
For quick start of Cued Speech inference system
"""

import os

class InferenceConfig:
    """Inference configuration class"""

    def __init__(self):
        # ==================== Data Configuration ====================
        self.data = self.Data()

        # ==================== Model Configuration ====================
        self.model = self.Model()

        # ==================== Path Configuration ====================
        # Pretrained model path (must be set)
        self.pretrained_model_path = ""  # e.g., "/path/to/checkpoint.ckpt"

        # ==================== API Configuration ====================
        # These API Keys need to be set in corresponding module files
        # hand_recognition_agent/CustomizedPromptTemplate.py: OPENAI_API_KEY
        # self-p2w-agent/PostProcess_deepseek.py: DeepSeek API Key

        # ==================== Other Configuration ====================
        self.exp_dir = "./exp"
        self.exp_name = "cued_speech_inference"
        self.gpus = 1
        self.transfer_frontend = False
        self.transfer_encoder = False

    class Data:
        """Data related configuration"""
        def __init__(self):
            self.modality = "video"  # Modality type: "video" or "audio"
            self.input_size = (88, 88)  # Input video size
            self.max_frames = 150  # Maximum number of frames

    class Model:
        """Model related configuration"""
        def __init__(self):
            self.visual_backbone = self.VisualBackbone()
            self.encoder_type = "conformer"
            self.decoder_type = "transformer"

        class VisualBackbone:
            """Visual backbone network configuration"""
            def __init__(self):
                self.type = "resnet"
                self.num_layers = 18
                self.pretrained = True


# ==================== Usage Examples ====================

def example_usage():
    """Inference system usage example"""
    from Inference import CuedAgentInference

    # 1. Create configuration
    cfg = InferenceConfig()

    # 2. Set pretrained model path (required)
    cfg.pretrained_model_path = "path/to/your/model.ckpt"

    # 3. Initialize inference pipeline
    inference_pipeline = CuedAgentInference(
        cfg=cfg,
        detector="mediapipe",  # Choose detector: "mediapipe" or "retinaface"
        hand_weight=0.1,       # Hand prompt weight
        ctc_weight=0.1         # CTC weight
    )

    # 4. Execute inference
    video_path = "path/to/your/video.mp4"
    result = inference_pipeline(video_path)

    # 5. View results
    print("=" * 60)
    print("Inference Result:")
    print("-" * 60)
    print(f"Cued Speech Sequence: {result['Processed_Cued_Speech_Sequence']}")
    print(f"Pinyin Sequence: {result['Pinyin_Sequence']}")
    print(f"Chinese Sentence: {result['Mandarin_Sequence']}")
    print("=" * 60)

    return result


def batch_inference_example():
    """Batch inference example"""
    import json
    from Inference import CuedAgentInference

    # 1. Create configuration
    cfg = InferenceConfig()
    cfg.pretrained_model_path = "path/to/your/model.ckpt"

    # 2. Initialize inference pipeline
    inference_pipeline = CuedAgentInference(cfg=cfg)

    # 3. Batch processing
    video_dir = "path/to/video/directory"
    output_dir = "path/to/output/directory"
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    results = []
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        try:
            print(f"\nProcessing: {video_file}")
            result = inference_pipeline(video_path)

            # Save individual result
            output_file = os.path.join(output_dir,
                                      video_file.replace('.mp4', '.json'))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            results.append({
                'video': video_file,
                'result': result,
                'status': 'success'
            })
            print(f"✓ Success: {result['Mandarin_Sequence']}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'video': video_file,
                'error': str(e),
                'status': 'failed'
            })

    # Save summary results
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nBatch processing complete! Results saved in: {output_dir}")
    return results


def custom_config_example():
    """Custom configuration example"""
    from Inference import CuedAgentInference

    # Create custom configuration
    cfg = InferenceConfig()

    # Customize model path
    cfg.pretrained_model_path = "/path/to/your/checkpoint.ckpt"

    # Customize data configuration
    cfg.data.max_frames = 200  # Increase maximum frames

    # Adjust parameters during initialization
    inference_pipeline = CuedAgentInference(
        cfg=cfg,
        detector="retinaface",  # Use more accurate detector
        hand_weight=0.15,       # Increase hand weight
        ctc_weight=0.15         # Increase CTC weight
    )

    return inference_pipeline


if __name__ == '__main__':
    # Run single example
    print("="*60)
    print("Cued Speech Inference System Configuration Example")
    print("="*60)

    print("\nPlease follow these steps:")
    print("1. Set API Keys (OpenAI and DeepSeek)")
    print("2. Set pretrained model path")
    print("3. Run example_usage() function")

    print("\nExample code:")
    print("""
    from config_example import InferenceConfig, example_usage
    
    # Single video inference
    result = example_usage()
    
    # Batch inference
    from config_example import batch_inference_example
    results = batch_inference_example()
    """)
