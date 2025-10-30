"""
Cued Speech Inference System - Quick Start Script
Ensure API Keys and model paths are properly configured before use
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Inference import CuedAgentInference
from config_example import InferenceConfig


def main():
    parser = argparse.ArgumentParser(description='Cued Speech Inference System')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='./inference_result.json',
                       help='Output result path (default: ./inference_result.json)')
    parser.add_argument('--model', type=str, default='',
                       help='Pretrained model path')
    parser.add_argument('--detector', type=str, default='mediapipe',
                       choices=['mediapipe', 'retinaface'],
                       help='Face detector type (default: mediapipe)')
    parser.add_argument('--hand-weight', type=float, default=0.1,
                       help='Hand prompt weight (default: 0.1)')
    parser.add_argument('--ctc-weight', type=float, default=0.1,
                       help='CTC weight (default: 0.1)')
    parser.add_argument('--no-self-correction', action='store_true',
                       help='Skip self-correction step')

    args = parser.parse_args()

    # Check video file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    # Create configuration
    cfg = InferenceConfig()
    if args.model:
        cfg.pretrained_model_path = args.model

    # Check model path
    if not cfg.pretrained_model_path or not os.path.exists(cfg.pretrained_model_path):
        print("Warning: Valid pretrained model path not set")
        print("Please set pretrained_model_path in config_example.py")

    try:
        # Initialize inference pipeline
        print("\n" + "="*60)
        print("Initializing Cued Speech Inference System")
        print("="*60)

        inference_pipeline = CuedAgentInference(
            cfg=cfg,
            detector=args.detector,
            hand_weight=args.hand_weight,
            ctc_weight=args.ctc_weight
        )

        # Execute inference
        print(f"\nStarting video processing: {args.video}")
        result = inference_pipeline(args.video)

        # If skipping self-correction, remove that part
        if args.no_self_correction:
            result = {
                "Processed_Cued_Speech_Sequence": result.get('Processed_Cued_Speech_Sequence', ''),
                "Pinyin_Sequence": "Skipped",
                "Mandarin_Sequence": "Skipped",
                "Note": "Self-correction step skipped"
            }

        # Save result
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nResult saved to: {args.output}")

        # Display result summary
        print("\n" + "="*60)
        print("Inference Result Summary")
        print("="*60)
        print(f"Cued Speech Sequence: {result['Processed_Cued_Speech_Sequence']}")
        print(f"Pinyin Sequence: {result['Pinyin_Sequence']}")
        print(f"Chinese Sentence: {result['Mandarin_Sequence']}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError: Inference failed")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
