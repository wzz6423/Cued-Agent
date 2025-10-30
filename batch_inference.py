"""
Cued Speech Inference System - Batch Processing Script
Support batch processing of multiple video files
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Inference import CuedAgentInference
from config_example import InferenceConfig


def process_batch(video_dir, output_dir, cfg, detector='mediapipe',
                  hand_weight=0.1, ctc_weight=0.1, video_ext='.mp4'):
    """
    Batch process video files

    Args:
        video_dir: Video folder path
        output_dir: Output folder path
        cfg: Configuration object
        detector: Detector type
        hand_weight: Hand weight
        ctc_weight: CTC weight
        video_ext: Video file extension
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all video files
    video_files = [f for f in os.listdir(video_dir)
                   if f.endswith(video_ext)]

    if not video_files:
        print(f"Error: No {video_ext} files found in {video_dir}")
        return

    print(f"\nFound {len(video_files)} video files")

    # Initialize inference pipeline
    print("\nInitializing inference system...")
    try:
        inference_pipeline = CuedAgentInference(
            cfg=cfg,
            detector=detector,
            hand_weight=hand_weight,
            ctc_weight=ctc_weight
        )
    except Exception as e:
        print(f"Error: Initialization failed - {e}")
        return

    # Processing result statistics
    results = []
    success_count = 0
    failed_count = 0

    # Batch processing
    print(f"\nStarting batch processing...")
    print("="*60)

    for video_file in tqdm(video_files, desc="Processing progress"):
        video_path = os.path.join(video_dir, video_file)
        output_file = os.path.join(output_dir,
                                   video_file.replace(video_ext, '.json'))

        try:
            # Execute inference
            result = inference_pipeline(video_path)

            # Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Record success
            results.append({
                'video': video_file,
                'status': 'success',
                'cued_sequence': result['Processed_Cued_Speech_Sequence'],
                'pinyin': result['Pinyin_Sequence'],
                'mandarin': result['Mandarin_Sequence'],
                'output_file': output_file
            })
            success_count += 1

            print(f"✓ {video_file}: {result['Mandarin_Sequence']}")

        except Exception as e:
            # Record failure
            results.append({
                'video': video_file,
                'status': 'failed',
                'error': str(e)
            })
            failed_count += 1

            print(f"✗ {video_file}: {str(e)}")

    # Save summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(video_files),
        'success': success_count,
        'failed': failed_count,
        'success_rate': f"{success_count/len(video_files)*100:.2f}%",
        'config': {
            'detector': detector,
            'hand_weight': hand_weight,
            'ctc_weight': ctc_weight,
            'model_path': cfg.pretrained_model_path
        },
        'results': results
    }

    summary_file = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print statistics
    print("\n" + "="*60)
    print("Batch Processing Complete!")
    print("="*60)
    print(f"Total videos: {len(video_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_count/len(video_files)*100:.2f}%")
    print(f"\nResults saved in: {output_dir}")
    print(f"Summary report: {summary_file}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Cued Speech Batch Inference')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Video folder path')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output folder path')
    parser.add_argument('--model', type=str, default='',
                       help='Pretrained model path')
    parser.add_argument('--detector', type=str, default='mediapipe',
                       choices=['mediapipe', 'retinaface'],
                       help='Face detector type (default: mediapipe)')
    parser.add_argument('--hand-weight', type=float, default=0.1,
                       help='Hand prompt weight (default: 0.1)')
    parser.add_argument('--ctc-weight', type=float, default=0.1,
                       help='CTC weight (default: 0.1)')
    parser.add_argument('--video-ext', type=str, default='.mp4',
                       help='Video file extension (default: .mp4)')

    args = parser.parse_args()

    # Check video directory
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return

    # Create configuration
    cfg = InferenceConfig()
    if args.model:
        cfg.pretrained_model_path = args.model

    # Check model path
    if not cfg.pretrained_model_path or not os.path.exists(cfg.pretrained_model_path):
        print("Warning: Valid pretrained model path not set")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return

    # Execute batch processing
    try:
        process_batch(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            cfg=cfg,
            detector=args.detector,
            hand_weight=args.hand_weight,
            ctc_weight=args.ctc_weight,
            video_ext=args.video_ext
        )
    except Exception as e:
        print(f"\nError: Batch processing failed")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

