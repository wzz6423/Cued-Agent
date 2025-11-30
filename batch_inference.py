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
                  ctc_weight=0.1, video_ext='.mp4'):
    """
    Batch process video files

    Args:
        video_dir: Video folder path
        output_dir: Output folder path
        cfg: Configuration object
        detector: Detector type
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
            ctc_weight=ctc_weight
        )
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Process statistics
    total = len(video_files)
    success = 0
    failed = 0
    results = {}

    print(f"\nStarting batch processing...")
    pbar = tqdm(video_files)

    for video_file in pbar:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        pbar.set_description(f"Processing {video_name}")

        try:
            # Execute inference
            result = inference_pipeline(video_path)

            # Save individual result
            result_path = os.path.join(output_dir, f"{video_name}.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            results[video_name] = result
            success += 1

        except Exception as e:
            print(f"\nError processing {video_file}: {e}")
            failed += 1
            results[video_name] = {"error": str(e)}

    # Save summary report
    summary_path = os.path.join(output_dir, "batch_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "success": success,
        "failed": failed,
        "results": results
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nBatch processing complete!")
    print(f"Total: {total}, Success: {success}, Failed: {failed}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Cued Speech Batch Inference')
    parser.add_argument('--input', type=str, required=True,
                       help='Input video directory')
    parser.add_argument('--output', type=str, default='./batch_results',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='',
                       help='Pretrained model path')
    parser.add_argument('--detector', type=str, default='mediapipe',
                       choices=['mediapipe', 'retinaface'],
                       help='Face detector type')
    parser.add_argument('--ctc-weight', type=float, default=0.1,
                       help='CTC weight')
    parser.add_argument('--ext', type=str, default='.mp4',
                       help='Video file extension (default: .mp4)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return

    # Create configuration
    cfg = InferenceConfig()
    if args.model:
        cfg.pretrained_model_path = args.model

    if not cfg.pretrained_model_path or not os.path.exists(cfg.pretrained_model_path):
        print("Warning: Valid pretrained model path not set")
        print("Please set pretrained_model_path in config_example.py")

    process_batch(
        video_dir=args.input,
        output_dir=args.output,
        cfg=cfg,
        detector=args.detector,
        ctc_weight=args.ctc_weight,
        video_ext=args.ext
    )


if __name__ == '__main__':
    main()
