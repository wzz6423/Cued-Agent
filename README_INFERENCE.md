# Cued Speech Inference System User Guide

## System Overview

This inference system implements a complete Cued Speech recognition pipeline that converts input videos into phoneme sequences and corresponding Chinese sentences.

## System Architecture

The inference pipeline consists of four main stages:

```
Input Video 
    ↓
[1] Video Preprocessing (Extract Hand ROI + Lip ROI)
    ↓
[2] Hand Recognition Agent (GPT-4V recognizes hand shape and position)
    ↓
[3] Lip and Hand Joint Decoding Agent (Transformer + Beam Search)
    ↓
[4] Self-Correction P2W Agent (DeepSeek post-processing)
    ↓
Output: Cued Speech Sequence + Pinyin + Chinese Sentence
```

### Module Descriptions

#### 1. Video Preprocessing Module
- **Function**: Extracts lip and hand ROI regions from the input video
- **Tools Used**: MediaPipe or RetinaFace face detector
- **Output**: 
  - Lip video tensor (for lip reading)
  - Hand video frames (for hand recognition)
  - Hand position information (for keyframe extraction)

#### 2. Hand Recognition Agent
- **Function**: Uses GPT-4V to recognize hand shape and position in keyframes
- **Key Technologies**: 
  - Motion-based keyframe extraction
  - Few-shot learning (using support set)
  - Hand pose classification (5 positions + 8 shapes)
- **Output**: Hand matrix [T, 2] (position and shape labels for each frame)

#### 3. Lip and Hand Joint Decoding Agent
- **Function**: Fuses visual information and hand cues for sequence decoding
- **Model Architecture**: Conformer Encoder + Transformer Decoder
- **Decoding Strategy**: Beam Search with CTC
- **Output**: Preliminary Cued Speech phoneme sequence

#### 4. Self-Correction P2W Agent
- **Function**: Uses DeepSeek large language model to correct recognition errors
- **Correction Strategy**: 
  - Checks grammatical correctness of phoneme combinations
  - Converts to Chinese and checks semantic fluency
  - Makes minimal modifications based on common confusions
- **Output**: Final Cued Speech sequence, Pinyin sequence, and Chinese sentence

## Installation Dependencies

```bash
# PyTorch and vision libraries
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy

# MediaPipe (for face detection)
pip install mediapipe

# OpenAI API (for GPT-4V)
pip install openai

# Other dependencies
pip install pydantic
```

## Configuration Setup

### 1. Set API Keys

Set API Keys in the corresponding configuration files:

**hand_recognition_agent/CustomizedPromptTemplate.py**:
```python
OPENAI_API_KEY = 'your-openai-api-key'
```

**self-p2w-agent/PostProcess_deepseek.py**:
```python
client = OpenAI(api_key="your-deepseek-api-key", base_url="https://api.deepseek.com")
```

### 2. Prepare Pre-trained Model

Download the pre-trained weights for the lip reading model and set the path in the configuration:

```python
cfg.pretrained_model_path = "path/to/your/model.ckpt"
```

### 3. Prepare Video Preprocessing Module

Ensure the `preparation` module is properly installed, including:
- `preparation/detectors/mediapipe/` (MediaPipe detector)
- `preparation/detectors/retinaface/` (RetinaFace detector)

## Usage

### Method 1: Run Script Directly

```python
python Inference.py
```

Modify the `video_path` variable in the script to your video path.

### Method 2: Import as Module

```python
from Inference import CuedAgentInference

# Create configuration object
class SimpleConfig:
    class Data:
        modality = "video"
    
    class Model:
        class VisualBackbone:
            pass
        visual_backbone = VisualBackbone()
    
    data = Data()
    model = Model()
    pretrained_model_path = "path/to/model.ckpt"

# Initialize inference pipeline
cfg = SimpleConfig()
inference_pipeline = CuedAgentInference(
    cfg, 
    detector="mediapipe",  # or "retinaface"
    hand_weight=0.1,
    ctc_weight=0.1
)

# Run inference
result = inference_pipeline("path/to/video.mp4")

# View results
print("Cued Speech Sequence:", result['Processed_Cued_Speech_Sequence'])
print("Pinyin Sequence:", result['Pinyin_Sequence'])
print("Chinese Sentence:", result['Mandarin_Sequence'])
```

### Method 3: Batch Processing

```python
import os
import json
from Inference import CuedAgentInference

# Initialize
cfg = SimpleConfig()
inference_pipeline = CuedAgentInference(cfg)

# Batch processing
video_dir = "path/to/videos"
output_dir = "path/to/results"

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        
        try:
            result = inference_pipeline(video_path)
            
            # Save results
            output_file = os.path.join(output_dir, 
                                      video_file.replace('.mp4', '.json'))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Processing completed: {video_file}")
        except Exception as e:
            print(f"✗ Processing failed: {video_file}, Error: {e}")
```

## Parameter Description

### CuedAgentInference Initialization Parameters

- **cfg**: Configuration object containing model and data settings
- **detector**: Face detector type
  - `"mediapipe"`: Faster, suitable for real-time processing
  - `"retinaface"`: More accurate, suitable for offline processing
- **hand_weight**: Weight of hand cues (default: 0.1)
  - Higher values indicate more reliance on hand information
- **ctc_weight**: Weight of CTC loss (default: 0.1)
  - Used for Beam Search decoding

## Output Format

The inference result is in JSON format and contains the following fields:

```json
{
  "Processed_Cued_Speech_Sequence": "n i - j y ao - sh en - m e - m y eng - z i",
  "Pinyin_Sequence": "ni jiao shen me ming zi",
  "Mandarin_Sequence": "你叫什么名字",
  "Reasoning_Process": "Detailed reasoning process..."
}
```

## Important Notes

### 1. Video Requirements
- Format: Common formats like MP4, AVI
- Resolution: 720p or higher recommended
- Frame Rate: 25-30 fps
- Content: Clear facial and hand regions

### 2. Performance Optimization
- Using GPU significantly accelerates processing (auto-detected)
- Keyframe extraction reduces GPT-4V API calls
- Pay attention to memory management during batch processing

### 3. Cost Control
- GPT-4V API calls incur costs
- DeepSeek API calls incur costs
- Costs can be reduced by adjusting keyframe extraction strategy

### 4. Error Handling
- If hand recognition fails, the system continues with default values
- If self-correction fails, the uncorrected result is returned
- All steps have detailed logging output

## Troubleshooting

### Issue 1: Import Error
```
ImportError: No module named 'preparation'
```
**Solution**: Ensure the preparation module is in the correct location, or modify the import path

### Issue 2: API Call Failure
```
OpenAI API Error: Invalid API Key
```
**Solution**: Check if the API Key is set correctly

### Issue 3: Out of Memory
```
CUDA out of memory
```
**Solution**: 
- Reduce video resolution
- Use CPU mode
- Decrease batch size

### Issue 4: Video Loading Failure
```
FileNotFoundError: Video file not found
```
**Solution**: 
- Check if the video path is correct
- Ensure the video file format is supported
- Check file permissions

## Performance Metrics

Performance on standard test set:

- **CER (Character Error Rate)**: ~15%
- **WER (Word Error Rate)**: ~25%
- **Processing Speed**: ~2-5 seconds/video (depending on length and hardware)

## Change Log

### v1.0.0 (2025-01-12)
- Initial release
- Implemented complete four-stage inference pipeline
- Support for MediaPipe and RetinaFace detectors
- Integrated GPT-4V hand recognition
- Integrated DeepSeek self-correction

## Contact

For questions or suggestions, please contact the project maintainer.

## License

Please refer to the LICENSE file in the project root directory.

