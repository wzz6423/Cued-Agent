# Cued-Agent: A Multi-Agent Framework for Automatic Cued Speech Recognition

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.00391-b31b1b.svg)](https://arxiv.org/abs/2508.00391)

**The first multi-agent system for automatic Cued Speech recognition, integrating visual lip reading, hand cue recognition, and self-correction mechanisms.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Descriptions](#module-descriptions)
- [Code Status](#code-status)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

**Cued Speech** is a visual communication system that use hand cues (hand shapes and positions) to assist lip readings for hearing-impairs. This repository presents **Cued-Agent**, a novel multi-agent system that tackles the challenging problem of automatic Cued Speech recognition.


### The Challenge

Automatic Cued Speech Recognition (ACSR) is significantly more challenging than traditional lip reading because:
1. Hand cues are small, fast-moving, and often occluded
2. Precise temporal alignment between lip movements and hand cues is critical
3. Limited training data especially for hand cue recognition
4. Complex multi-modal fusion requirements

---

## 🏗️ Architecture

### System Overview

**[Framework Diagram Placeholder]**

```

```

**[Detailed Architecture Diagram Placeholder - To be added]**

### Multi-Agent Pipeline

Our framework consists of four specialized agents:

1. **Hand Recognition Agent**: Uses methods from STF-ACSR as training-free recognition agent to recognize hand shapes and positions
2. **Lip Reading Agent**: Employs a Conformer-based encoder for lip feature extraction,only finetuned on lip-reading task
3. **Joint Decoding Agent**: Fuses lip and hand information in the training-free manner for Cued Speech sequence decoding
4. **Self-Correction Agent**: Leverages LLM to post-process and correct recognition errors and outputs the final Cued Speech sequence and corresponding sentences

---



## 🔧 Installation

#

## 🚀 Quick Start

Will be added soon.

## 📦 Module Descriptions

### 1. Video Preprocessing (`lip_hand_seg_CS_latest.py`)

**Status: ✅ Complete**

Extracts lip and hand ROI regions from input videos:
- Face detection using MediaPipe or RetinaFace
- Lip region cropping and normalization
- Hand region tracking and extraction
- Output: Lip video tensor + Hand video frames

**Key Functions:**
- `extract_lip_roi()`: Extracts mouth region for lip reading
- `extract_hand_roi()`: Tracks and extracts hand regions
- `preprocess_video()`: Complete preprocessing pipeline

### 2. Hand Recognition Agent (`hand_recognition_agent/`)

**Status: ✅ Complete**

Uses GPT-4o for hand shape and position recognition:
- Few-shot learning with support set (40 examples for Mandarin Cued Speech System)
- Keyframe extraction based on slow-motion detection
- Prompt engineering for optimal recognition
- Output: Hand matrix [T, 44] with shape and position labels

**Key Components:**
- `CustomizedPromptTemplate.py`: Prompt templates for GPT-4o
- `supportprompt_shape.py`: Hand shape classification prompts
- `supportprompt_position.py`: Hand position classification prompts
- `support_set/`: 40 annotated images for few-shot learning

### 3. Lip and Hand Joint Decoding Agent (`lip_agent_and_prompt_decoding_agent/`)

**Status: ✅ Complete**

Multi-modal fusion and sequence decoding:
- Conformer encoder for visual speech features
- Transformer decoder with hand cue integration
- Beam search with CTC for sequence generation
- Cross-modal attention mechanisms

**Key Components:**
- `lightning_CCS_hand_prompt_decoding.py`: Main model architecture
- `train_lip_agent.py`: Training script for lip reading model
- `test_CCS_hand_free.py`: Evaluation script
- `espnet/`: Beam search and decoding utilities
- `datamodule/`: Data loading and preprocessing
- `configs/`: Model and training configurations

### 4. Self-Correction P2W Agent (`self-p2w-agent/`)

**Status: ✅ Complete**

LLM-based post-processing for error correction:
- DeepSeek integration for phoneme correction
- Linguistic constraint checking
- Minimal modification strategy
- Phoneme-to-word conversion

**Key Components:**
- `PostProcess_deepseek.py`: DeepSeek API integration
- `CuedseqSamples.py`: Sample generation for few-shot prompting

### 5. Inference Pipeline (`Inference.py`, `run_inference.py`)

**Status: ⚠️ In Progress - Being Refined**

Complete end-to-end inference pipeline integrating all agents.

**Current Status:**
- ✅ Basic pipeline structure implemented
- ✅ Agent integration completed
- ⚠️ Performance optimization in progress
- ⚠️ Error handling being enhanced
- ⚠️ Documentation being finalized

**Planned Improvements:**
- Enhanced keyframe extraction algorithms
- Better memory management for batch processing
- More robust error handling
- Additional output formats
- Real-time inference optimization

### 6. Utilities (`util/`)

**Status: ✅ Complete**

Helper functions and tools:
- `hand_decode.py`: Hand cue decoding utilities
- `mediapipe/`: MediaPipe-based video processing
- `detector.py`: Face and hand detection
- `video_process.py`: Video I/O and preprocessing

---

## 📊 Code Status

### ✅ Completed Modules

All core modules have been fully implemented and tested:

- **Video Preprocessing Module**: Robust ROI extraction for lips and hands
- **Hand Recognition Agent**: GPT-4V integration with few-shot learning
- **Lip Reading Agent**: Pre-trained Conformer models
- **Joint Decoding Agent**: Multi-modal fusion and beam search
- **Self-Correction Agent**: DeepSeek-based post-processing
- **Utility Functions**: Complete support libraries

### ⚠️ In Progress

**Inference Pipeline Refinement:**

The inference code (`Inference.py`, `run_inference.py`, `batch_inference.py`) is functional but undergoing refinement:

- **What Works:**
  - End-to-end video to text conversion
  - All four agents properly integrated
  - Basic error handling and logging
  - JSON output generation

- **Being Enhanced:**
  - Performance optimization for faster processing
  - Better memory management for long videos
  - More comprehensive error handling
  - Extended documentation and examples
  - Additional configuration options
  - Batch processing improvements

**Estimated Completion:** Within 1-2 weeks

### 🔜 Planned Features

- Web-based demo interface
- Real-time inference support
- Multi-language support (beyond Mandarin Chinese)
- Model compression for edge deployment
- Additional evaluation metrics and benchmarks

---

## 📈 Performance

### Benchmark Results




### Ablation Studies



Demonstrates the contribution of each agent to the overall system performance.

---

## 📖 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3746027.3755423,
author = {Huang, Guanjie and Tsang, Danny H.K. and Yang, Shan and Lei, Guangzhi and Liu, Li},
title = {Cued-Agent: A Collaborative Multi-Agent System for Automatic Cued Speech Recognition},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3755423},
doi = {10.1145/3746027.3755423},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {8313–8321},
numpages = {9},
keywords = {automatic cued speech recognition, multi-agent system, multimodal learning},
location = {Dublin, Ireland},
series = {MM '25}
}
```

### Related Publications

**[Additional citations to be added]**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We thank the following for their contributions to this project:



- **Auto-AVSR** for VSR model implementations

### Datasets

This work uses data from:
- [Dataset names and citations to be added]

### Funding

**[Funding information to be added]**

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please:

- Open an issue on GitHub
- Contact the maintainers: [ghuang565@connect.hkust-gz.edu.cn]

---


## 🗺️ Roadmap

- [x] Core module implementation
- [x] Hand recognition agent with GPT-4V
- [x] Lip reading agent training
- [x] Joint decoding agent
- [x] Self-correction agent
- [ ] Inference pipeline optimization (In Progress)
- [ ] Web demo interface
- [ ] Real-time inference
- [ ] Model compression
- [ ] Multi-language support
- [ ] Public dataset release

---

**Last Updated:** October 31, 2025

**Version:** 1.0.0-beta 

