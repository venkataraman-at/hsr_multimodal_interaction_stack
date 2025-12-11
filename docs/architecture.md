---
title: "System Architecture"
nav_order: 2
---

# System Architecture

This page describes the multimodal interaction pipeline used for the Toyota HSR platform.  
The system integrates speech recognition, tone/emotion analysis, LLM-based dialogue reasoning, and ROS 2 behavior execution.

---

## Architecture Diagram

The architecture diagram will be added here once the image is prepared.

For now, this page introduces the main components of the pipeline.

---

## Components

### 1. Audio Processing
- Voice Activity Detection (VAD)
- Whisper ASR for speech-to-text
- Mel-Spectrogram extraction for tone analysis

### 2. Tone / Emotion Analysis
- CRNN classifier predicts affective state
- Works in parallel with ASR

### 3. Dialogue Manager
- GPT-based LLM with emotion and context inputs
- Produces meaningful and supportive responses

### 4. HSR Behavior Execution
- Gesture control (arm, head)
- Engagement routines (breathing exercises, positivity prompts)
- ROS 2 action clients and publishers

