# System Architecture

This page describes the multimodal interaction architecture developed for the Toyota Human Support Robot (HSR). The system integrates speech recognition, voice activity detection, tone/emotion analysis, LLM-based dialogue reasoning, and ROS 2 gesture/behavior actuation to enable natural, adaptive, and emotionally aware human–robot interaction.

## High-Level Architecture
```mermaid
flowchart LR
    B[Voice Activity Detection (VAD)] --> C[Whisper ASR (Speech-to-Text)]
    C --> D[Text Preprocessing]
    D --> G[LLM Dialogue Manager (GPT-based)]

    E[Audio Feature Extraction (Mel-Spectrogram)] --> F[Tone / Emotion Analysis (CRNN Model)]
    F --> G

    G --> H{Response Type?}

    H -- Verbal --> I[HSR Verbal Response (Console Output / TTS)]
    H -- Gesture --> J[HSR Gesture Command (Arm/Head Movement)]
    H -- Behavioral --> K[HSR Routine (Wellness / Engagement)]

    J --> L[ROS 2 HSR Nodes (Trajectory, Head, LEDs)]
    K --> L
```

