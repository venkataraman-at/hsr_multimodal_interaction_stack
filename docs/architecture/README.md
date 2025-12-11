# System Architecture

This page describes the multimodal interaction architecture developed for the Toyota Human Support Robot (HSR). The system integrates speech recognition, voice activity detection, tone/emotion analysis, LLM-based dialogue reasoning, and ROS 2 gesture/behavior actuation to enable natural, adaptive, and emotionally aware human–robot interaction.

---

## High-Level Architecture

```mermaid
flowchart LR
    A[User Speech Input] --> B[Voice Activity Detection (VAD)]
    B --> C[Whisper ASR<br/>Speech-to-Text]
    C --> D[Text Preprocessing]
    A --> E[Audio Feature Extraction<br/>(Mel-Spectrogram)]
    E --> F[Tone / Emotion Analysis<br/>CRNN Model]
    D --> G[LLM Dialogue Manager<br/>(GPT-based)]
    F --> G
    G --> H{Response Type?}

    H -- Verbal --> I[HSR Verbal Response<br/>(Console Output / TTS)]
    H -- Gesture --> J[HSR Gesture Command<br/>(Arm/Head Movement)]
    H -- Behavioral --> K[HSR Routine<br/>(Wellness / Engagement)]

    J --> L[ROS 2 HSR Nodes<br/>(Trajectory, Head, LEDs)]
    K --> L
```
