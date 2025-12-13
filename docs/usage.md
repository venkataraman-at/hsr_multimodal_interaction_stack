---
title: "Usage Guide"
nav_order: null
---

# Usage Guide

This page explains how to run and interact with the multimodal HSR interaction stack.

---

## 1. Starting the Interaction Pipeline

Run the full interaction script:

```
python3 scripts/llm_convo.py
```

This launches:
- Whisper ASR  
- Voice Activity Detection (VAD)  
- Tone/emotion classifier (CRNN)  
- LLM dialogue engine  
- ROS 2 gesture/behavior manager  

The system will begin listening for user speech and respond accordingly.

---

## 2. Supported Interaction Modes

### Speech Responses
The robot interprets user speech and generates:
- conversational replies  
- task responses  
- supportive / wellness dialogue  

### Tone-Aware Behavior
The CRNN model detects emotion categories such as:
- neutral  
- happy  
- sad  
- angry  
- stressed  

The detected emotional tone modifies the LLM’s response strategy.

### Gestures & Movements
Depending on LLM intent, the robot may perform:
- head nods  
- arm gestures  
- engagement poses  

### Wellness / Support Routines
The robot can initiate:
- breathing exercises  
- grounding prompts  
- positivity messages  

---

## 3. Stopping the Pipeline

Press:

```
CTRL + C
```

to safely terminate the interaction loop.

---

# Usage complete.
