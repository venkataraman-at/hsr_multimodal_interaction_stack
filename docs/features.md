---
title: "Features"
nav_order: 4
---

# Features

The HSR Multimodal Interaction Stack enables natural, adaptive, and emotionally aware communication between a human user and the Toyota Human Support Robot (HSR).  
This page provides a detailed overview of the system’s major capabilities.

---

## 1. Whisper Speech Recognition
The system uses OpenAI’s Whisper model for robust, real-time speech-to-text transcription.

**Capabilities:**
- Handles accents, noise, and conversational speech  
- Processes streaming microphone input  
- Produces cleaned, normalized text for language-model reasoning  
- Works fully offline if a local Whisper model is used  

---

## 2. Voice Activity Detection (VAD)
A lightweight VAD module continuously monitors audio input to determine when the user is speaking.

**Capabilities:**
- Reduces unnecessary computation  
- Prevents accidental triggers  
- Allows natural interruptions and turn-taking  
- Supports continuous listening without blocking  

---

## 3. CRNN-Based Tone & Emotion Recognition
A Convolutional Recurrent Neural Network (CRNN) analyzes Mel-spectrograms to determine the user’s vocal tone and emotional state.

**Detected categories:**
- Neutral  
- Happy / positive  
- Sad  
- Angry / frustrated  
- Stressed / tense  

**Uses in the system:**
- Triggers wellness routines  
- Modulates robot responses  
- Enhances empathetic conversational behavior  

---

## 4. LLM Dialogue Manager (GPT-Based)
A custom prompt architecture provides the HSR with dialogue abilities, intent understanding, and multi-turn memory.

**Capabilities:**
- Natural conversation  
- Task-related dialogue (“remind me”, “guide me”, “tell a story”)  
- Emotional alignment with user tone  
- Safety and grounding rules to avoid unsafe actions  
- Supports custom personas and modes  

---

## 5. Behavioral Modes
The system implements multiple high-level robot interaction modes.

### **Wellness Mode**
Triggered by:
- Explicit user request (“I feel stressed”, “help me relax”)  
- Detected negative emotion via CRNN  

Robot behavior includes:
- Calming dialogue  
- Breathing prompts  

---

### **Story Mode**
Triggered by:
- “Tell me a story”  
- Requests for entertainment or distraction  

Robot behavior includes:
- Story narration   

---

### **Task / Command Mode**
Supports simple procedural tasks like:
- Explaining steps  
- Providing reminders  
- Conversational assistance for daily routines  

---

## 6. ROS 2 Integration with HSR
The dialogue system connects to ROS 2 nodes to enable physical HSR behaviors.

**Capabilities:**
- Head movement (nodding, shaking)  
- Arm gestures  
- LED feedback  
- Extensible to navigation or manipulation pipelines  

The system is designed to be modular so new ROS behaviors can be plugged in easily.

---

## 7. Multimodal Logging & Debug Tools
The system logs:
- Recognized speech  
- Detected emotions  
- Model confidence  
- LLM prompts and responses  
- Triggered robot behaviors  

This ensures reproducibility and easy debugging during experiments.

---

## 8. Extensible Architecture
The system is designed to be expanded with additional modules:

- Face recognition  
- Gesture recognition  
- Safety monitoring  
- Navigation/Manipulation pipelines  
- Multi-user interaction  

The modular approach allows future students or lab members to build on the framework.

---

# Summary
The HSR Multimodal Interaction Stack combines speech, emotion inference, LLM reasoning, and ROS-based robot control into a unified, easy-to-extend platform for natural human-robot interaction.

