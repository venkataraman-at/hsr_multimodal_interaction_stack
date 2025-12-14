---
title: "System Architecture"
nav_order: 3
---

# System Architecture

This page describes the multimodal interaction pipeline used for the Toyota Human Support Robot (HSR).  
The system integrates speech recognition, tone/emotion analysis, LLM-based dialogue reasoning, and ROS 2 behavior execution to create natural and adaptive human–robot interaction.

---

## Overview

The HSR interaction system processes user speech and emotional tone in parallel.  
It generates context-aware verbal responses and triggers appropriate robot behaviors such as gestures or wellness routines.

The pipeline operates in five main stages:

---

## 1. Audio Input & Detection

### **Voice Activity Detection (VAD)**
- Detects when the user begins and stops speaking  
- Prevents unnecessary processing  
- Segments audio into meaningful chunks  

---

## 2. Speech Understanding Path

### **Whisper ASR (Speech-to-Text)**
- Converts user speech into text  
- Handles noise, accents, and conversational inputs

### **Text Processing**
- Cleans, normalizes, and formats text for the LLM  

---

## 3. Emotion & Tone Analysis Path

This runs **simultaneously** with ASR.

### **Mel-Spectrogram Extraction**
- Converts audio into spectral features suitable for emotion classification

### **CRNN Emotion Classifier**
Predicts the emotional tone of the user, such as:
- neutral  
- happy  
- sad  
- stressed  
- angry  

The emotion label is sent to the dialogue manager to produce emotionally appropriate responses.

---

## 4. Dialogue Manager (LLM)

A GPT-based model receives:
- ASR text  
- detected emotion  
- conversation context  

The LLM generates:
- verbal responses  
- engagement prompts  
- wellness routines (when tone indicates stress)  
- storytelling or supportive behaviors  
- optional gesture/behavior commands  

---

## 5. Behavior Execution (ROS 2)

When the LLM requests physical action, the system interacts with ROS 2 nodes for:

### **Gestures**
- head nods  
- arm movements  
- expressive behaviors  

### **Wellness / Supportive Routines**
Triggered if:
- the user explicitly asks and when the robot detects mood fluctuations  
- the emotion model detects stress or sadness  

### ROS 2 Interfaces
- Trajectory controllers
- Head control nodes
  
---
## Summary

The system combines:
- ASR for understanding words  
- CRNN for understanding emotion  
- LLM for reasoning and generating supportive responses  
- ROS 2 for executing robot behaviors  

This allows the HSR to function as a conversational and emotionally aware companion capable of both verbal and physical interaction.
