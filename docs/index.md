---
title: "Overview"
nav_order: 1
---

# HSR Multimodal Interaction Stack

The **HSR Multimodal Interaction Stack** is a speech-, tone-, and LLM-driven conversational interaction system developed for the **Toyota Human Support Robot (HSR)**.  
It enables natural, emotionally aware conversations by combining modern AI models with ROS 2 robot behavior controllers.

The system integrates:
- Whisper ASR  
- Voice Activity Detection (VAD)  
- CRNN-based tone/emotion analysis  
- GPT-based dialogue manager  
- ROS 2 gesture and behavior controllers  

Its goal is to allow the HSR to respond not only to *what* a user says but also to *how* they say it, enabling supportive and adaptive human–robot interaction.
---

## Pediatric Interaction Context

While the framework is applicable across general HRI scenarios, this project is particularly informed by pediatric clinical environments, where users may exhibit varied communication patterns due to stress, treatment routines, or limited social interaction. In such settings, a robot that can adapt to vocal tone and conversational cues can help maintain engagement and provide more comfortable interaction.

The multimodal stack supports these scenarios by:
- identifying shifts in vocal tone that may reflect changes in mood or comfort,
- adjusting dialogue style to match the user’s affect,
- offering supportive behaviors such as breathing prompts or grounding activities, and
- maintaining engagement through simple storytelling or interactive conversation.

By integrating emotion inference with adaptive dialogue and expressive behaviors, the system aims to make the HSR more responsive to the needs and communication styles of young users in treatment-oriented environments.

---

## Key Capabilities

### **Multimodal Speech Understanding**
- Robust speech recognition with Whisper  
- Real-time VAD to detect and segment speech   
- Multi-turn memory for extended conversation  

### **Emotion-Aware Dialogue**
The LLM adjusts responses based on:
- Detected user emotion  
- Stress markers  
- Context of previous dialogue  
- Trigger phrases for special modes  

### **Behavioral Modes**
The system supports several high-level interaction modes:

- **Wellness Mode**: breathing exercises, calming routines  
- **Story Mode**: interactive storytelling experiences  
- **Dialogue Mode**: general conversation  
- **Task Mode**: informational or task-oriented support  
- **Gesture Mode**: nods, head shakes, expressive poses via ROS 2  

Modes can be:
- explicitly requested by the user, or  
- implicitly triggered by emotional cues from the CRNN  

---

## System Pipeline Summary

1. **User speaks**  
2. **VAD detects speech**  
3. **Whisper converts speech to text**  
4. **CRNN analyzes tone/emotion**  
5. **LLM generates an emotion-aware response**  
6. **Robot executes gestures or wellness behaviors via ROS 2**  
7. **Response delivered to user (spoken or text)**  

This architecture supports **parallel processing**, **emotion conditioning**, and **intent-driven robot behavior**.

---

## Documentation Contents

Use the navigation sidebar or the links below:

- [System Architecture](architecture.md)  
- [Installation & Setup](setup.md)  
- [Usage Guide](usage.md)  
- [Features](features.md)  

---

## Purpose of the Project

This project aims to develop a multimodal interaction framework that allows the HSR to interpret and respond to both the linguistic and affective aspects of user speech. The system is designed to support scenarios where emotional awareness and conversational adaptability are important, including pediatric clinical environments.

Key objectives include:
- enabling the HSR to engage in emotionally adaptive dialogue rather than command-based interaction,  
- providing a modular interaction stack that future researchers can extend with new perceptual or behavioral capabilities,  
- demonstrating how tone and emotion cues can guide LLM-driven responses and robot behaviors, and  
- supporting user comfort and engagement through modes such as wellness prompts, grounding activities, and storytelling.

Overall, the system serves as a foundation for exploring affect-aware HRI and personalized conversational support on the Toyota HSR.
