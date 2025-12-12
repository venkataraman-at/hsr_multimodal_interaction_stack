---
title: "Demonstrations"
nav_order: 5
---

# Demonstrations

This page showcases example interactions with the **HSR Multimodal Interaction Stack**, including speech recognition, emotion inference, dialogue responses, and robot behaviors.

You may upload videos, GIFs, screenshots, or console transcripts here.  
Demo files can be placed inside `docs/media/` and referenced on this page.

---

## 1. Full Multimodal Interaction Demo

A complete demonstration of the system, including:
- Whisper ASR speech recognition  
- VAD-based segmentation  
- CRNN emotion detection  
- GPT dialogue generation  
- ROS 2 gesture execution  

**Demo will be uploaded soon.**

_A placeholder will be replaced with video or GIF soon._

---

## 2. Example Conversations (Text-Only Samples)

### Calm Interaction
**User:** _“Hello, how are you today?”_  
**HSR:** _“I’m here and happy to talk with you. How can I help today?”_

### Stress Detection & Wellness Trigger
**User (stressed tone):** _“I feel overwhelmed… I don’t know what to do.”_  
**HSR:**  
- _“I hear that you’re feeling overwhelmed. Let's try a short calming exercise.”_  
- *Robot performs gentle head nod and begins wellness mode.*

### Story Mode Trigger
**User:** _“Can you tell me a story?”_  
**HSR:** _“Of course. Let me share a short uplifting story…”_  
*(story continues)*

---

## 3. Emotion Detection Examples

You can upload screenshots here tomorrow.  
For now, placeholders:

**Input Audio:** _sad tone_  
**Emotion Model Output:** `sad (0.82 confidence)`

**Input Audio:** _frustrated tone_  
**Emotion Model Output:** `angry/frustrated (0.71 confidence)`

---

## 4. Robot Behavior Demos

These will show gestures or routines the robot performs:

- Head nods  
- Arm gestures  

**Demo files will be uploaded soon.**

---

## How to Add Your Videos Later

Upload your file (e.g., `demo1.mp4`) into:

```
docs/media/
```

Then link it in Markdown like this:

```
![Demo Video](media/demo1.mp4)
```

or for GIF:

```
![Interaction Demo](media/demo.gif)
```

or for image screenshots:

```
![Screenshot](media/screenshot1.png)
```

---

More demos can be added as the project expands.
