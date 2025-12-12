---
title: "Abstract"
nav_order: 0
---

# Abstract
This work presents a multimodal interaction framework for the Toyota Human Support Robot (HSR) that integrates speech recognition, voice activity detection, vocal emotion estimation, and large language model (LLM)–based dialogue generation within a unified ROS 2 architecture. The system supports natural, continuous human–robot interaction by analyzing both the linguistic content and affective characteristics of a user’s speech in real time. Whisper-based ASR provides robust transcription, while a CRNN classifier processes Mel-spectrogram features to infer vocal tone. These signals condition an LLM dialogue module that adapts responses and selects context-appropriate interaction modes, including wellness prompts, storytelling, and general conversational behavior.

The framework is motivated by pediatric care scenarios in which emotional state, anxiety levels, and engagement vary significantly across interactions. Children undergoing Radiotherapy often experience stress, uncertainty, and periods of social or emotional withdrawal. By combining affective cues with adaptive dialogue and expressive robot behaviors, the system aims to provide supportive interaction that can help reduce anxiety, encourage engagement, and promote comfort during extended care routines.

The resulting architecture is modular, extensible, and suitable for future HRI research focused on affect-aware interaction, therapeutic support, and personalized robot behavior. It also provides a foundation for integrating additional perceptual or behavioral capabilities, including navigation, manipulation, and longitudinal user modeling.
