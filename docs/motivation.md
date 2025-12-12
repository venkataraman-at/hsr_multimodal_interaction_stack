---
title: "Research Motivation"
nav_order: 2
---

# Research Motivation

The Toyota HSR was originally designed as an assistive platform for home and clinical environments, where interaction quality and emotional comfort can be just as important as task execution. This project explores how multimodal perception and adaptive dialogue can improve the HSR’s ability to support users who may be experiencing stress, uncertainty, or emotional discomfort—particularly in pediatric settings.

Children undergoing medical treatment often face fluctuating anxiety, limited autonomy, and long periods of isolation. Their communication patterns can vary widely depending on mood, energy, fear, or discomfort. A robot that can only process literal speech is unable to respond in a way that acknowledges these emotional cues. This motivates the need for an interaction system that considers *how* something is said, not just *what* is said.

This work investigates how real-time speech transcription, vocal tone estimation, and LLM-driven dialogue can be combined to create an emotionally aware interaction loop. By detecting stress or negative affect through a CRNN model and conditioning LLM responses with this information, the system can shift into more supportive modes such as breathing prompts, grounding activities, or storytelling. These behaviors are delivered through both language and expressive robot motions, aiming to create interactions that feel more comforting and engaging for young users.

Beyond the pediatric context, the framework contributes a modular pipeline for affect-aware HRI that can be adapted for other assistive or therapeutic scenarios. The architecture is intentionally designed to support future extensions, including personalized user profiles, longitudinal emotional tracking, and integration with navigation or manipulation behaviors.

In summary, the motivation for this system stems from the need to move beyond command-based interfaces toward emotionally responsive robot behavior—especially in settings where user wellbeing and emotional safety are central to the interaction.
