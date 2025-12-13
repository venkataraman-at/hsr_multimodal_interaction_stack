# HSR Multimodal Interaction Stack

A ROS 2–based multimodal human–robot interaction framework for Toyota’s Human Support Robot (HSR), integrating speech recognition, emotion-aware dialogue, and expressive robot behaviors.

Developed at the **RIVeR Lab, Northeastern University**  
Platform: Toyota Human Support Robot (HSR)

This project implements a modular interaction stack that enables natural, emotionally aware communication between a human user and the HSR. The system combines real-time speech recognition, vocal emotion analysis, large language model–based dialogue generation, and ROS 2–controlled robot gestures.

The stack is designed for research and experimentation in socially assistive robotics, with an emphasis on safety, extensibility, and reproducibility.

## Key Features
- **Speech Recognition:** Real-time ASR using OpenAI Whisper  
- **Emotion Recognition:** CRNN-based vocal emotion analysis from Mel-spectrograms  
- **LLM Dialogue Manager:** Tone-aware, multi-turn conversational behavior  
- **Safety & Grounding:** Explicit constraints to prevent unsafe or medical advice  
- **Expressive Behaviors:** Head and arm gestures via ROS 2 controllers  
- **Modular Design:** Easily extensible to navigation or manipulation pipelines

## Current Scope and Limitations

- The system does **not** perform task-oriented manipulation or navigation in its current version  
- Task / Command Mode is limited to conversational and informational assistance  
- Physical task execution is intended as future work

## Quick Start (Gazebo)

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/venkataraman-at/hsr_multimodal_interaction_stack.git
cd ~/ros2_ws
colcon build
source install/setup.bash
ros2 launch hsrb_gazebo_launch hsrb_apartment_world.launch.py



