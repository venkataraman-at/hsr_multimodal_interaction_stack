---
title: "Installation & Setup"
nav_order: 3
---

# Installation & Setup

This page describes how to install and run the multimodal interaction stack for the Toyota Human Support Robot (HSR).

---

## 1. System Requirements

### Hardware
- HSR robot or Gazebo simulation environment
- Microphone input (HSR's onboard mic or external USB mic)

### Software
- Ubuntu 20.04 or 22.04
- ROS 2 Humble
- Python 3.8+
- OpenAI API key
- Required Python packages (Whisper, Torch, etc.)

---

## 2. Clone the Repository

```
git clone https://github.com/venkataraman-at/hsr_multimodal_interaction_stack.git
cd hsr_multimodal_interaction_stack
```

---

## 3. Install Python Dependencies

Recommended: create a virtual environment.

```
pip install -r requirements.txt
```

(Note: Add a `requirements.txt` later if needed.)

---

## 4. ROS 2 Setup

Source ROS:

```
source /opt/ros/humble/setup.bash
```

(Optional) Build your workspace:

```
cd ros2_ws
colcon build
source install/setup.bash
```

---

## 5. Running the Interaction Script

Run the full pipeline:

```
python3 scripts/llm_convo.py
```

This launches:
- Whisper ASR
- VAD
- Tone analysis
- LLM dialogue manager
- HSR gesture module (via ROS 2)

---

## 6. Connecting to the HSR

Ensure:
- Robot is powered ON
- Network connection is active
- ROS_DOMAIN_ID matches
- Services for head/arm control are available

```
ros2 topic list
ros2 action list
```

---

# Setup complete.
