---
title: "Installation & Setup"
nav_order: 5
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

## 2. Create ROS 2 Workspace and Clone the Repository
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/venkataraman-at/hsr_multimodal_interaction_stack.git
cd hsr_multimodal_interaction_stack
```
---

## 3. Install Python Dependencies

Recommended: create a virtual environment.

```
pip install -r requirements.txt
```
---

## 4. ROS 2 Setup

Source ROS:

```
source /opt/ros/humble/setup.bash
```
(Optional) Build your workspace:

```
cd ~/ros2_ws
colcon build
source install/setup.bash
```
---

## 5. Running in Gazebo (Simulation)

Launch the system across **three terminals**.

### Terminal 1 — Launch Gazebo world

```bash
ros2 launch hsrb_gazebo_launch hsrb_apartment_world.launch.py
```

### Terminal 2 — Start the interaction node
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run hsr_hri llm_convo
```

### Terminal 3 — Start the conversation service
```bash
cd ~/ros2_ws
source install/setup.bash
ros2 service call /start_convo std_srvs/srv/SetBool "{data: true}"
```

### Verify memory state (optional)

```bash
cat ~/.hsr_gpt_hri/memory.json | jq .
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

### Determining and Setting `ROS_DOMAIN_ID`

`ROS_DOMAIN_ID` is used by ROS 2 DDS to isolate communication between different robots, simulations, or users on the same network.  
Your laptop and the HSR **must use the same `ROS_DOMAIN_ID`** to communicate.

#### 1. Check the current `ROS_DOMAIN_ID` on your machine

```bash
echo $ROS_DOMAIN_ID
```

- If a number is returned (e.g., `30`), that is your current domain ID  
- If nothing is returned, the default value `0` is being used  

#### 2. Check the robot’s `ROS_DOMAIN_ID`
On the HSR (or via SSH into the robot), run:

```bash
echo $ROS_DOMAIN_ID
```

The value on the robot **must match** the value on your workstation.

> In many lab setups, the robot’s `ROS_DOMAIN_ID` is predefined by system configuration or lab policy.

#### 3. Set `ROS_DOMAIN_ID` (if needed)
If the values do not match, set it manually on your machine:

```bash
export ROS_DOMAIN_ID=<robot_domain_id>
```

Example:

```bash
export ROS_DOMAIN_ID=30
```

To make this persistent across terminals:
```bash
echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
source ~/.bashrc
```

#### 4. Verify ROS 2 communication
After setting the domain ID, confirm connectivity:

```bash
ros2 topic list
ros2 service list
ros2 action list
```

If topics and services from the HSR are visible, the domain configuration is correct.

## 7. Supported Interaction Modes

### Speech Responses
The robot interprets user speech and generates:
- conversational replies  
- task responses  
- supportive / wellness dialogue  

### Tone-Aware Behavior
The CRNN model detects emotion categories such as:
- neutral/calm  
- happy  
- sad  
- angry  
- stressed
- pleasant_surprise
- disgust 

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

## 8. Stopping the Pipeline

Press:

```
CTRL + C
```

to safely terminate the interaction loop.

---

# Setup complete.
