#!/usr/bin/env python3
import os, sys, re, threading, json, queue, time, random, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from collections import deque
import whisper
import soundfile as sf
ASR_ENABLED = True

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile

from std_srvs.srv import SetBool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from scipy.signal import resample_poly

import sounddevice as sd

# --- LLM client (simple requests to OpenAI) ---
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"

# ===== Chat configuration =====
OPEN_DOMAIN_MODE = True
REPLY_MAX_TOKENS = 220
TEMPERATURE      = 0.8
TOP_P            = 0.95

# rolling memory / token control
MAX_TURNS_BEFORE_SUMMARY = 16
SUMMARY_MAX_TOKENS       = 160
HISTORY_HARD_CAP         = 60

# ===== VAD / noise gate config =====
VAD_AGGRESSIVENESS      = 2
VAD_FRAME_MS            = 30
VAD_MIN_ACTIVE_MS       = 200
VAD_MAX_SILENCE_MS      = 600
ENERGY_FLOOR_WINDOW_S   = 5.0
ENERGY_MIN_RMS          = 30
ENERGY_MARGIN_MULT      = 2.0

# wake-word behavior
DEFAULT_REQUIRE_WAKE = False
WAKE_WORDS = ("hello", "hsr", "robot", "hey robot")

# polite guardrails
SOFT_GUARDRAILS = (
    "Do not give medical, legal, or financial advice. "
    "If asked, steer to general info and suggest consulting a professional. "
    "Do not provide dangerous instructions. "
)

PERSONA_OPEN_DOMAIN = (
    "You are HSR, a warm, playful assistant for kids and families. "
    "Keep replies natural, 2–4 short sentences. Use contractions. "
    "Vary rhythm (some short, some longer). Use one short backchannel occasionally (≤1). "
    "Match the user's mood (calm/sad/excited/stressed) with tone, length, and pacing. "
    "Ask exactly one contextual follow-up only if helpful. No emojis. "
    + SOFT_GUARDRAILS
)

PERSONA_LAB_BRIEF = (
    "You are HSR, a lab assistant robot. Be concise (<= 2 sentences). "
    "Be friendly and professional. " + SOFT_GUARDRAILS
)

# === Persistent memory (disk) ===
MEM_DIR   = Path(os.getenv("HSR_MEM_DIR", str(Path.home()/".hsr_gpt_hri")))
MEM_FILE  = MEM_DIR / "memory.json"
LOG_DIR   = MEM_DIR / "logs"
MEM_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_write_json(path: Path, obj: dict):
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        try: os.remove(tmp_path)
        except FileNotFoundError: pass

def _load_memory_from_disk() -> dict:
    if MEM_FILE.exists():
        try:
            with open(MEM_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"summary_memory":"", "profile":{"name":"","likes":[],"dislikes":[],"last_mood":"calm"}, "last_updated":None}

def _save_memory_to_disk(summary_memory: str, profile: dict):
    data = {
        "summary_memory": summary_memory or "",
        "profile": profile or {},
        "last_updated": datetime.utcnow().isoformat()+"Z"
    }
    _atomic_write_json(MEM_FILE, data)

def _build_system_prompt():
    return PERSONA_OPEN_DOMAIN if OPEN_DOMAIN_MODE else PERSONA_LAB_BRIEF

def _headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def _chat(payload: dict) -> str:
    r = requests.post(OPENAI_URL, headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def chatgpt_reply(history, max_tokens=REPLY_MAX_TOKENS, temperature=TEMPERATURE):
    data = {
        "model": OPENAI_MODEL,
        "messages": history,
        "temperature": float(temperature),
        "top_p": TOP_P,
        "max_tokens": int(max_tokens),
    }
    return _chat(data)

def stream_chat(history, temperature=TEMPERATURE, top_p=TOP_P):
    data = {
        "model": OPENAI_MODEL,
        "messages": history,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": True
    }
    with requests.post(OPENAI_URL, headers=_headers(), json=data, timeout=120, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]": break
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

def summarize_history_for_memory(summary: str, recent_msgs: list) -> str:
    sys_msg = {"role":"system","content":
               "Summarize the user's preferences, facts about them, and ongoing topics in <= 6 bullet points. "
               "Keep it neutral, short, and useful for future replies."}
    msgs = [sys_msg]
    if summary:
        msgs.append({"role":"system", "content": f"Previous summary:\n{summary}"})
    for m in recent_msgs[-12:]:
        msgs.append(m)
    try:
        return chatgpt_reply(msgs, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.3)
    except Exception:
        return summary or ""

# --------------------- Random Story Library ---------------------
STORIES = [
    "Once upon a time, in a magical forest, there was a curious squirrel named Nutty. Nutty loved to explore and discover new things. One day, Nutty found a glowing acorn and decided to follow it. The acorn led Nutty to a hidden treasure chest filled with golden nuts. Nutty shared the treasure with all the forest animals and became the hero of the forest. The end.",
    "There was once a young dragon named Sparky who was afraid of flying. One day, Sparky saw a bird soaring in the sky and decided to give it a try. With a little courage, Sparky took off into the sky, flapping his wings and soaring higher than ever before. From that day on, Sparky became the fastest dragon in the kingdom, always flying to new adventures. The end.",
    "A little robot named Zippy lived in a futuristic city. Zippy loved to help people, whether it was fixing broken machines or delivering packages. One day, Zippy helped a young girl find her lost puppy, and in return, the girl gave Zippy a shiny new upgrade. Zippy became the most helpful robot in the city, and everyone loved Zippy. The end."
]
_last_story = None
def tell_story():
    """Return a random story that is not the same as the last one."""
    global _last_story
    candidates = [s for s in STORIES if s != _last_story]
    story = random.choice(candidates) if candidates else random.choice(STORIES)
    _last_story = story
    return story

# --------------------- Automatic Mic Selection ---------------------
def _auto_select_mic(kind='input'):
    import sounddevice as sd
    for idx, dev in enumerate(sd.query_devices()):
        if kind == 'input' and dev.get('max_input_channels', 0) > 0:
            print(f"[Mic] Auto-selected device: {dev['name']} (index {idx})")
            return idx
    print("[Mic] No input device found!")
    return None

# --------------------- Continue with your original EmotiveTTS, ToneAnalyzer, HsrGestures ---------------------
# Only changes:
# - start_mic() uses _auto_select_mic() instead of "echomic"
# - handle_user_utterance() triggers tell_story() when "story" detected
# - _last_story prevents repetition
# All other code remains identical.


# ========== Emotive TTS ==========
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
except Exception:
    _tts_engine = None

EMOTE_PROFILES = {
    "calm":    {"rate": 170, "volume": 0.70, "voice_hint": ["en-us", "en"], "interj": ["Okay.", "Got it."]},
    "happy":   {"rate": 185, "volume": 0.78, "voice_hint": ["en-us+m3","en+m3","en"], "interj": ["Nice!", "Awesome!", "Sweet!"]},
    "sad":     {"rate": 160, "volume": 0.66, "voice_hint": ["en-us+f3","en+f3","en"], "interj": ["Hmm…", "I hear you."]},
    "stressed":{"rate": 165, "volume": 0.68, "voice_hint": ["en-us","en"], "interj": ["Understood.", "Okay."]},
}
EMOTE_RATE_JITTER = 8
EMOTE_SENTENCE_PAUSE = (0.08, 0.18)

def _contractify(text: str) -> str:
    # tiny contraction pass
    repl = [
        (r"\bI am\b", "I'm"), (r"\byou are\b", "you're"), (r"\bwe are\b", "we're"),
        (r"\bthat is\b", "that's"), (r"\bit is\b", "it's"), (r"\bdo not\b", "don't"),
        (r"\bdoes not\b", "doesn't"), (r"\bcan not\b", "cannot"), (r"\bcan not\b", "can't"),
    ]
    s = text
    for pat, rep in repl:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

class EmotiveTTS:
    def __init__(self, engine):
        self.engine = engine
        try:
            self._voices = engine.getProperty('voices') if engine else []
        except Exception:
            self._voices = []
        try:
            if engine:
                engine.setProperty('volume', 0.7)
                engine.setProperty('rate', 175)
        except Exception:
            pass
        self._speak_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._is_speaking = False

    def stop(self):
        # barge-in stop
        if self.engine is None: return
        with self._speak_lock:
            self._stop_flag.set()
            try: self.engine.stop()
            except Exception: pass
            self._is_speaking = False

    def _pick_voice(self, hints):
        for h in hints or []:
            for v in self._voices:
                vn = (getattr(v, 'name', '') + ' ' + getattr(v, 'id', '')).lower()
                if h.lower() in vn:
                    return v.id
        return None

    def speak_chunks_streaming(self, chunk_iterable, mood: str = "calm", gesture_cb=None, bargein_flag_getter=None):
        """Speak chunks progressively; stop if barge-in flag triggers."""
        if self.engine is None:
            # print fallback
            for ch in chunk_iterable:
                sys.stdout.write(ch); sys.stdout.flush()
            print()
            return

        prof = EMOTE_PROFILES.get(mood, EMOTE_PROFILES['calm'])
        try:
            vid = self._pick_voice(prof.get('voice_hint', []))
            if vid:
                self.engine.setProperty('voice', vid)
        except Exception:
            pass

        def _worker():
            with self._speak_lock:
                self._stop_flag.clear()
                self._is_speaking = True
            try:
                # optional interjection
                interj = random.choice(prof.get('interj', [])) if prof.get('interj') else ''
                if interj:
                    self.engine.setProperty('rate', int(prof['rate']))
                    self.engine.say(interj)

                buf = ""
                last_gesture = 0.0
                last_time = time.time()

                for raw in chunk_iterable:
                    if self._stop_flag.is_set(): break
                    buf += raw
                    # speak on sentence/pause boundary or min len
                    speak_now = False
                    if re.search(r"[.!?]\s$", buf) or len(buf) >= 80:
                        speak_now = True
                    # barge-in?
                    if bargein_flag_getter and bargein_flag_getter():
                        break
                    if speak_now:
                        text = _contractify(buf.strip())
                        if not text:
                            continue
                        # mood prosody
                        try:
                            base = int(prof['rate'])
                            jitter = random.randint(-EMOTE_RATE_JITTER, EMOTE_RATE_JITTER)
                            self.engine.setProperty('rate', max(140, base + jitter))
                            self.engine.setProperty('volume', float(prof['volume']))
                        except Exception:
                            pass
                        # mid-utterance gesture timing on sentiment keywords
                        if gesture_cb:
                            now = time.time()
                            if now - last_gesture > 0.6:
                                if re.search(r"\b(yes|great|awesome|perfect|love)\b", text.lower()):
                                    gesture_cb("NOD")
                                    last_gesture = now
                                elif re.search(r"\b(no|not really|hmm|uh|sad)\b", text.lower()):
                                    gesture_cb("SHAKE")
                                    last_gesture = now
                        self.engine.say(text + " ")
                        self.engine.runAndWait()
                        buf = ""
                        # micro-pause
                        time.sleep(random.uniform(*EMOTE_SENTENCE_PAUSE))
                # flush any remainder
                if not self._stop_flag.is_set():
                    text = _contractify(buf.strip())
                    if text:
                        self.engine.say(text)
                        self.engine.runAndWait()
            finally:
                with self._speak_lock:
                    self._is_speaking = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=120)
        with self._speak_lock:
            self._is_speaking = False

    def speak_once(self, text: str, mood: str = "calm"):
        """Non-streaming one-shot (used for small system lines)."""
        if self.engine is None:
            print(f"[TTS disabled] ({mood}) {text}")
            return
        prof = EMOTE_PROFILES.get(mood, EMOTE_PROFILES['calm'])
        try:
            vid = self._pick_voice(prof.get('voice_hint', []))
            if vid: self.engine.setProperty('voice', vid)
            self.engine.setProperty('rate', int(prof['rate']))
            self.engine.setProperty('volume', float(prof['volume']))
        except Exception:
            pass
        with self._speak_lock:
            self._stop_flag.clear()
            self._is_speaking = True
        try:
            self.engine.say(_contractify(text))
            self.engine.runAndWait()
        finally:
            with self._speak_lock:
                self._is_speaking = False

_emotive = EmotiveTTS(_tts_engine)

def _tone_to_mood(tone: dict) -> str:
    lbl = (tone or {}).get('label', 'calm')
    if lbl in ('excited',):        return 'happy'
    if lbl in ('sad/tired',):      return 'sad'
    if lbl in ('stressed/angry',): return 'stressed'
    return 'calm'

# Optional offline ASR (Vosk) + VAD
# _ASR_AVAILABLE = True
# try:
#     import sounddevice as sd
#     from vosk import Model, KaldiRecognizer
# except Exception:
#     _ASR_AVAILABLE = False

import whisper
import sounddevice as sd
_ASR_AVAILABLE = True

try:
    import webrtcvad
    _VAD_AVAILABLE = True
except Exception:
    _VAD_AVAILABLE = False

def _find_device(name_contains: str, kind='input'):
    try:
        for idx, dev in enumerate(sd.query_devices()):
            name = str(dev.get('name', ''))
            if name_contains.lower() in name.lower():
                if kind == 'input' and dev.get('max_input_channels', 0) > 0:
                    return idx
                if kind == 'output' and dev.get('max_output_channels', 0) > 0:
                    return idx
    except Exception:
        pass
    return None

# ====== CRNN EMOTION MODEL (replacement for old ToneAnalyzer) ======
import torch
import torchaudio
from hsr_hri.model_crnn import CRNN 

EMOTION_LABELS = [
    "angry","disgust","fearful","happy","sad",
    "neutral","surprised","calm","pleasant_surprise"
]

TARGET_SR = 16000

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=1024,
    hop_length=256,
    n_mels=64
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

# Load the trained model
def load_emotion_model():
    model = CRNN(num_classes=len(EMOTION_LABELS))
    from importlib.resources import files

    def load_emotion_model():
        model = CRNN(num_classes=len(EMOTION_LABELS))

        # resolve installed location safely
        model_path = files("hsr_hri").joinpath("emotion_crnn.pth")

        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()
        return model

    model.eval()
    return model

EMOTION_MODEL = load_emotion_model()

def predict_emotion(raw_audio: np.ndarray, sr: int):
    wav = torch.tensor(raw_audio).float().unsqueeze(0)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.abs().max() > 0:
        wav = wav / wav.abs().max()
    mel = mel_transform(wav)
    mel = amp_to_db(mel)
    mel = mel.unsqueeze(0)
    with torch.no_grad():
        logits = EMOTION_MODEL(mel)
        pred = torch.argmax(logits, dim=1).item()
    return EMOTION_LABELS[pred]
    
# Initialize VAD (optional)
try:
    import webrtcvad
    _VAD_AVAILABLE = True
except Exception:
    _VAD_AVAILABLE = False

class HsrGestures(Node):
    HEAD_JOINTS = ['head_pan_joint', 'head_tilt_joint']
    ARM_JOINTS  = ['arm_lift_joint','arm_flex_joint','arm_roll_joint','wrist_flex_joint','wrist_roll_joint']

    def __init__(self):
        super().__init__("llm_convo")
        self._last_tone = {"label": "calm", "f0": 0.0, "wpm": 0.0, "rms": 0.0}
        qos = QoSProfile(depth=10)

        # Action clients
        self.head_client = ActionClient(self, FollowJointTrajectory, '/head_trajectory_controller/follow_joint_trajectory')
        self.arm_client  = ActionClient(self, FollowJointTrajectory, '/arm_trajectory_controller/follow_joint_trajectory')

        # Base publisher
        self.base_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # On/off service
        self.srv = self.create_service(SetBool, 'start_convo', self.handle_start)

        self.convo_on = False

        # Conversation state
        self.require_wake = DEFAULT_REQUIRE_WAKE
        self._wake_words = WAKE_WORDS
        self._last_user_at = 0.0
        self._idle_prompt_after_s = 45.0

        # History + memory
        self.llm_history = [{"role":"system","content":_build_system_prompt()}]
        self.turn_count = 0

        persisted = _load_memory_from_disk()
        self.summary_memory = persisted.get("summary_memory", "")
        self.profile = persisted.get("profile", {"name":"","likes":[],"dislikes":[],"last_mood":"calm"})

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._transcript_path = LOG_DIR / f"session-{ts}.jsonl"
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t":"meta","event":"session_start","ts":ts})+"\n")
        except Exception:
            pass

        # Echo/VAD state
        self._last_hello_ts = 0.0
        self._asr_ignore_until = 0.0
        self._asr_streaming = False
        self._last_spoken_text = ""
        self._user_bargein_flag = False  # flips true if user starts talking while TTS is speaking

        self.get_logger().info("HSR GPT HRI node ready. Call: ros2 service call /start_convo std_srvs/srv/SetBool \"{data: true}\"")

        self._wait_for_actions()

        # # ASR init
        # self._asr_q = queue.Queue()
        # self._asr_model = None
        # self._asr_rec = None
        # if _ASR_AVAILABLE:
        #     try:
        #         asr_path = os.path.expanduser("~/models/vosk-en")
        #         if os.path.isdir(asr_path):
        #             self._asr_model = Model(asr_path)
        #             self._asr_rec = KaldiRecognizer(self._asr_model, 16000)
        #             try:
        #                 self._asr_rec.SetWords(True)
        #             except Exception:
        #                 pass
        #             self.get_logger().info("Vosk ASR model loaded.")
        #         else:
        #             self.get_logger().warn(f"Vosk model not found at {asr_path}. Mic will be disabled.")
        #     except Exception as e:
        #         self.get_logger().warn(f"Vosk init failed: {e}")
        # else:
        #     self.get_logger().info("Vosk/sounddevice not available; running text-only.")

        # ASR init (Whisper)
        self._asr_q = queue.Queue()
        try:
            self.whisper_model = whisper.load_model("base", device="cpu")
            self.get_logger().info("Whisper ASR model loaded (base).")
        except Exception as e:
            self.get_logger().warning(f"Whisper init failed: {e}")
            self.whisper_model = None


        # VAD state
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if _VAD_AVAILABLE else None
        self._gate_open = False
        self._active_ms = 0
        self._silence_ms = 0
        self._energy_hist = deque(maxlen=int(ENERGY_FLOOR_WINDOW_S * 1000 / VAD_FRAME_MS))
        self._rms_floor = ENERGY_MIN_RMS

        self.create_timer(0.1, self._poll_asr)
        self.create_timer(2.0, self._idle_nudge_timer)

    def _wait_for_actions(self):
        self.get_logger().info("Waiting for head/arm trajectory servers...")
        self.head_client.wait_for_server()
        self.arm_client.wait_for_server()
        self.get_logger().info("Action servers available.")

    # ====== Trajectory helpers ======
    def _send_head_traj(self, positions, durations):
        traj = JointTrajectory()
        traj.joint_names = self.HEAD_JOINTS
        for pos, t in zip(positions, durations):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.head_client.send_goal_async(goal)

    def _send_arm_traj(self, points):
        traj = JointTrajectory()
        traj.joint_names = self.ARM_JOINTS
        for pos, t in points:
            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.arm_client.send_goal_async(goal)

    # ====== Gestures ======
    def gesture_nod(self):
        self._send_head_traj(positions=[[0.0, 0.20],[0.0,-0.10],[0.0, 0.00]], durations=[0.7, 1.4, 2.0])

    def gesture_shake(self):
        self._send_head_traj(positions=[[ 0.35, 0.0],[-0.35, 0.0],[ 0.00, 0.0]], durations=[0.7, 1.4, 2.0])

    def gesture_wave(self):
        pts = [([0.06, -0.5, 0.0, -0.3,  0.8], 0.8),([0.08, -0.5, 0.0, -0.3, -0.8], 1.3),([0.08, -0.5, 0.0, -0.3,  0.8], 1.8),([0.00,  0.0, 0.0,  0.0,  0.0], 2.6)]
        self._send_arm_traj(pts)

    def go_home(self):
        self._send_head_traj(positions=[[0.0, 0.0]], durations=[0.7])
        self._send_arm_traj([([0.00, 0.0, 0.0, 0.0, 0.0], 1.0)])

    # ====== Base motion ======
    def drive_forward(self, speed=0.2, duration=2.0):
        twist = Twist(); twist.linear.x = float(speed)
        end_time = time.time() + float(duration)
        def _drive_loop():
            rate = 0.05
            while time.time() < end_time and rclpy.ok():
                self.base_pub.publish(twist); time.sleep(rate)
            self.base_pub.publish(Twist())
        threading.Thread(target=_drive_loop, daemon=True).start()

    # ====== Tiny preference memory updaters ======
    def _maybe_update_preferences(self, user_text: str):
        # naive extractors for "I like ..." / "I don't like ..."
        like = re.findall(r"\bI like ([^.,!]+)", user_text, flags=re.IGNORECASE)
        dislike = re.findall(r"\bI (?:do not|don't) like ([^.,!]+)", user_text, flags=re.IGNORECASE)
        changed = False
        for x in like:
            x = x.strip()
            if x and x.lower() not in [v.lower() for v in self.profile.get("likes", [])]:
                self.profile.setdefault("likes", []).append(x)
                changed = True
        for x in dislike:
            x = x.strip()
            if x and x.lower() not in [v.lower() for v in self.profile.get("dislikes", [])]:
                self.profile.setdefault("dislikes", []).append(x)
                changed = True
        if changed:
            try: _save_memory_to_disk(self.summary_memory, self.profile)
            except Exception: pass

    # ====== Safe TTS helpers ======
    def _bargein_active(self) -> bool:
        # user considered "barge-in" if VAD gate opens while TTS speaking
        return bool(self._gate_open)

    def speak_streaming_from_tokens(self, token_iter, mood="calm"):
        global ASR_ENABLED
        ASR_ENABLED = False
        _emotive.stop()

        _emotive.speak_chunks_streaming(
            token_iter,
            mood=mood,
            gesture_cb=lambda tag: self._gesture_from_tag(tag),
            bargein_flag_getter=self._bargein_active
        )

        ASR_ENABLED = True           
        self._asr_ignore_until = time.time() + 0.8


    def speak_once(self, text: str, mood="calm", ignore_extra=1.2):
        global ASR_ENABLED
        ASR_ENABLED = False            
        _emotive.stop()

        _emotive.speak_once(text, mood=mood)

        ASR_ENABLED = True            
        self._asr_ignore_until = time.time() + ignore_extra


    # ====== Intent & paralinguistic cues ======
    def _gesture_from_tag(self, tag: str):
        if tag == "NOD": self.gesture_nod()
        elif tag == "SHAKE": self.gesture_shake()

    def map_and_do_gesture(self, text: str, speaker: str = "assistant"):
        low = text.lower()
        if speaker == "user" and hasattr(self, "_last_tone"):
            t = getattr(self, "_last_tone", {})
            lbl = t.get("label", "")
            if lbl == "sad/tired":
                self.gesture_nod()
            elif lbl == "stressed/angry":
                self.gesture_shake()
        if speaker == "user" and re.search(r"\b(hello|hi|hey|yo|howdy)\b", low):
            now = time.time()
            if now - self._last_hello_ts > 2.0:
                self._last_hello_ts = now; self.gesture_wave(); return "HELLO_DEMO"
        if speaker == "assistant" and re.search(r"\b(hello|hi|hey)\b", low):
            return "ASSISTANT_GREETING"
        if re.search(r"\b(yes|yeah|yep|sure|ok|okay|affirm|agree|correct|right|sounds good)\b", low):
            self.gesture_nod();   return "NOD"
        if re.search(r"\b(no|nope|nah|cannot|can’t|won’t|disagree|negative)\b", low):
            self.gesture_shake(); return "SHAKE"
        if re.search(r"\b(thanks|thank you|goodbye|bye|see you|nice talking)\b", low):
            self.gesture_wave();  return "WAVE"
        if re.search(r"\b(home|reset pose|neutral)\b", low):
            self.go_home();       return "HOME"
        if re.search(r"\b(escort|guide|come with me|lead the way|follow me)\b", low):
            self.drive_forward(speed=0.25, duration=2.5); return "ESCORT"
        if re.search(r"\b(great|awesome|cool|nice|love|amazing|perfect)\b", low):
            self.gesture_nod();   return "NOD_POS"
        if re.search(r"\b(bad|hate|awful|terrible|no way|not good)\b", low):
            self.gesture_shake(); return "SHAKE_NEG"
        return None

    # ====== ASR (mic) with WebRTC-VAD + adaptive energy floor ======
    def _rms16(self, pcm_bytes):
        if not pcm_bytes: return 0.0
        x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0: return 0.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    # def start_mic(self):
    #     if self._asr_streaming or (self._asr_model is None or self._asr_rec is None):
    #         return
    #     self._asr_streaming = True

    #     sd.default.latency = ('high', 'high')  # more buffering headroom

    #     def _audio_cb(indata, frames, time_info, status):
    #         if status and "overflow" in str(status).lower():
    #             # optional: comment this line to silence overflow logs
    #             #print(f"[WARN] [llm_convo]: Audio overflow (harmless)")
    #             return
    #         try:
    #             pcm48 = np.frombuffer(indata, dtype=np.int16)
    #             # --- downsample 48 kHz → 16 kHz ---
    #             pcm16 = resample_poly(pcm48, up=1, down=3).astype(np.int16)
    #             pcm_bytes = pcm16.tobytes()

    #             if self._asr_rec.AcceptWaveform(pcm_bytes):
    #                 res = json.loads(self._asr_rec.Result()).get("text", "").strip()
    #                 if res:
    #                     self._asr_q.put(res)
    #         except Exception as e:
    #             print(f"[ASR] chunk error: {e}", file=sys.stderr)

    #     try:
    #         dev_index = _auto_select_mic()  # or _find_device("EarPods")
    #         if dev_index is None:
    #             self.get_logger().warning("No microphone detected; ASR disabled.")
    #             self._asr_streaming = False
    #             return

    #         self.get_logger().info(f"Using microphone device index: {dev_index} (48 kHz native)")
    #         self._asr_stream = sd.RawInputStream(
    #             samplerate=48000,
    #             blocksize=4096,       # larger block = fewer overflows
    #             dtype='int16',
    #             channels=1,
    #             device=dev_index,
    #             callback=_audio_cb
    #         )
    #         self._asr_stream.start()
    #         self.get_logger().info("Mic started (48 kHz downsampled → 16 kHz, VAD-gated Vosk ASR). Speak anytime.")
    #         # Let the robot record 2 s of neutral speech to calibrate
    #         self.get_logger().info("Calibrating baseline noise level...")
    #         time.sleep(2.0)
    #         try:
    #             x = np.array(list(self._tone.buf))
    #             if len(x) > 16000:  # 1 s of data
    #                 self._tone.calibrate_baseline(x)
    #                 self.get_logger().info("Voice baseline calibrated.")
    #         except Exception as e:
    #             self.get_logger().warn(f"Calibration skipped: {e}")

    #     except Exception as e:
    #         self._asr_streaming = False
    #         self.get_logger().warning(f"Mic init failed; continuing with text only: {e}")

    def start_mic(self):
        if self._asr_streaming or self.whisper_model is None:
           return
        self._asr_streaming = True

        import os
        dev_index = int(os.getenv("HSR_MIC_INDEX", _auto_select_mic() or 0))
        if dev_index is None:
            self.get_logger().warning("No microphone detected; ASR disabled.")
            self._asr_streaming = False
            return

        info = sd.query_devices(dev_index, 'input')
        sr = int(info['default_samplerate']) or 16000    # ← auto-detect native rate
        self.get_logger().info(f"Using mic index {dev_index} ({sr} Hz Whisper mode).")

        def _record_loop():
            duration = 3.0
            while self._asr_streaming and rclpy.ok():
                if not ASR_ENABLED:
                    time.sleep(0.05)
                    continue

                try:
                    self.get_logger().info("Listening…")

                    audio = sd.rec(
                        int(duration * sr),
                        samplerate=sr,
                        channels=1,
                        dtype='float32',
                        device=dev_index,
                    )

                    sd.wait()

                    # store raw audio for emotion model
                    self._last_audio_np = audio.flatten()
                    self._last_audio_sr = sr

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio, sr)
                        tmp_path = tmp.name

                    result = self.whisper_model.transcribe(tmp_path, language="en", fp16=False)
                    text = result.get("text", "").strip()

                    if text:
                        self._asr_q.put(text)

                    os.remove(tmp_path)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.get_logger().warning(f"[Whisper loop] {e}")
                    time.sleep(0.5)

        threading.Thread(target=_record_loop, daemon=True).start()
        self.get_logger().info("Whisper mic thread started.")

    
    def handle_start(self, request, response):
        """Start or stop the conversation service."""
        self.convo_on = bool(request.data)
        if self.convo_on:
            response.message = "Conversation started. Type or speak; press Ctrl+C to stop."
            threading.Thread(target=self._interaction_loop, daemon=True).start()
            # if _ASR_AVAILABLE and self._asr_model is not None and self._asr_rec is not None:
            if self.whisper_model is not None:
                try:
                    self.start_mic()
                except Exception as e:
                    print(f"[ASR] Mic start error: {e}", file=sys.stderr)
            else:
                print("[ASR] Vosk model not available; text-only mode.")
        else:
            response.message = "Conversation stopped."
            self._asr_streaming = False
            try:
                if hasattr(self, "_asr_stream"):
                    self._asr_stream.stop()
            except Exception:
                pass
        response.success = True
        return response

    def _poll_asr(self):
        # while not self._asr_q.empty():
        #     if time.time() < self._asr_ignore_until:
        #         _ = self._asr_q.get(); continue
        #     payload = self._asr_q.get()
        #     try:
        #         obj = json.loads(payload)
        #         user = obj.get("text", "")
        #         tone = obj.get("tone", self._last_tone)
        #         rms  = float(obj.get("rms", 0.0))
        #     except Exception:
        #         user = str(payload); tone = self._last_tone; rms = 0.0

        while not self._asr_q.empty():
            if time.time() < self._asr_ignore_until:
                _ = self._asr_q.get(); continue
            user = self._asr_q.get().strip()
            tone = self._last_tone
            rms = 0.0

            low = (user or "").lower().strip()
            if self._last_spoken_text and low == self._last_spoken_text:
                continue

            # Wake-word gate
            if self.require_wake:
                tokens = low.split()
                if not any(w in tokens for w in self._wake_words):
                    print("[ASR] Ignored (no wake word).")
                    continue
                for w in self._wake_words:
                    if low.startswith(w + " "):
                        user = user[len(w)+1:]
                        break

            print(f"\n[Mic] {user}  [tone={tone.get('label','calm')}, f0={tone.get('f0',0):.0f}Hz, wpm={tone.get('wpm',0):.0f}]")

            # low-confidence repair: very short + low RMS
            if len(user.split()) <= 2 and rms < (self._rms_floor * ENERGY_MARGIN_MULT * 1.1):
                confirm = f"Did you say “{user}”?"
                self.handle_assistant_initiated(confirm, mood=_tone_to_mood(tone))
                # small ignore window so user's correction isn't clipped by our TTS
                self._asr_ignore_until = time.time() + 0.6
            
        # ======== NEW EMOTION ANALYSIS (CRNN) ==========
            if hasattr(self, "_last_audio_np"):
                try:
                    emotion = predict_emotion(self._last_audio_np, self._last_audio_sr)
                except Exception as e:
                    self.get_logger().warn(f"Emotion model failed: {e}")
                    emotion = "neutral"
            else:
                emotion = "neutral"

            print(f"[Emotion] {emotion}")

            # Pass emotion to the main handler
            print(f"\n[Mic] {user} ...")

            _emotive.stop()
            self.handle_user_utterance(user, emotion=emotion)


    # ====== Idle nudge ======
    def _idle_nudge_timer(self):
        if not self.convo_on: return
        if (time.time() - self._last_user_at) > self._idle_prompt_after_s:
            self._last_user_at = time.time()
            # context-aware tiny follow-up using memory
            like_hint = ""
            if self.profile.get("likes"):
                like_hint = f" Want to chat about {random.choice(self.profile['likes'])}?"
            nudge = random.choice([
                "I'm here if you want to talk about anything—from music to robots.",
                "Curious about something? Ask me anything!",
                "Want to try a super quick game of 20 questions?",
            ]) + like_hint
            self.handle_assistant_initiated(nudge, mood=self.profile.get("last_mood","calm"))

    def handle_assistant_initiated(self, text: str, mood: str = "calm"):
        self.llm_history.append({"role":"assistant","content":text})
        print(f"HSR: {text}")
        self.speak_once(text, mood=mood)
        self.map_and_do_gesture(text, speaker="assistant")
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t":"assistant","text":text,"ts":time.time()})+"\n")
        except Exception:
            pass

    # ====== Conversation handling (tone-aware policy + streaming + backchannel) ======
    def _maybe_summarize(self):
        if self.turn_count % MAX_TURNS_BEFORE_SUMMARY != 0 or self.turn_count == 0:
            return
        base = [{"role":"system","content":_build_system_prompt()}]
        recent = self.llm_history[-24:]
        self.summary_memory = summarize_history_for_memory(self.summary_memory, self.llm_history)
        if self.summary_memory:
            base.append({"role":"system","content":"Conversation memory:\n" + self.summary_memory})
        self.llm_history = base + recent
        if len(self.llm_history) > HISTORY_HARD_CAP:
            self.llm_history = self.llm_history[-HISTORY_HARD_CAP:]
        try:
            _save_memory_to_disk(self.summary_memory, getattr(self, "profile", {}))
        except Exception:
            pass

    def _apply_soft_content_filters(self, user_text: str) -> str:
        if re.search(r"\b(hack|explosive|weapon|harm|kill|suicide|self-harm)\b", user_text.lower()):
            return ("I can’t help with that. Maybe we can chat about robotics, music, movies, or your projects instead?")
        return ""

    def _build_msgs(self, user_text: str, tone_hint: str):
        msgs = [{"role":"system","content":_build_system_prompt()}]
        # pass tiny robot state and profile to ground responses
        state_note = f"Robot state: head neutral. Profile: name={self.profile.get('name','')}, likes={self.profile.get('likes',[])[:3]}."
        msgs.append({"role":"system","content":state_note})
        if self.summary_memory:
            msgs.append({"role":"system","content":"Conversation memory:\n" + self.summary_memory})
        if tone_hint:
            msgs.append({"role":"system","content":tone_hint})
        msgs.extend(self.llm_history[1:])
        msgs.append({"role":"user","content":user_text})
        return msgs

    def _tone_policy_hint(self, tone: dict) -> str:
        if not tone: return ""
        lbl = tone.get('label','calm')
        base = (f"Paralinguistic cue: The user's voice sounds **{lbl}** "
                f"(approx f0={tone.get('f0',0):.0f} Hz, RMS={tone.get('rms',0):.3f}, WPM={tone.get('wpm',0):.0f}). ")
        if lbl == "sad/tired":
            policy = ("Behavior: Gently check in first. Start with one short question like "
                      "\"You sound a bit down—are you okay?\" Keep sentences short and validating.")
        elif lbl == "excited":
            policy = ("Behavior: Mirror positive mood. Acknowledge cheerfulness, invite sharing briefly.")
        elif lbl == "stressed/angry":
            policy = ("Behavior: De-escalate, acknowledge feelings, offer one practical next step. Slower pace.")
        else:
            policy = ("Behavior: Neutral, warm, helpful. One concise clarifying question if needed.")
        # update profile mood
        self.profile["last_mood"] = _tone_to_mood(tone)
        try: _save_memory_to_disk(self.summary_memory, self.profile)
        except Exception: pass
        return base + policy

    def _maybe_backchannel(self, reply_text: str) -> str:
        # insert at most one short backchannel 20% of time, if not starting with one already
        if random.random() > 0.2: return reply_text
        if re.match(r"^(mm|uh|hmm|got it|okay|nice|cool)\b", reply_text.strip().lower()):
            return reply_text
        bc = random.choice(["Mm-hmm.", "I see.", "Gotcha.", "Okay."])
        return f"{bc} {reply_text}"

    def handle_user_utterance(self, user: str, tone: dict = None, emotion: str = None):
        self._last_user_at = time.time()
        self._maybe_update_preferences(user)
        tag = self.map_and_do_gesture(user, speaker="user")
        steer = self._apply_soft_content_filters(user)
        if steer:
            self.speak_once(steer, mood=_tone_to_mood(tone))
            return
        tone = tone or getattr(self, "_last_tone", None)
        tone_hint = self._tone_policy_hint(tone)

        # Build LLM messages
        msgs = self._build_msgs(user, tone_hint)

        # STREAMING reply
        try:
            token_iter = stream_chat(msgs, temperature=TEMPERATURE, top_p=TOP_P)
        except Exception as e:
            print(f"[LLM stream error] {e} -> falling back to non-streaming")
            text = chatgpt_reply(msgs)
            self.llm_history.append({"role":"user","content":user})
            self.llm_history.append({"role":"assistant","content":text})
            print(f"HSR: {text}")
            self.speak_once(text, mood=_tone_to_mood(tone))
            self.map_and_do_gesture(text, speaker="assistant")
            self.turn_count += 1
            self._maybe_summarize()
            return

        # Capture streamed text, also feed to TTS progressively
        collected = []
        def _gen():
            for tok in token_iter:
                collected.append(tok)
                yield tok

        self.llm_history.append({"role":"user","content":user})
        print("HSR: ", end="", flush=True)
        self.speak_streaming_from_tokens(_gen(), mood=_tone_to_mood(tone))
        text = "".join(collected).strip()
        text = self._maybe_backchannel(text)
        print(text)

        # Save, gesture, and contextual follow-up
        self.llm_history.append({"role":"assistant","content":text})
        self.turn_count += 1
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t":"user","text":user,"ts":time.time(),"tone":tone}, ensure_ascii=False)+"\n")
                f.write(json.dumps({"t":"assistant","text":text,"ts":time.time(),"tone_used":bool(tone_hint)}, ensure_ascii=False)+"\n")
        except Exception:
            pass
        # Mid/post gesture mapping
        self.map_and_do_gesture(text, speaker="assistant")

        # Contextual follow-up only if no question already
        if not re.search(r"\?\s*$", text.strip()):
            follow = self._contextual_followup(user, text)
            if follow:
                self.handle_assistant_initiated(follow, mood=_tone_to_mood(tone))

        self._maybe_summarize()

    def _contextual_followup(self, user: str, reply: str) -> str:
        # tiny heuristic: ask one small, relevant follow-up
        if len(user.split()) < 3:  # very short user input → clarify
            return "Want me to keep it short and simple, or dive a bit deeper?"
        if "story" in user.lower():
            return "Want a short adventure story or something funny?"
        if self.profile.get("likes"):
            return f"Should we connect this to {random.choice(self.profile['likes'])}?"
        return ""

    # ====== Console loop + modes ======
    def _interaction_loop(self):
        print("\n[HSR GPT] You can type or just speak. (Wake-word required: {} | words: {})".format(self.require_wake, ", ".join(self._wake_words)))
        while self.convo_on:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.convo_on = False; break
            if not user: continue
            if user.lower() in ("wake on", "wake-on"):
                self.require_wake = True;  print("[HSR] Wake-word enabled.");  continue
            if user.lower() in ("wake off", "wake-off"):
                self.require_wake = False; print("[HSR] Wake-word disabled."); continue
            if user.lower() in ("lab mode", "lab-mode"):
                global OPEN_DOMAIN_MODE
                OPEN_DOMAIN_MODE = False
                print("[HSR] Switched to Lab mode (concise).")
                self.llm_history = [{"role":"system","content":_build_system_prompt()}]
                continue
            if user.lower() in ("chatty mode", "chatty", "open mode", "open-mode"):
                OPEN_DOMAIN_MODE = True
                print("[HSR] Switched to Open-domain chat mode.")
                self.llm_history = [{"role":"system","content":_build_system_prompt()}]
                continue
            if user.lower().startswith("name is "):
                self.profile["name"] = user.split(" ", 2)[-1].strip()
                _save_memory_to_disk(self.summary_memory, self.profile)
                print("[HSR] Noted your name.")
                continue
            if user.lower() in ("forget memory", "wipe memory", "reset memory"):
                self.summary_memory = "";  self.profile = {"name":"","likes":[],"dislikes":[],"last_mood":"calm"}
                _save_memory_to_disk(self.summary_memory, self.profile)
                print(f"[HSR] Memory cleared and saved to {MEM_FILE}")
                continue
            if user.lower() in ("save memory", "save"):
                _save_memory_to_disk(self.summary_memory, getattr(self, "profile", {}))
                print(f"[HSR] Memory saved to {MEM_FILE}")
                continue
            if user.lower() in ("export transcript", "show transcript", "transcript"):
                print(f"[HSR] Transcript at: {self._transcript_path}")
                continue
            # quick modes
            if user.lower() in ("story mode", "story"):
                self._run_story_mode(); continue
            if user.lower() in ("mindfulness mode", "breathe", "breathing"):
                self._run_mindfulness_mode(); continue

            self.handle_user_utterance(user)

    # ====== Simple Story Mode (entertainment scaffolding) ======
    def _run_story_mode(self):
        prompt = "Tell a short, cheerful 6–8 sentence adventure story for a child, with simple language and 1 question at the end."
        msgs = [{"role":"system","content":_build_system_prompt()},
                {"role":"user","content":prompt}]
        token_iter = stream_chat(msgs, temperature=0.9, top_p=0.95)
        self.speak_streaming_from_tokens(token_iter, mood=self.profile.get("last_mood","calm"))

    # ====== Simple Mindfulness Mode (wellness scaffolding) ======
    def _run_mindfulness_mode(self):
        script = [
            "Let's try a one-minute breathing exercise together.",
            "Sit comfortably. Gently lower your shoulders.",
            "We’ll breathe in for a count of four, and out for a count of six.",
            "Ready? Inhale… 1… 2… 3… 4…",
            "Exhale… 1… 2… 3… 4… 5… 6…",
            "Again—Inhale… 1… 2… 3… 4…",
            "Exhale… 1… 2… 3… 4… 5… 6…",
            "Nice. One more time—Inhale four… and exhale six…",
            "How do you feel—lighter, calm, or the same?"
        ]
        for line in script:
            self.speak_once(line, mood="calm", ignore_extra=0.4)
            # small nod on encouragement lines
            if re.search(r"\bNice|Ready|Again|One more\b", line):
                self.gesture_nod()
            time.sleep(0.2)

def main():
    rclpy.init()
    node = HsrGestures()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    _emotive.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
