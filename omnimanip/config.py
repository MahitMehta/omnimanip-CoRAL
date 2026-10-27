import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

VLM_MODEL = os.getenv(
    "VLM_MODEL", "gemini/gemini-3.8-flash"
)  # any litellm model id; key read from .env by litellm
VLM_REASONING = {
    "rrc": "medium"
}  # litellm reasoning_effort per prompt; other prompts: the provider default
HOST = os.getenv(
    "HOST", "100.89.37.109"
)  # the GPU box running every sidecar (CoRAL, or the isaac workstation)
# sidecar ports; request schemas in ur5e-manip-hardware/<name>_server/server.py (all msgpack except anygrasp: pickle)
SIDECARS = {
    "pose": 5667,
    "trellis2": 5669,
    "sam3": 5670,
    # CuroboServer-OmniManip (same image and code as Albus's on 5671): motion.py resets and refills the ESDF, which
    # must never touch the live map his bridge feeds
    "curobo": int(os.getenv("CUROBO_PORT", "5681")),
    "any6d": 5672,
    "oriany": 5673,
    "anygrasp": int(os.getenv("ANYGRASP_PORT", "5666")),
}
