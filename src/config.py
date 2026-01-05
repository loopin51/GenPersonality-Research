import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Settings ---
# --- LLM Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-3.5-turbo") # Default if not set
LLM_BASE_URL = "https://openrouter.ai/api/v1"
SITE_URL = os.getenv("YOUR_SITE_URL", "https://localhost")
SITE_NAME = os.getenv("YOUR_SITE_NAME", "ChildLLM")

CHILD_MODEL_NAME = MODEL_NAME
PARENT_MODEL_NAME = MODEL_NAME

# --- Simulation Settings ---
TOTAL_EPISODES = 10
MAX_TURNS_PER_EPISODE = 3

# --- RL Settings ---
# Phase 1: Exploration
EPSILON_PHASE_1_START = 0.8
EPSILON_PHASE_1_END = 0.5
PHASE_1_EPISODES = 3

# Phase 2: Pattern Forming
EPSILON_PHASE_2_START = 0.5
EPSILON_PHASE_2_END = 0.1
PHASE_2_EPISODES = 7 # 4~10

# Phase 3: Consolidation
EPSILON_PHASE_3 = 0.05
PHASE_3_EPISODES = 0 # Skipped for 10-episode test

# --- Emotion Settings ---
PAD_MIN = -1.0
PAD_MAX = 1.0
EMOTION_DECAY_RATE = 0.2  # Lambda in receipt (assumption, will make adjustable)

# --- Memory Settings ---
VECTOR_DB_PATH = "./chroma_db"
MEMORY_RETRIEVAL_K = 3
MEMORY_ALPHA = 0.6 # Default weight for memory impact in Evaluator

# --- Reward Weights (Project_receipt.md 5.4) ---
WEIGHT_P = 1.0  # Pleasure increase
WEIGHT_D = 1.0  # Dominance increase
WEIGHT_A = 1.5  # Penalty for excess Arousal
AROUSAL_THRESHOLD = 0.7 
