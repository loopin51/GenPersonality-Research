import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import (
    CHILD_MODEL_NAME, 
    OPENROUTER_API_KEY, 
    LLM_BASE_URL, 
    SITE_URL, 
    SITE_NAME,
    PAD_MIN, PAD_MAX, 
    WEIGHT_P, WEIGHT_D, WEIGHT_A, AROUSAL_THRESHOLD, 
    EMOTION_DECAY_RATE, MEMORY_ALPHA
)
from src.models import PADState, ActionType, MemoryItem
from src.memory import MemorySystem

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ChildAgent:
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self.pad_state = PADState(0.0, 0.0, 0.0) # Baseline
        
        # RL Setup
        # State Dim: P, A, D (3) + Emotion Shock History (3) = 6 (Simplified for now)
        # Or maybe just P, A, D + Last Reward? Let's stick to P, A, D + Context Vector from Memory (let's simplify to 3 PAD + 3 Memory Impact = 6)
        self.state_dim = 6 
        self.action_dim = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dqn = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        # LLM for Cognitive Tasks
        self.llm = ChatOpenAI(
            model=CHILD_MODEL_NAME,
            temperature=0.8,
            api_key=OPENROUTER_API_KEY,
            base_url=LLM_BASE_URL,
            default_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )
        self.core_beliefs = [] # List of strings

    def get_state_vector(self, memory_impact: List[float] = [0.0, 0.0, 0.0]) -> np.ndarray:
        """Combine current PAD and memory impact into a state vector for DQN."""
        return np.concatenate([self.pad_state.to_vector(), np.array(memory_impact)])

    def decide_action(self, state_vector: np.ndarray, epsilon: float) -> int:
        """Epsilon-Greedy Action Selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.dqn(state_tensor)
            return q_values.argmax().item()


    def perceive_and_evaluate(self, parent_text: str, context: str) -> Tuple[List[float], List[Dict]]:
        """
        Cognitive Evaluator:
        1. Retrieve relevant memories.
        2. Ask LLM to calculate delta PAD.
        """
        # 1. Retrieve
        query = f"{context} Parent said: {parent_text}"
        retrieved_memories = self.memory.retrieve(query, k=3)
        
        memory_context_str = ""
        for i, mem in enumerate(retrieved_memories):
            m_meta = mem['metadata']
            memory_context_str += f"- Memory {i+1}: When '{m_meta['trigger']}', I did '{m_meta['action']}', Parent said '{m_meta['outcome']}'. Emotion Impact: P={m_meta['delta_p']}, A={m_meta['delta_a']}, D={m_meta['delta_d']}\n"

        if not memory_context_str:
            memory_context_str = "No specific relevant memories."

        # 2. LLM Evaluator (Load from prompts.templates)
        from prompts.templates import CHILD_EVALUATOR_PROMPT
        evaluator_prompt = CHILD_EVALUATOR_PROMPT.format(
            context=context,
            parent_text=parent_text,
            p_val=f"{self.pad_state.P:.2f}",
            a_val=f"{self.pad_state.A:.2f}",
            d_val=f"{self.pad_state.D:.2f}",
            memory_context=memory_context_str,
            core_beliefs=self._format_beliefs()
        )

        # Define JSON Schema for Structured Output
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "emotion_delta",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "delta_p": {
                            "type": "number",
                            "description": "Change in Pleasure (-1.0 to 1.0)"
                        },
                        "delta_a": {
                            "type": "number",
                            "description": "Change in Arousal (-1.0 to 1.0)"
                        },
                        "delta_d": {
                            "type": "number",
                            "description": "Change in Dominance (-1.0 to 1.0)"
                        }
                    },
                    "required": ["delta_p", "delta_a", "delta_d"],
                    "additionalProperties": False
                }
            }
        }

        try:
            # Bind response_format to the LLM call
            structured_llm = self.llm.bind(response_format=json_schema, max_tokens=4096)
            response = structured_llm.invoke(evaluator_prompt).content
            
            import json
            # Remove markdown code blocks if present (OpenRouter/Model variance safety)
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            deltas = json.loads(cleaned_response)
            delta_e = [float(deltas.get('delta_p', 0)), float(deltas.get('delta_a', 0)), float(deltas.get('delta_d', 0))]
        except Exception as e:
            print(f"Evaluator Error: {e}, using default.")
            print(f"Raw Response causing error: {response}")
            delta_e = [0.0, 0.0, 0.0]

        return delta_e, retrieved_memories

    def update_emotion(self, delta_e: List[float]):
        """Apply delta to current state and clip."""
        # Simple addition as per Receipt (could be weighted)
        self.pad_state.P += delta_e[0]
        self.pad_state.A += delta_e[1]
        self.pad_state.D += delta_e[2]
        self.pad_state.clip()

    def generate_speech(self, action_idx: int, context: str, parent_text: str) -> str:
        """
        Generates what the child actually says/does based on the chosen Action Strategy.
        Uses prompts/templates.py
        """
        action_name = ActionType.get_name(action_idx)
        action_desc = ActionType.get_description(action_idx)
        
        from prompts.templates import CHILD_SPEECH_PROMPT
        prompt = CHILD_SPEECH_PROMPT.format(
            context=context,
            parent_text=parent_text,
            p_val=f"{self.pad_state.P:.2f}",
            a_val=f"{self.pad_state.A:.2f}",
            d_val=f"{self.pad_state.D:.2f}",
            action_name=action_name,
            action_desc=action_desc
        )

        return self.llm.invoke(prompt, max_tokens=4096).content

    def calculate_reward(self, delta_pad: List[float], next_a: float) -> float:
        """
        Reward Function from Project_receipt.md 5.4
        R = w_p * dP + w_d * dD - w_a * max(0, A_next - A_thresh)
        """
        dP, dA, dD = delta_pad
        penalty_a = max(0, next_a - AROUSAL_THRESHOLD)
        
        reward = (WEIGHT_P * dP) + (WEIGHT_D * dD) - (WEIGHT_A * penalty_a)
        return reward

    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Single Step DQN Update (Simplified for online learning)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        
        # Q(s, a)
        q_values = self.dqn(state_t)
        q_value = q_values.gather(1, action_t)
        
        # Target Q = r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.dqn(next_state_t)
            next_max_q = next_q_values.max(1)[0].unsqueeze(1)
            target_q = reward_t + 0.9 * next_max_q # gamma=0.9 fixed
            
        loss = self.loss_fn(q_value, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def decay_emotion(self):
        self.pad_state.decay(EMOTION_DECAY_RATE)

    def _format_beliefs(self) -> str:
        if not self.core_beliefs:
            return "None yet."
        return "\n".join([f"- {b}" for b in self.core_beliefs])

    def reflect(self, start_episode: int, end_episode: int):
        """
        Periodically reflect on recent memories to form/update Core Beliefs.
        Uses prompts/templates.py
        """
        memories = self.memory.get_recent_memories(start_episode, end_episode)
        if not memories:
            return

        # Prepare summary of recent events
        events_str = ""
        for m in memories:
            meta = m['metadata']
            events_str += f"- When '{meta['trigger']}', I did '{meta['action']}' -> Parent said: '{meta['outcome']}' (Emotion change: P={meta['delta_p']:.1f}, D={meta['delta_d']:.1f})\n"

        from prompts.templates import CHILD_REFLECTION_PROMPT
        prompt = CHILD_REFLECTION_PROMPT.format(
            events_str=events_str,
            current_beliefs=self._format_beliefs()
        )

        # Schema for list of beliefs
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "core_beliefs_update",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "new_core_beliefs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of new or updated core beliefs."
                        }
                    },
                    "required": ["new_core_beliefs"],
                    "additionalProperties": False
                }
            }
        }

        try:
            structured_llm = self.llm.bind(response_format=json_schema, max_tokens=4096)
            response = structured_llm.invoke(prompt).content
            
            import json
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_response)
            new_beliefs = data.get("new_core_beliefs", [])
        except Exception as e:
            print(f"Reflection Error: {e}")
            print(f"Raw Response causing error: {response}")
            new_beliefs = []
        
        # In a real sophisticated system we would merge/deduplicate, but for now we append/replace keys
        # Let's just keep the last 5 unique beliefs to avoid explosion
        for b in new_beliefs:
            if b not in self.core_beliefs:
                self.core_beliefs.append(b)
        
        
        print(f"\n[Reflection] Derived Beliefs: {new_beliefs}")
        return new_beliefs

    def save_model(self, path: str):
        """Save the DQN weights (Personality)."""
        torch.save(self.dqn.state_dict(), path)
        print(f"Model saved to {path}")
