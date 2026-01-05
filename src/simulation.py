import random
import time
from typing import List, Dict
import pandas as pd
from datetime import datetime

from src.config import (
    TOTAL_EPISODES, MAX_TURNS_PER_EPISODE,
    PHASE_1_EPISODES, PHASE_2_EPISODES, PHASE_3_EPISODES,
    EPSILON_PHASE_1_START, EPSILON_PHASE_1_END,
    EPSILON_PHASE_2_START, EPSILON_PHASE_2_END,
    EPSILON_PHASE_3
)
from src.models import SimulationLog, MemoryItem, PADState
from src.agents.child import ChildAgent
from src.agents.parent import ParentAgent
from src.memory import MemorySystem

import json
import os

# Scenarios will be loaded from file
# SCENARIOS_DB is not strictly needed for the loop anymore if playlist has full objects
# but beneficial for reference or if playlist only had IDs. 
# New Architecture: EPISODE_PLAYLIST contains full objects.
EPISODE_PLAYLIST = []

def load_data():
    global EPISODE_PLAYLIST
    
    # Load Episode Playlist (which now contains full varied scenarios)
    try:
        with open("data/episodes.json", "r", encoding="utf-8") as f:
            EPISODE_PLAYLIST = json.load(f)
    except FileNotFoundError:
        print("Error: data/episodes.json not found. Run utils_generate_episodes.py first.")
        EPISODE_PLAYLIST = []

class SimulationEngine:
    def __init__(self, parent_type: str):
        # Load data if not loaded
        if not EPISODE_PLAYLIST:
            load_data()
            
        self.parent_type = parent_type
        self.memory_system = MemorySystem(collection_name=f"child_mem_{parent_type.lower()}_{int(time.time())}")
        self.child = ChildAgent(self.memory_system)
        self.parent = ParentAgent(parent_type)
        self.logs = []
        self.detailed_logs = [] # Detailed event stream
        
    def get_epsilon(self, episode_idx: int) -> float:
        """Calculate Epsilon based on Phase."""
        # 1-indexed episode number
        n = episode_idx + 1
        
        if n <= PHASE_1_EPISODES:
            # Linear decay from Start to End
            progress = (n - 1) / (PHASE_1_EPISODES - 1)
            return EPSILON_PHASE_1_START - progress * (EPSILON_PHASE_1_START - EPSILON_PHASE_1_END)
        
        elif n <= PHASE_1_EPISODES + PHASE_2_EPISODES:
            # Phase 2
            p2_idx = n - PHASE_1_EPISODES
            progress = (p2_idx - 1) / (PHASE_2_EPISODES - 1)
            return EPSILON_PHASE_2_START - progress * (EPSILON_PHASE_2_START - EPSILON_PHASE_2_END)
        
        else:
            # Phase 3
            return EPSILON_PHASE_3

    def _log_event(self, event_type: str, data: Dict):
        """Helper to append to detailed logs."""
        self.detailed_logs.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        })

    def run_episode(self, episode_idx: int, scenario: Dict):
        # 1. Setup
        epsilon = self.get_epsilon(episode_idx)
        
        print(f"\n=== Episode {episode_idx+1} [Phase {self.get_phase(episode_idx+1)}] ===")
        # scenario is now a dict passed directly
        print(f"Scenario: {scenario['trigger']} (ID: {scenario['id']}, Epsilon: {epsilon:.2f})")
        print(f"Context: {scenario['context']}")
        
        self._log_event("episode_start", {
            "episode": episode_idx + 1,
            "phase": self.get_phase(episode_idx+1),
            "scenario_id": scenario['id'],
            "context": scenario['context'],
            "epsilon": epsilon
        })
        
        # 2. Parent Initial Action
        # Ideally parent reacts to context first, but for now we let parent speak first based on context
        parent_text = self.parent.respond(scenario['context'], "(Situation Start)")
        print(f"[Parent]: {parent_text}")

        self._log_event("parent_speak", {
            "text": parent_text,
            "context": "Initial Situation"
        })
        
        transcript = [{"role": "system", "text": scenario['context']}, {"role": "parent", "text": parent_text}]
        
        current_state_vec = self.child.get_state_vector() # Initial state
        
        # 3. Loop
        total_reward = 0
        
        for turn in range(MAX_TURNS_PER_EPISODE):
            # --- Child Turn ---
            # Perceive (Evaluate Parent's last text)
            delta_e, memories = self.child.perceive_and_evaluate(parent_text, scenario['context'])
            self._log_event("child_evaluator", {
                "parent_input": parent_text,
                "delta_pad": delta_e,
                "memories_retrieved": len(memories)
            })
            
            self.child.update_emotion(delta_e)
            self._log_event("child_emotion_update", {
                "new_pad": self.child.pad_state.to_vector().tolist()
            })
            
            # Decide Action
            # State for RL includes current mood + memory impact
            # Simplified: memory impact is currently implicitly in PAD if we updated it first.
            # But let's pass explicit delta_e as "input features" to state if we want short-term reaction
            state_vec = self.child.get_state_vector(memory_impact=delta_e)
            action_idx = self.child.decide_action(state_vec, epsilon)
            
            # Act (Generate Speech)
            child_text = self.child.generate_speech(action_idx, scenario['context'], parent_text)
            print(f"[Child ({self.child.pad_state})]: {child_text} (Action: {action_idx})")
            
            self._log_event("child_act", {
                "action_idx": action_idx,
                "speech": child_text,
                "pad_state": str(self.child.pad_state)
            })

            transcript.append({"role": "child", "text": child_text, "action": action_idx, "pad": str(self.child.pad_state)})
            
            # --- Parent Turn ---
            parent_text_response = self.parent.respond(scenario['context'], child_text)
            print(f"[Parent]: {parent_text_response}")
            
            self._log_event("parent_speak", {
                "text": parent_text_response,
                "trigger_child_text": child_text
            })

            transcript.append({"role": "parent", "text": parent_text_response})
            parent_text = parent_text_response # Update for next turn
            
            # --- Reward Calculation & Learning ---
            # We need "Next State" to learn. 
            # In this simplification, "Next State" is the state AFTER Parent's response and Child's evaluation of it.
            # But that happens in the NEXT loop iteration.
            # So standard Deep Q-Learning usually requires (s, a, r, s').
            # We will approximate: 
            #   Reward is calculated based on "Immediate emotional change" (delta_e we just got) 
            #   BUT wait, delta_e was from the PREVIOUS parent text.
            #   The reward for THIS action is how the parent responds NEXT.
            #   So we need to evaluate the *new* parent_text immediately to get the reward.
            
            new_delta_e, _ = self.child.perceive_and_evaluate(parent_text, scenario['context'])
            # We don't update emotion yet for the state, just calculate reward? 
            # Or we update it to get s'. Let's update.
            self.child.update_emotion(new_delta_e)
            
            reward = self.child.calculate_reward(new_delta_e, self.child.pad_state.A)
            next_state_vec = self.child.get_state_vector(memory_impact=new_delta_e)
            
            loss = self.child.learn(state_vec, action_idx, reward, next_state_vec)
            total_reward += reward
            
            self._log_event("rl_learning", {
                "reward": reward,
                "loss": loss
            })

            # Save Memory of this turn
            mem_item = MemoryItem(
                episode_id=f"{episode_idx}_{turn}",
                trigger=scenario['trigger'],
                action=child_text, # Storing text action
                outcome=parent_text,
                emotion_impact=new_delta_e,
                timestamp=episode_idx
            )
            self.memory_system.add_memory(mem_item)
            self._log_event("memory_stored", {
                "episode_id": f"{episode_idx}_{turn}",
                "trigger": scenario['trigger']
            })
            
        # 4. End of Episode
        self.child.decay_emotion()
        self._log_event("episode_end", {
            "total_reward": total_reward,
            "final_pad": self.child.pad_state.to_vector().tolist()
        })
        
        # Log
        self.logs.append({
            "episode": episode_idx,
            "phase": self.get_phase(episode_idx+1),
            "scenario": scenario['trigger'],
            "final_pad": self.child.pad_state.to_vector(),
            "total_reward": total_reward,
            "epsilon": epsilon
        })
        
    def get_phase(self, n):
        if n <= PHASE_1_EPISODES: return 1
        if n <= PHASE_1_EPISODES + PHASE_2_EPISODES: return 2
        return 3

    def run_simulation(self):
        # Determine number of episodes to run based on available playlist or TOTAL_EPISODES constant
        # We should respect TOTAL_EPISODES from config
        count = min(TOTAL_EPISODES, len(EPISODE_PLAYLIST))
        
        for i in range(count):
            scenario = EPISODE_PLAYLIST[i] # This is now the object
            self.run_episode(i, scenario)
            
            # Periodic Reflection (Every 10 episodes, starting from Phase 2)
            # Phase 1 ends at PHASE_1_EPISODES
            if (i + 1) > PHASE_1_EPISODES and (i + 1) % 10 == 0:
                start_ep = (i + 1) - 10
                end_ep = i # inclusive 0-indexed
                new_beliefs = self.child.reflect(start_ep, end_ep)
                self._log_event("reflection", {
                    "new_beliefs": new_beliefs,
                    "all_core_beliefs": self.child.core_beliefs
                })
        
        # Save logs
        os.makedirs("log", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        timestamp = int(time.time())
        
        # 1. Summary CSV
        df = pd.DataFrame(self.logs)
        filename_csv = f"log/sim_results_{self.parent_type}_{timestamp}.csv"
        df.to_csv(filename_csv, index=False)
        print(f"Simulation Complete. Results saved to {filename_csv}")
        
        # 2. Detailed JSONL
        filename_jsonl = f"log/sim_detailed_{self.parent_type}_{timestamp}.jsonl"
        with open(filename_jsonl, "w", encoding="utf-8") as f:
            for entry in self.detailed_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Detailed logs saved to {filename_jsonl}")

        # Save Model
        filename_model = f"output/child_model_{self.parent_type}_{timestamp}.pth"
        self.child.save_model(filename_model)

        # Run Final Evaluation (Strange Situation)
        from src.evaluation import StrangeSituationEvaluator
        evaluator = StrangeSituationEvaluator(self.child)
        diagnosis_results = evaluator.run_evaluation()
        
        self._log_event("final_diagnosis", diagnosis_results)
        
        # Update detailed log with diagnosis
        with open(filename_jsonl, "a", encoding="utf-8") as f:
             f.write(json.dumps({
                 "timestamp": time.time(),
                 "event_type": "final_diagnosis",
                 "data": diagnosis_results
             }, ensure_ascii=False) + "\n")

        # Save Diagnosis
        filename_diag = f"output/diagnosis_{self.parent_type}_{timestamp}.txt"
        with open(filename_diag, "w", encoding="utf-8") as f:
            for k, v in diagnosis_results.items():
                f.write(f"[{k}]\n{v}\n\n")
        print(f"Diagnosis saved to {filename_diag}")
