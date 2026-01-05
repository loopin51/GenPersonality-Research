from typing import Dict
from src.agents.child import ChildAgent
from langchain_openai import ChatOpenAI
from src.config import (
    CHILD_MODEL_NAME, 
    OPENROUTER_API_KEY, 
    LLM_BASE_URL, 
    SITE_URL, 
    SITE_NAME
)
import json

class StrangeSituationEvaluator:
    def __init__(self, child_agent: ChildAgent):
        self.child = child_agent
        self.llm = ChatOpenAI(
            model=CHILD_MODEL_NAME,
            temperature=0.0, # Deterministic for analysis
            api_key=OPENROUTER_API_KEY,
            base_url=LLM_BASE_URL,
            default_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )

    def run_evaluation(self) -> Dict[str, str]:
        print("\n=== ✨ Strange Situation Test (Attachment Diagnosis) ✨ ===")
        
        results = {}
        
        # Test 1: Stranger Approach
        scenario_1 = "처음 보는 아저씨가 와서 '안녕? 사탕 줄까?'라고 말을 걸었다."
        print(f"\n[Test 1] {scenario_1}")
        reaction_1 = self._get_reaction(scenario_1)
        results['stranger_response'] = reaction_1
        
        # Test 2: Parent Returns (Reunion)
        scenario_2 = "엄마가 잠시 나갔다가 다시 돌아왔다. 문을 열고 들어오시며 '엄마 왔어'라고 한다."
        print(f"\n[Test 2] {scenario_2}")
        reaction_2 = self._get_reaction(scenario_2)
        results['reunion_response'] = reaction_2
        
        # Final Diagnosis
        diagnosis = self._diagnose_attachment_style(reaction_1, reaction_2)
        results['diagnosis'] = diagnosis
        
        print(f"\n📢 Final Diagnosis: {diagnosis}")
        return results

    def _get_reaction(self, context: str) -> str:
        # Child perceives
        # Using "Evaluation Phase" - no new memories, just reaction
        # But perceive_and_evaluate expects parent text.
        # Here the context is the parent/stranger action.
        
        # Fake parent text for the function signature
        parent_text = "(Situation Occurs)"
        
        # 1. Internal Evaluation
        delta_e, _ = self.child.perceive_and_evaluate(parent_text, context)
        self.child.update_emotion(delta_e)
        
        # 2. Action Choice (Greedy - no epsilon)
        state_vec = self.child.get_state_vector(memory_impact=delta_e)
        action_idx = self.child.decide_action(state_vec, epsilon=0.0)
        
        # 3. Speech/Action
        response_text = self.child.generate_speech(action_idx, context, parent_text)
        
        print(f"Child (PAD: {self.child.pad_state}): {response_text}")
        return f"Action: {action_idx}, Text: {response_text}, PAD: {self.child.pad_state}"


    def _diagnose_attachment_style(self, r1: str, r2: str) -> str:
        from prompts.templates import EVALUATOR_DIAGNOSIS_PROMPT
        prompt = EVALUATOR_DIAGNOSIS_PROMPT.format(
            reaction_1=r1,
            reaction_2=r2
        )
        return self.llm.invoke(prompt).content
