import sys
import os
from unittest.mock import MagicMock, patch
import json

# Mock config dependencies
import src.config
src.config.TOTAL_EPISODES = 12
src.config.PHASE_1_EPISODES = 2
src.config.PHASE_2_EPISODES = 10
src.config.PHASE_3_EPISODES = 0

# Mock LLM Responses
def mock_llm_invoke(messages):
    content = ""
    msg_str = str(messages)
    
    if "Role" in msg_str and ("Inner Mind" in msg_str or "무의식적 자아" in msg_str):
        # Reflection Prompt -> JSON
        content = '{"new_core_beliefs": ["엄마가 화내면 사과해야 한다.", "나는 나쁜 아이다."]}'
    elif "인지 평가 시스템" in msg_str or "Cognitive Emotion System" in msg_str:
        # Child Evaluator returning JSON
        content = '{"delta_p": -0.1, "delta_a": 0.2, "delta_d": -0.1}'
    elif "전략을 수행하세요" in msg_str or "Instruction" in msg_str:
        # Child Speech (Text)
        content = "엄마 미안해요. (고개를 푹 숙이며 뒷걸음질 친다)"
    elif "Response Generation Task" in msg_str or "Parenting Persona Constraints" in msg_str:
        # Parent Speech -> JSON
        content = '{"internal_thought": "아이가 겁을 먹었군.", "response_speech": "그래, 다음부터는 조심해라."}'
    elif "발달 심리학자" in msg_str or "Diagnostic Task" in msg_str:
         # Evaluator Diagnosis (Text)
         content = "불안-회피 애착: 재회 상황에서 부모에게 다가가지 않고 장난감에만 집중하는 '회피적 무시' 행동이 관찰됨."
    else:
        content = "Mock Response"
        
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp

def test_mock_main():
    print("=== MOCK TEST RUN: Child Simulation (Logic Only) ===")
    
    # Patch MemorySystem to avoid DB issues
    with patch('src.agents.child.ChatOpenAI') as MockChildLLM, \
         patch('src.agents.parent.ChatOpenAI') as MockParentLLM, \
         patch('src.evaluation.ChatOpenAI') as MockEvalLLM, \
         patch('src.simulation.MemorySystem') as MockMemorySys:

         # Setup LLM Mocks
        child_llm_instance = MockChildLLM.return_value
        child_llm_instance.invoke.side_effect = mock_llm_invoke
        # Support .bind(response_format=...).invoke(...)
        child_llm_instance.bind.return_value = child_llm_instance
        
        parent_llm_instance = MockParentLLM.return_value
        parent_llm_instance.invoke.side_effect = mock_llm_invoke
        parent_llm_instance.bind.return_value = parent_llm_instance
        
        eval_llm_instance = MockEvalLLM.return_value
        eval_llm_instance.invoke.side_effect = mock_llm_invoke
        eval_llm_instance.bind.return_value = eval_llm_instance
        
        # Setup Memory Mock
        memory_instance = MockMemorySys.return_value
        # Mock retrieve to return empty list or dummy memories
        # Dummy memory structure
        dummy_mem = {
            "metadata": {
                "trigger": "Test Trigger",
                "action": "Test Action",
                "outcome": "Test Outcome",
                "delta_p": 0.0, "delta_a": 0.0, "delta_d": 0.0
            }
        }
        memory_instance.retrieve.return_value = [dummy_mem]

        from src.simulation import SimulationEngine
        
        # Instantiate Engine (it will use the mocked MemorySystem class)
        engine = SimulationEngine(parent_type="Warm")

        
        # Run
        engine.run_simulation()
        
        
        print("\nMock Test Complete.")
        
        # Verify Logs
        import glob
        logs = glob.glob("log/sim_detailed_*.jsonl")
        if logs:
            print(f"Verified Detailed Log Created: {logs[0]}")
        else:
            print("ERROR: Detailed Log NOT found.")

if __name__ == "__main__":
    test_mock_main()
