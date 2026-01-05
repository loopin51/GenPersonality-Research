import sys
import os
from dotenv import load_dotenv

# Mock config to speed up test
import src.config
src.config.TOTAL_EPISODES = 2 # Only run 2 episodes for testing
src.config.PHASE_1_EPISODES = 1
src.config.PHASE_2_EPISODES = 1
src.config.PHASE_3_EPISODES = 0

from src.simulation import SimulationEngine

def test_main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping test: OPENAI_API_KEY not found. Please set it to run the test.")
        # For the sake of the agent run, we might want to fake it if we can't run real calls, 
        # but the user likely has keys if they asked for this.
        return

    print("=== TEST RUN: Child Simulation (Warm Parent) ===")
    # Initialize Engine
    engine = SimulationEngine(parent_type="Warm")
    
    # Run
    engine.run_simulation()
    
    print("\nTest Complete. Checking generated files...")
    # List generated csv
    files = [f for f in os.listdir('.') if f.startswith('sim_results_') and f.endswith('.csv')]
    if files:
        print(f"Generated Log File: {files[0]}")
    else:
        print("Error: No log file generated.")

if __name__ == "__main__":
    test_main()
