import sys
import os
from dotenv import load_dotenv
from src.simulation import SimulationEngine

def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        return

    print("=== Child Personality Development Simulation ===")
    print("Select Parent Persona:")
    print("1. Warm (Affectionate, Validating)")
    print("2. Cold (Strict, Result-Oriented)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        parent_type = "Warm"
    elif choice == "2":
        parent_type = "Cold"
    else:
        print("Invalid choice. Defaulting to Warm.")
        parent_type = "Warm"

    print(f"\nStarting Simulation with {parent_type} Parent...")
    engine = SimulationEngine(parent_type)
    engine.run_simulation()

if __name__ == "__main__":
    main()
