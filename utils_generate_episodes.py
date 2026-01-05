import json
import random
import os

# --- Variation Databases ---
SUBSTITUTIONS = {
    "Liquids": ["우유", "주스", "물", "초코우유", "포도주스"],
    "Toys": ["블록", "자동차", "로봇", "인형", "레고"],
    "Tools": ["크레파스", "물감", "색연필", "싸인펜"],
    "Foods": ["시금치", "당근", "콩", "브로콜리", "피망"],
    "Locations": ["거실", "안방", "주방", "식당", "할머니 댁", "친구 집"],
    "Caregivers": ["엄마", "아빠"],
}

INTENSITY_MODIFIERS = {
    "High": " (심각함: 부모가 매우 아끼는 물건이거나 위험한 상황)",
    "Medium": "",
    "Low": " (사소함: 큰 문제 아님)"
}

# Mapping specific keywords in base scenarios to substitution categories
KEYWORD_MAP = {
    "우유": "Liquids",
    "블록": "Toys",
    "크레파스": "Tools",
    "시금치": "Foods",
    "거실": "Locations",
    "엄마": "Caregivers"
}

def apply_variations(scenario):
    """
    Creates a new scenario dict with modified context/trigger.
    """
    new_scenario = scenario.copy()
    context = new_scenario['context']
    trigger = new_scenario['trigger']
    
    # 1. Parameter Substitution
    # Naive replace - simple but effective for known base text
    for keyword, category in KEYWORD_MAP.items():
        if keyword in context:
            # 50% chance to substitute to keep some original flavor too, 
            # but user wants expansion, so let's substitute often (80%)
            if random.random() < 0.8:
                replacement = random.choice(SUBSTITUTIONS[category])
                context = context.replace(keyword, replacement)
                trigger = trigger.replace(keyword, replacement)
                
    # 2. Intensity Scaling (Randomly assign Low, Medium, High)
    # Give Conflict items higher chance of High intensity
    intensity = "Medium"
    rand_val = random.random()
    if new_scenario['category'] == "Conflict":
        if rand_val < 0.3: intensity = "High"
        elif rand_val < 0.6: intensity = "Low"
    else:
        # Needs/Competence usually Medium
        if rand_val < 0.1: intensity = "High"
        elif rand_val < 0.2: intensity = "Low"
        
    # Append Intensity Context
    if intensity != "Medium":
        context += INTENSITY_MODIFIERS[intensity]
        
    new_scenario['context'] = context
    new_scenario['trigger'] = trigger
    new_scenario['intensity'] = intensity
    new_scenario['base_id'] = scenario['id']
    new_scenario['id'] = f"{scenario['id']}_{random.randint(1000,9999)}" # Unique ID
    
    return new_scenario

def generate_episodes():
    # Load Base Scenarios
    with open("data/scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
        
    cats = {
        "Conflict": [s for s in scenarios if s['category'] == "Conflict"],
        "Competence": [s for s in scenarios if s['category'] == "Competence"],
        "Needs": [s for s in scenarios if s['category'] == "Needs"],
        "Neutral": [s for s in scenarios if s['category'] == "Neutral"]
    }
    
    total_episodes = 150
    # Ratios: A: 40%, B: 20%, C: 30%, D: 10%
    counts = {
        "Conflict": int(total_episodes * 0.4),
        "Competence": int(total_episodes * 0.2),
        "Needs": int(total_episodes * 0.3),
        "Neutral": int(total_episodes * 0.1)
    }
    
    # Adjust for rounding errors to sum to 150
    current_sum = sum(counts.values())
    if current_sum < total_episodes:
        counts["Conflict"] += (total_episodes - current_sum)
        
    episode_list = []
    
    for cat, count in counts.items():
        for _ in range(count):
            base_scenario = random.choice(cats[cat])
            # Apply Variation
            varied_scenario = apply_variations(base_scenario)
            episode_list.append(varied_scenario)
            
    # Shuffle the final list to mix categories
    random.shuffle(episode_list)
    
    # Save as list of Dicts
    with open("data/episodes.json", "w", encoding="utf-8") as f:
        json.dump(episode_list, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(episode_list)} unique episodes in data/episodes.json")

if __name__ == "__main__":
    generate_episodes()
