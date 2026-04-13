import json
import os
from reward_generator import generate_reward_candidates
from trainer import train_agent

HISTORY_FILE = "history.json"

STATUS_FILE = "status.json"

def update_status(message, progress=0, busy=True):
    with open(STATUS_FILE, "w") as f:
        json.dump({"message": message, "progress": progress, "busy": busy}, f)
        f.flush()
        os.fsync(f.fileno())

def run_iteration(iteration_num):
    update_status(f"Starting Iteration {iteration_num}...", 5)
    print(f"\n--- Iteration {iteration_num} ---")
    
    # Load history safely
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    history = json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load history.json ({e}). Starting fresh.")
            
    # Generate 3 candidates
    update_status("Agentic Drafting & Self-Critikue Phase...", 15)
    candidates = generate_reward_candidates(history)
    
    if not candidates:
        update_status("Error: Optimization failed.", 0, False)
        return
    
    iteration_results = []
    
    for i, c_data in enumerate(candidates):
        prog = 20 + (i * 25)
        # Handle both raw strings and dicts for backward/fallback compatibility
        if isinstance(c_data, dict):
            code = c_data.get("reward_code", "")
            critique = c_data.get("critique", "N/A")
            draft = c_data.get("draft_logic", "N/A")
        else:
            code = c_data
            critique = "No critique in basic mode"
            draft = "Simple reward function"

        update_status(f"Training Refined Candidate {i+1}/3...", prog)
        try:
            # REDUCED TIMESTEPS: To see variety and increase difficulty
            diag, model = train_agent(code, total_timesteps=2000)
            result = {
                "iteration": iteration_num,
                "candidate": i + 1,
                "reward_code": code,
                "critique": critique,
                "draft": draft,
                "raw_score": diag["avg_reward"],
                "score": diag["avg_reward"], # Placeholder for now
                "narrative": diag["failure_summary"],
                "trajectory": diag["sample_trajectory"],
                "stability": diag.get("stability_score", "0%"),
                "centering": diag.get("centering_score", "0%")
            }
            iteration_results.append(result)
            
            # STREAM PARTIAL RESULTS
            with open("latest_group.json", "w") as f:
                json.dump(iteration_results, f)
                
        except Exception as e:
            iteration_results.append({
                "iteration": iteration_num,
                "candidate": i + 1,
                "reward_code": code,
                "raw_score": 0.0,
                "score": -100.0,
                "narrative": f"Failure: {str(e)}"
            })

    # --- GRPO LOGIC: Reward Best, Punish Rest ---
    # We calculate the Relative Advantage across the group of 3
    raw_scores = [r["raw_score"] for r in iteration_results]
    mean_score = sum(raw_scores) / len(raw_scores)
    
    # Calculate Standard Deviation safely
    variance = sum((s - mean_score) ** 2 for s in raw_scores) / len(raw_scores)
    std_dev = (variance ** 0.5) + 1e-8 # Add epsilon to prevent division by zero

    # --- KNOWLEDGE BANK: Extraction ---
    # Save lessons learned for the next run
    lessons = []
    if os.path.exists("knowledge_bank.json"):
        with open("knowledge_bank.json", "r") as f:
            lessons = json.load(f)
    
    worst_candidate = min(iteration_results, key=lambda x: x["advantage"])
    lessons.append({
        "iteration": iteration_num,
        "lesson": f"Candidate {worst_candidate['candidate']} failed because: {worst_candidate['narrative']}. Avoid this strategy."
    })
    
    with open("knowledge_bank.json", "w") as f:
        json.dump(lessons[-10:], f, indent=4) # Keep last 10 lessons

    # Save full group results for UI Comparison
    with open("latest_group.json", "w") as f:
        json.dump(iteration_results, f)
        
    # Pick the best from this iteration based on relative advantage
    best_candidate = max(iteration_results, key=lambda x: x["advantage"])
    
    # Update global history
    history.append(best_candidate)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
    update_status(f"Iteration {iteration_num} complete. Best Raw Score: {best_candidate['raw_score']:.1f}", 100, False)
    return iteration_results

if __name__ == "__main__":
    # Auto-detect next iteration number
    it_start = 1
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                h = json.load(f)
                if h:
                    it_start = h[-1]["iteration"] + 1
        except: pass
        
    for i in range(it_start, it_start + 3): # Run next 3 iterations
        run_iteration(i)
