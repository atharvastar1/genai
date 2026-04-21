import json
import os
import concurrent.futures
from reward_generator import generate_reward_candidates
from trainer import train_agent

HISTORY_FILE = "history.json"
STATUS_FILE = "status.json"

def update_status(message, progress=0, busy=True):
    temp_path = STATUS_FILE + ".tmp"
    with open(temp_path, "w") as f:
        json.dump({"message": message, "progress": progress, "busy": busy}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, STATUS_FILE)

def train_candidate_wrapper(args):
    """Wrapper for parallel execution of trainer."""
    iteration_num, candidate_index, c_data, total_timesteps = args
    
    if isinstance(c_data, dict):
        code = c_data.get("code") or c_data.get("reward_code", "")
        critique = c_data.get("critique", "N/A")
        draft = c_data.get("draft") or c_data.get("draft_logic", "N/A")
    else:
        code = c_data
        critique = "No critique in basic mode"
        draft = "Simple reward function"

    try:
        diag, model = train_agent(code, total_timesteps=total_timesteps)
        return {
            "iteration": iteration_num,
            "candidate": candidate_index + 1,
            "reward_code": code,
            "critique": critique,
            "draft": draft,
            "raw_score": diag["avg_reward"],
            "score": diag["avg_reward"],
            "narrative": diag["failure_summary"],
            "trajectory": diag["sample_trajectory"],
            "stability": diag.get("stability_score", "0%"),
            "centering": diag.get("centering_score", "0%"),
            "advantage": 0.0
        }
    except Exception as e:
        return {
            "iteration": iteration_num,
            "candidate": candidate_index + 1,
            "reward_code": code,
            "raw_score": -500.0,
            "score": -500.0,
            "advantage": -1.0,
            "narrative": f"Execution Failure: {str(e)}",
            "trajectory": [],
            "stability": "0%",
            "centering": "0%",
            "draft": draft
        }

def run_iteration(iteration_num):
    update_status(f"Starting Iteration {iteration_num}...", 5)
    print(f"\n--- Iteration {iteration_num} ---")
    
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if content: history = json.loads(content)
        except: pass
            
    update_status("Agentic Drafting Phase...", 15)
    candidates = generate_reward_candidates(iteration_num, history)
    
    if not candidates:
        update_status("Error: Optimization failed.", 0, False)
        return

    update_status(f"Parallel Training: 3 Candidates...", 30)
    
    # Prepare arguments for parallel execution
    train_args = [
        (iteration_num, i, c_data, 10000) 
        for i, c_data in enumerate(candidates)
    ]
    
    iteration_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(train_candidate_wrapper, arg): arg for arg in train_args}
        for future in concurrent.futures.as_completed(futures):
            iteration_results.append(future.result())
            # Update status incrementally
            progress = 30 + (len(iteration_results) * 20)
            update_status(f"Progress: {len(iteration_results)}/3 Trained...", progress)

    # Calculate Advantages (GRPO Logic)
    raw_scores = [r["raw_score"] for r in iteration_results]
    mean_score = sum(raw_scores) / len(raw_scores)
    variance = sum((s - mean_score) ** 2 for s in raw_scores) / len(raw_scores)
    std_dev = (variance ** 0.5) + 1e-8

    for r in iteration_results:
        r["advantage"] = (r["raw_score"] - mean_score) / std_dev

    # Knowledge Bank Update
    lessons = []
    if os.path.exists("knowledge_bank.json"):
        with open("knowledge_bank.json", "r") as f: lessons = json.load(f)
    
    worst_candidate = min(iteration_results, key=lambda x: x["advantage"])
    lessons.append({
        "iteration": iteration_num,
        "lesson": f"Strategy '{worst_candidate['draft']}' achieved {worst_candidate['raw_score']:.1f}. Improvement needed."
    })
    
    with open("knowledge_bank.json", "w") as f: json.dump(lessons[-10:], f, indent=4)
    with open("latest_group.json", "w") as f: json.dump(iteration_results, f, indent=4)
    
    best_candidate = max(iteration_results, key=lambda x: x["advantage"])
    history.append(best_candidate)
    
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
        
    update_status(f"Iteration {iteration_num} complete. Best Score: {best_candidate['raw_score']:.1f}", 100, False)
    return iteration_results

if __name__ == "__main__":
    it_start = 1
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                h = json.load(f)
                if h: it_start = h[-1]["iteration"] + 1
        except: pass
        
    for i in range(it_start, it_start + 3):
        run_iteration(i)
