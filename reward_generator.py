import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "EMPTY"),
    base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
)
MODEL = os.getenv("LLM_MODEL", "llama3")

SYSTEM_PROMPT = """
You are an expert Reinforcement Learning reward engineer.
Your goal is to design a reward function for the "CartPole-v1" environment using an AGENTIC REFINEMENT loop.

Environment Context:
- obs[0]: Cart Position (-4.8 to 4.8)
- obs[1]: Cart Velocity
- obs[2]: Pole Angle (~ -24 to 24 degrees)
- obs[3]: Pole Angular Velocity

THE REFINEMENT LOOP:
1. DRAFT: Propose a reward function.
2. CRITIQUE: Analyze potential flaws (e.g., instability, reward hacking, energy waste).
3. FINAL REWARD: Output the final, refined code.

YOUR OUTPUT MUST BE A JSON OBJECT WITH EXACTLY THESE FIELDS:
{
    "draft_logic": "Explain the initial idea here.",
    "critique": "Identify potential faults and safety concerns.",
    "reward_code": "def reward_fn(obs):\\n    # logic...\\n    return result"
}
"""

def generate_reward_candidates(history=None, num_candidates=3):
    # FALLBACK MODE FOR DEMO (if no API Key provided)
    if not os.getenv("LLM_API_KEY") or os.getenv("LLM_API_KEY") == "your_api_key_here":
        it_num = len(history) + 1 if history else 1
        
        # Simulated Expert-Critique outcomes for Demo
        # We return a list of DICTIONARIES now
        candidates = []
        for i in range(num_candidates):
            v = it_num + i
            candidates.append({
                "draft_logic": f"Stabilize the angle based on quadratic penalty v{v}.",
                "critique": "Initial draft might be too sensitive to minor jitters. Adding angular velocity damping.",
                "reward_code": f"def reward_fn(obs):\n    # Agentic Version {v}\n    return 1.0 - (obs[2]**2) - 0.05 * abs(obs[3])"
            })
        return candidates

    # REAL LLM MODE
    prompt = f"Design {num_candidates} different reward function candidates using the DRAFT-CRITIQUE-REFINEMENT loop."
    
    if history:
        history_text = "\n".join([
            f"It {h['iteration']}: Score {h['score']:.1f}, Code: {h['reward_code']}, Summary: {h.get('narrative', 'N/A')}" 
            for h in history
        ])
        prompt += f"\n\nPrevious Research History:\n{history_text}\n\nPlease improve based on the previous iterations."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=2048,
            response_format={ "type": "json_object" } # Force JSON if supported
        )
        
        content = response.choices[0].message.content
        # Try to parse JSON directly or find it
        try:
            # If the model returned multiple objects or text around it, we'd need more logic.
            # For simplicity, we assume one JSON object containing a list or we loop the call.
            # Let's handle a single refined candidate per call for high quality.
            data = json.loads(content)
            return [data] # Return it in a list
        except:
             # Fallback: Extract code block only
             code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
             return [{"draft_logic": "Refinement", "critique": "None", "reward_code": c} for c in code_blocks]
             
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return [{
            "draft_logic": "Fallback",
            "critique": "Manual override",
            "reward_code": "def reward_fn(obs): return 1.0 - abs(obs[2])"
        }]

if __name__ == "__main__":
    candidates = generate_reward_candidates()
    print(json.dumps(candidates, indent=2))
