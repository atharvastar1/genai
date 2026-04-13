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

KNOWLEDGE_FILE = "knowledge_bank.json"

def get_knowledge():
    if not os.path.exists(KNOWLEDGE_FILE):
        return "No previous lessons learned yet."
    with open(KNOWLEDGE_FILE, "r") as f:
        data = json.load(f)
        return "\n".join([f"- Lesson: {k['lesson']}" for k in data[-5:]])

def generate_reward_candidates(iteration, history):
    knowledge = get_knowledge()
    
    prompt = f"""
    You are an AI Reward Engineer optimizing 'Acrobot-v1'. 
    OBJECTIVE: Swing the two-joint arm above the goal line (Height > 1.0).
    OBSERVATION SPACE (6 values):
    - obs[0], obs[1]: cos/sin of joint 1 angle
    - obs[2], obs[3]: cos/sin of joint 2 angle
    - obs[4], obs[5]: angular velocities
    
    KNOWLEDGE FROM PREVIOUS RUNS:
    {knowledge}
    
    TASK: Generate 3 candidate Python reward functions.
    Format your response as a JSON list of 3 objects:
    {{"critique": "why previous failed", "draft": "strategy", "code": "def reward_fn(obs): ..."}}
    """
    
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        # Professional Mock Candidates for Acrobot
        return [
            {
                "critique": "Basic potential energy reward.",
                "draft": "Maximize height by using cos values.",
                "code": "def reward_fn(obs):\n    # Height is roughly -cos(theta1) - cos(theta1+theta2)\n    height = -obs[0] - (obs[0]*obs[2] - obs[1]*obs[3])\n    return height"
            },
            {
                "critique": "Adding velocity damping to prevent wild spinning.",
                "draft": "Height + negative velocity penalty.",
                "code": "def reward_fn(obs):\n    height = -obs[0] - (obs[0]*obs[2] - obs[1]*obs[3])\n    velocity_penalty = 0.1 * (abs(obs[4]) + abs(obs[5]))\n    return height - velocity_penalty"
            },
            {
                "critique": "Joint centering to keep momentum efficient.",
                "draft": "Height + centered sin bonus.",
                "code": "def reward_fn(obs):\n    height = -obs[0] - (obs[0]*obs[2] - obs[1]*obs[3])\n    shaping = 0.5 * (1.0 - abs(obs[1])) # Bonus for being vertical\n    return height + shaping"
            }
        ]

    # REAL LLM MODE
    history_text = "No history yet."
    if history:
        history_text = "\n".join([
            f"It {h['iteration']}: Score {h['score']:.1f}, Code: {h['reward_code']}, Summary: {h.get('narrative', 'N/A')}" 
            for h in history
        ])

    final_prompt = prompt + f"\n\nPrevious Research History:\n{history_text}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional Reinforcement Learning Reward Engineer."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.8,
            max_tokens=2048,
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
