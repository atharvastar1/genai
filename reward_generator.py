import os
import json
import re
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from dotenv import load_dotenv
load_dotenv()

# Setup OpenAI-compatible client
if HAS_OPENAI:
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        base_url=os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    )
else:
    client = None

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
        # Expert Fallback Candidates for Acrobot Swing-up
        return [
            {
                "critique": "Potential energy baseline.",
                "draft": "Maximize height sum to push joints upward.",
                "code": "def reward_fn(obs):\n    # obs[0]=cos(th1), obs[2]=cos(th2)\n    # Height is high when cos values are negative\n    h1 = -obs[0]\n    h2 = -obs[2]\n    return h1 + h2"
            },
            {
                "critique": "Energy pumping strategy.",
                "draft": "Reward angular velocity when below goal to build momentum.",
                "code": "def reward_fn(obs):\n    height = -obs[0] - obs[2]\n    # Pump energy if we are low\n    velocity_bonus = 0.1 * (abs(obs[4]) + abs(obs[5])) if height < 1.0 else 0.0\n    return height + velocity_bonus"
            },
            {
                "critique": "Singular point avoidance.",
                "draft": "Reward height but penalize extreme joint folding.",
                "code": "def reward_fn(obs):\n    height = -obs[0] - obs[2]\n    # obs[2] is cos of joint 2 relative to joint 1 usually\n    folding_penalty = 0.5 * abs(obs[3]) # Penalize sin of joint 2\n    return height - folding_penalty"
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
        if not HAS_OPENAI:
            raise ImportError("OpenAI library not installed.")
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
