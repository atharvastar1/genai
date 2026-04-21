#!/bin/bash
# FORGE-GRPO Professional Setup Script

echo "🔍 Checking System Requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install it to continue."
    exit 1
fi

echo "📦 Initializing Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "⬆️ Upgrading Package Manager..."
pip install --upgrade pip

echo "🏗️ Installing Core RL Frameworks..."
pip install gymnasium stable-baselines3 torch numpy pandas

echo "🌐 Installing Web & AI Middleware..."
pip install openai python-dotenv flask flask-cors

# Initial environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️ Created .env file. Update it with your API key if needed."
fi

echo ""
echo "✅ Setup Complete. Run ./run.sh to launch the dashboard."
chmod +x run.sh
