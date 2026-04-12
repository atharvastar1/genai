#!/bin/bash
echo "🚀 Setting up FORGE-GRPO Mini project..."

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Initial environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Created .env file for you. Update it with your API key if needed."
fi

echo "✅ Setup complete! run './run.sh' to start the dashboard."
chmod +x run.sh
