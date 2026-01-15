#!/bin/bash
# Quick start script for Voice Assistant

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found.${NC}"
    echo "Please run: ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if main script exists
if [ ! -f "voice_assistant.py" ]; then
    echo -e "${RED}Error: voice_assistant.py not found${NC}"
    exit 1
fi

# Check if model exists
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    echo -e "${RED}Error: LLM model not found${NC}"
    echo "Please run: ./setup.sh to download the model"
    exit 1
fi

# Run assistant
echo -e "${GREEN}Starting Voice Assistant...${NC}"
python3 voice_assistant.py

# Deactivate on exit
deactivate



