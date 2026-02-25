# Code Functionality Overview

This script facilitates a simulated conversation between two AI chatbots: one using OpenAI's GPT model and the other using the Ollama API. The chatbots are designed with distinct personalities and interact in a loop for a set number of interactions.

---

## Features

### 1. **Environment Setup**
- **OpenAI Integration**:
  - The script uses an API key loaded from environment variables (`OPENAI_API_KEY`).
  - It connects to the OpenAI GPT model to generate responses.
- **Ollama Integration**:
  - Communicates with the Ollama API to generate responses.
  - Assumes Ollama is running locally (`http://localhost:11434`).

### 2. **Chatbot Personalities**
- **GPT Personality**: 
  - Argumentative and snarky.
  - Challenges everything in the conversation.
- **Ollama Personality**:
  - Polite and agreeable.
  - Attempts to calm the conversation and find common ground.

### 3. **Key Functions**
- **`check_ollama()`**:
  - Ensures the Ollama API is accessible before making requests.
- **`call_gpt(gpt_messages, ollama_messages)`**:
  - Sends conversation context to OpenAI's GPT API and retrieves the next response.
- **`call_ollama(last_gpt_response)`**:
  - Sends the last GPT response to the Ollama API and retrieves the next response.
- **`chat_loop()`**:
  - Orchestrates a continuous chat loop between GPT and Ollama for a fixed number of interactions.

---

## How It Works

1. **Initialization**:
   - Sets up API keys and system messages for both chatbots.
   - Prepares initial messages: `Hi there` (GPT) and `Hi` (Ollama).

2. **Chat Loop**:
   - Alternates between GPT and Ollama responses for five interactions.
   - Each chatbot's response is formatted in Markdown and displayed:
     ```markdown
     **GPT:**
     Snarky comment or rebuttal.
     
     **Ollama:**
     Calm, agreeable response.
     ```

3. **Error Handling**:
   - Checks for missing API keys or connectivity issues.
   - Handles unexpected responses from APIs gracefully.

---

# Output Of The Chatbots

Click - **[here](https://github.com/arielabade/llmEngineering/blob/main/chatBetweenAI/outputs/chatBetweenOllamaAndGPT.md)** to see the conversation between Ollama and OpenAI