# =============================================================================
# LOCAL LLM EVALUATION - Ollama backend
# =============================================================================
# A simple, readable demo that:
#   1. Runs a local LLM via Ollama
#   2. Scores responses across several task types
#   3. Logs runtime metrics (latency, tokens/sec)
#   4. Saves structured results to a JSON file
#
# HOW TO USE:
#   1. pip install -r requirements.txt
#   2. ollama pull <model>  (replace with local model name)
#   3. python ollama-eval.py
#
# TO SWAP MODELS:
#   Edit the MODEL variable below and re-run.
#   Compare the resulting JSON files to see performance differences.
# =============================================================================

import subprocess
import time

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from eval_utils import TASKS, score_response, print_summary, save_results

# =============================================================================
# CONFIG - edit these values to change behaviour
# =============================================================================

MODEL = "gemma4:e2b"        # swap to compare
TEMPERATURE = 0.3           # lower = more deterministic; try 0.0 or 0.7 too

# Sanitize model name for use as a filename
MODEL_SAFE_NAME = MODEL.replace(":", "_").replace(" ", "_").lower()
RESULTS_FILE = f"results_{MODEL_SAFE_NAME}.json"

# =============================================================================
# STEP 1 - run a single prompt and return the response + metrics
# =============================================================================

def run_prompt(llm, prompt):
    """Send a prompt to the ChatOllama model and return text + timing metadata."""

    start_time = time.perf_counter()
    response = llm.invoke([HumanMessage(content=prompt)])
    end_time = time.perf_counter()

    text = response.content
    
    # Extract token counts from usage_metadata if available
    usage_metadata = getattr(response, 'usage_metadata', None) or {}
    tokens_in = usage_metadata.get('input_tokens') or 0
    tokens_out = usage_metadata.get('output_tokens') or 0
    
    # Calculate latency and tokens per second from elapsed time
    latency_ms = (end_time - start_time) * 1000
    elapsed_sec = end_time - start_time
    tokens_per_sec = round(tokens_out / elapsed_sec, 1) if elapsed_sec > 0 else 0

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
        "tokens_per_sec": tokens_per_sec,
    }

# =============================================================================
# STEP 2 - run every task, print live progress, collect results
# =============================================================================

print(f"\n{'='*60}")
print(f"  Local LLM Evaluation  (Ollama)")
print(f"  Model      : {MODEL}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Tasks      : {len(TASKS)}")
print(f"{'='*60}\n")

# Start ollama serve in background
print("Starting Ollama server...")
proc = subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)

# Initialize ChatOllama client
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, base_url="http://localhost:11434")

# Wait for ollama to be ready before proceeding
max_retries = 10
for attempt in range(max_retries):
    try:
        # Test connection by attempting a simple invocation
        llm.invoke([HumanMessage(content="test")])
        print("Ollama server is ready!\n")
        break
    except Exception:
        if attempt < max_retries - 1:
            time.sleep(1)
        else:
            print("ERROR: Could not connect to Ollama after 10 retries.")
            proc.terminate()
            raise SystemExit(1)

try:
    results = []

    for i, task in enumerate(TASKS, start=1):
        print(f"\n[{i:02d}/{len(TASKS)}] Starting {task['task']:<16} task...")
        print(f"    Prompt: {task['prompt'][:70]}..." if len(task['prompt']) > 70 else f"    Prompt: {task['prompt']}")
        print(f"    Running inference...", end="", flush=True)

        result = run_prompt(llm, task["prompt"])
        score  = score_response(task, result["text"])

        print(f" Done!")
        print(f"    Result: {score:<10} | {result['latency_ms']:>8.0f} ms  | {result['tokens_per_sec']:>6.1f} tok/s")
        print(f"    Tokens: {result['tokens_in']} in, {result['tokens_out']} out")

        # Store everything - the full output text is included so you can read it
        results.append({
            "task": task["task"],
            "prompt": task["prompt"],
            "score": score,
            "output": result["text"],
            "latency_ms": result["latency_ms"],
            "tokens_in": result["tokens_in"],
            "tokens_out": result["tokens_out"],
            "tokens_per_sec": result["tokens_per_sec"],
        })

    print(f"\n{'='*60}")
    print(f"  All tasks completed! Processing results...")
    print(f"{'='*60}")

    # STEP 3 - print summary and save results
    print_summary(MODEL, results)
    save_results(MODEL, TEMPERATURE, results, RESULTS_FILE, script_type="ollama")

    print("To compare models: change MODEL at the top of the script and re-run.")
    print("Each model's results are saved to a separate file for easy comparison.\n")

finally:
    # Clean up ollama process and all children
    print("\nCleaning up Ollama server...")
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], timeout=3)
    except Exception:
        pass
