# =============================================================================
# LOCAL LLM EVALUATION - llama-cpp-python backend
# =============================================================================
# A simple, readable demo that:
#   1. Runs a local GGUF model via llama-cpp-python + LangChain ChatLlamaCpp
#   2. Scores responses across several task types
#   3. Logs runtime metrics (latency, tokens/sec)
#   4. Saves structured results to a JSON file
#
# HOW TO USE:
#   1. pip install -r requirements.txt
#   2. Download a GGUF model file (e.g. from HuggingFace)
#   3. Set MODEL_PATH below to the absolute path of the .gguf file
#   4. python llama-cpp-eval.py
#
# TO SWAP MODELS:
#   Edit MODEL_PATH (and optionally MODEL_SAFE_NAME) below and re-run.
#   Compare the resulting JSON files to see performance differences.
# =============================================================================

import time

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage

from eval_utils import TASKS, score_response, print_summary, save_results

# =============================================================================
# CONFIG - edit these values to change behaviour
# =============================================================================

# Absolute path to your GGUF model file
MODEL_PATH = r"C:\models\gemma-4-E2B-it-Q4_K_M.gguf"

# Human-readable name used in results output and the JSON filename
MODEL_SAFE_NAME = "gemma4_e2b_q4km"

TEMPERATURE   = 0.3    # lower = more deterministic; try 0.0 or 0.7 too
N_CTX         = 4096   # context window size (tokens)
N_GPU_LAYERS  = -1     # layers to offload to GPU; -1 = all (set 0 for CPU-only)
MAX_TOKENS    = 2048    # max tokens to generate per response

RESULTS_FILE = f"results_{MODEL_SAFE_NAME}.json"

# =============================================================================
# STEP 1 - load model once at startup
# =============================================================================

print(f"\n{'='*60}")
print(f"  Local LLM Evaluation  (llama-cpp-python)")
print(f"  Model      : {MODEL_SAFE_NAME}")
print(f"  Path       : {MODEL_PATH}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Tasks      : {len(TASKS)}")
print(f"{'='*60}\n")

print("Loading model (this may take a moment)...")
llm = ChatLlamaCpp(
    model_path=MODEL_PATH,
    temperature=TEMPERATURE,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    max_tokens=MAX_TOKENS,
    verbose=False,
)
print("Model loaded!\n")

# =============================================================================
# STEP 2 - run a single prompt and return the response + metrics
# =============================================================================

def run_prompt(prompt):
    """Send a prompt to the llama-cpp model and return text + timing metadata."""

    start = time.perf_counter()
    
    # Use ChatLlamaCpp's invoke() method which handles chat formatting properly
    # This ensures instruction-tuned models (like Gemma 4) get correct formatting
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    text = response.content
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # For token tracking with chat interface, we need to estimate since
    # the chat wrapper may not expose raw token counts
    # Count approximate tokens (rough estimate: ~1 token per word)
    prompt_tokens = len(prompt.split())
    output_tokens = len(text.split()) if text.strip() else 0

    elapsed_sec = elapsed_ms / 1000
    tokens_per_sec = round(output_tokens / elapsed_sec, 1) if elapsed_sec > 0 and output_tokens > 0 else 0.0

    return {
        "text": text,
        "tokens_in": prompt_tokens,
        "tokens_out": output_tokens,
        "latency_ms": round(elapsed_ms, 1),
        "tokens_per_sec": tokens_per_sec,
    }

# =============================================================================
# STEP 3 - run every task, print live progress, collect results
# =============================================================================

results = []

for i, task in enumerate(TASKS, start=1):
    print(f"\n[{i:02d}/{len(TASKS)}] Starting {task['task']:<16} task...")
    print(f"    Prompt: {task['prompt'][:70]}..." if len(task['prompt']) > 70 else f"    Prompt: {task['prompt']}")
    print(f"    Running inference...", end="", flush=True)

    result = run_prompt(task["prompt"])
    
    # Debug: Show if output is empty
    if not result["text"].strip():
        print(f" Done! (WARNING: Empty response)")
    else:
        print(f" Done!")
        
    score  = score_response(task, result["text"])
    print(f"    Result: {score:<10} | {result['latency_ms']:>8.0f} ms  | {result['tokens_per_sec']:>6.1f} tok/s")
    print(f"    Tokens: {result['tokens_in']} in, {result['tokens_out']} out")

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

# STEP 4 - print summary and save results
print_summary(MODEL_SAFE_NAME, results)
save_results(MODEL_SAFE_NAME, TEMPERATURE, results, RESULTS_FILE, script_type="llamacpp")

print("To compare models: change MODEL_PATH and MODEL_SAFE_NAME at the top and re-run.")
print("Each model's results are saved to a separate file for easy comparison.\n")
