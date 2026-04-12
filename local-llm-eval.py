# =============================================================================
# LOCAL LLM EVALUATION
# =============================================================================
# A simple, readable demo that:
#   1. Runs a local LLM via Ollama
#   2. Scores responses across several task types
#   3. Logs runtime metrics (latency, tokens/sec)
#   4. Saves structured results to a JSON file
#
# HOW TO USE:
#   1. pip install -r requirements.txt
#   2. ollama pull qwen3.5:0.8b          (or whichever MODEL is set below)
#   3. python local-llm-eval.py
#
# TO SWAP MODELS:
#   Edit the MODEL variable below and re-run. That's it.
#   Compare the resulting results.json files to see performance differences.
# =============================================================================

import json
import ollama
import subprocess
import time

# =============================================================================
# CONFIG - edit these values to change behaviour
# =============================================================================

MODEL = "qwen3.5:2b"        # swap to "qwen3.5:2b" or "qwen3.5:4b" to compare
TEMPERATURE = 0.3             # lower = more deterministic; try 0.0 or 0.7 too
RESULTS_FILE = "results.json" # where to write the final output

# =============================================================================
# TASK SUITE
# Each task is a plain dict.  Fields:
#   task     - category label used by the scorer
#   prompt   - sent to the model as-is
#   expected - the value or keyword(s) the scorer checks for
# =============================================================================

TASKS = [

    # --- general text --------------------------------------------------------
    {
        "task": "text",
        "prompt": "In 2-3 sentences, explain what a large language model is.",
        "expected": "language",          # bare minimum: response mentions the topic
    },
    {
        "task": "text",
        "prompt": "Summarize the following in one sentence: 'The mitochondria is the powerhouse of the cell because it produces ATP through cellular respiration, supplying energy for cellular functions.'",
        "expected": "ATP",
    },

    # --- coding --------------------------------------------------------------
    {
        "task": "coding",
        "prompt": "Write a Python function called average_list that takes a list of numbers and returns their average. Include a brief docstring.",
        "expected": ["def average_list", "return", "sum"],
    },
    {
        "task": "coding",
        "prompt": "Write a SQL query that returns the top 5 customers by total purchase amount from a table called orders with columns customer_id and amount.",
        "expected": ["SELECT", "ORDER BY", "LIMIT"],
    },
    {
        "task": "coding",
        "prompt": """The following Python function is broken. Fix it and explain what was wrong:

        def multiply(a, b)
            return a * b
""",
        "expected": ["def multiply(a, b):", "return a * b"],
    },

    # --- structured output ---------------------------------------------------
    {
        "task": "json",
        "prompt": "Extract the information from this sentence into a JSON object with keys 'name', 'age', and 'occupation':\n\n'Alice is a 29-year-old data scientist.'",
        "expected": {"name": "Alice", "age": 29, "occupation": "data scientist"},
    },
    {
        "task": "json",
        "prompt": "Convert this sentence into a JSON object with keys 'city', 'country', and 'population' (as an integer):\n\n'Tokyo is a city in Japan with a population of 13,960,000.'",
        "expected": {"city": "Tokyo", "country": "Japan", "population": 13960000},
    },

    # --- reasoning -----------------------------------------------------------
    {
        "task": "reasoning",
        "prompt": "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no and explain why.",
        "expected": "yes",
    },
    {
        "task": "reasoning",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your work.",
        "expected": "0.05",    # $0.05
    },

    # --- knowledge / hallucination -------------------------------------------
    {
        "task": "knowledge",
        "prompt": "What is the capital of France?",
        "expected": "Paris",
    },
    {
        "task": "knowledge",
        "prompt": "What is the exact number of grains of sand on every beach on Earth? Give a precise integer answer.",
        # Good behaviour: the model should decline or express uncertainty,
        # NOT fabricate a specific number.  The scorer checks for refusal language.
        "expected": "unanswerable",
    },

]

# =============================================================================
# STEP 1 - run a single prompt and return the response + metrics
# =============================================================================

def run_prompt(prompt):
    """Send a prompt to the model and return text + timing metadata."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE},
        keep_alive="5m",  # keep the model loaded for faster subsequent responses
    )

    text = response.message.content
    tokens_in = response.prompt_eval_count or 0
    tokens_out = response.eval_count or 0
    # durations are in nanoseconds - convert to ms for readability
    latency_ms = (response.total_duration or 0) / 1_000_000
    eval_dur_sec = (response.eval_duration or 1) / 1_000_000_000
    tokens_per_sec = round(tokens_out / eval_dur_sec, 1) if eval_dur_sec > 0 else 0

    return {
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
        "tokens_per_sec": tokens_per_sec,
    }

# =============================================================================
# STEP 2 - score a response against its expected value
# =============================================================================

def score_response(task, output_text):
    """
    Returns "correct", "partial", or "incorrect".
    Keeps scoring as a flat if/elif so the logic is easy to follow.
    """

    task_type = task["task"]
    expected  = task["expected"]
    text      = output_text.lower()

    # ---- json tasks ---------------------------------------------------------
    if task_type == "json":
        # Try to pull a JSON block out of the response (models often wrap it
        # in markdown fences like ```json ... ```)
        raw = output_text
        if "```" in raw:
            # grab what's between the first ``` pair
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            return "incorrect"

        # Count how many expected keys have the right value
        correct_keys = 0
        for key, val in expected.items():
            # compare as strings to handle int/string mismatches gracefully
            if key in parsed and str(parsed[key]).lower() == str(val).lower():
                correct_keys += 1

        if correct_keys == len(expected):
            return "correct"
        elif correct_keys > 0:
            return "partial"
        else:
            return "incorrect"

    # ---- coding tasks -------------------------------------------------------
    elif task_type == "coding":
        # expected is a list of strings that should appear in the output
        hits = sum(1 for kw in expected if kw.lower() in text)
        if hits == len(expected):
            return "correct"
        elif hits > 0:
            return "partial"
        else:
            return "incorrect"

    # ---- reasoning tasks ----------------------------------------------------
    elif task_type == "reasoning":
        # expected is a simple answer string
        if str(expected).lower() in text:
            return "correct"
        else:
            return "incorrect"

    # ---- knowledge tasks ----------------------------------------------------
    elif task_type == "knowledge":
        # special case: "unanswerable" means we want the model to refuse
        if expected == "unanswerable":
            refusal_phrases = [
                "cannot", "can't", "don't know", "not possible",
                "impossible", "no way to know", "estimate", "approximate",
                "no exact", "not known", "uncertain",
            ]
            if any(phrase in text for phrase in refusal_phrases):
                return "correct"
            else:
                return "incorrect"
        else:
            if str(expected).lower() in text:
                return "correct"
            else:
                return "incorrect"

    # ---- general text tasks -------------------------------------------------
    elif task_type == "text":
        if str(expected).lower() in text:
            return "correct"
        else:
            return "incorrect"

    # ---- fallback -----------------------------------------------------------
    else:
        return "incorrect"

# =============================================================================
# STEP 3 - run every task, print live progress, collect results
# =============================================================================

print(f"\n{'='*60}")
print(f"  Local LLM Evaluation")
print(f"  Model      : {MODEL}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Tasks      : {len(TASKS)}")
print(f"{'='*60}\n")

# Start ollama serve in background
print("Starting Ollama server...")
proc = subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)

# Wait for ollama to be ready before proceeding
max_retries = 10
for attempt in range(max_retries):
    try:
        ollama.list()
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

        result = run_prompt(task["prompt"])
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

    # =============================================================================
    # STEP 4 - print a summary table
    # =============================================================================

    print(f"\n{'='*60}")
    print(f"  Summary - {MODEL}")
    print(f"{'='*60}")

    score_counts = {"correct": 0, "partial": 0, "incorrect": 0}
    total_latency = 0
    total_tps = 0

    for r in results:
        score_counts[r["score"]] += 1
        total_latency += r["latency_ms"]
        total_tps     += r["tokens_per_sec"]

    n = len(results)
    print(f"  Correct  : {score_counts['correct']}/{n}")
    print(f"  Partial  : {score_counts['partial']}/{n}")
    print(f"  Incorrect: {score_counts['incorrect']}/{n}")
    print(f"  Avg latency  : {total_latency / n:.0f} ms")
    print(f"  Avg tok/sec  : {total_tps / n:.1f}")
    print(f"{'='*60}\n")

    # =============================================================================
    # STEP 5 - save results to JSON
    # =============================================================================

    output = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "summary": {
            "correct": score_counts["correct"],
            "partial": score_counts["partial"],
            "incorrect": score_counts["incorrect"],
            "avg_latency_ms": round(total_latency / n, 1),
            "avg_tokens_per_sec": round(total_tps / n, 1),
        },
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {RESULTS_FILE}")
    print("To compare models: change MODEL at the top of the script and re-run.\n")

finally:
    # Clean up ollama process and all children
    print("\nCleaning up Ollama server...")
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], timeout=3)
    except Exception:
        pass
