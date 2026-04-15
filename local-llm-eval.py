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
#   2. ollama pull ### (replace with local model name)
#   3. python local-llm-eval.py
#
# TO SWAP MODELS:
#   Edit the MODEL variable below and re-run.
#   Compare the resulting JSON files to see performance differences.
# =============================================================================

import json
import ollama
import subprocess
import time

# =============================================================================
# CONFIG - edit these values to change behaviour
# =============================================================================

MODEL = "gemma4:e2b"        # swap to compare
TEMPERATURE = 0.3           # lower = more deterministic; try 0.0 or 0.7 too

# Dynamically generate results filename based on model name
# Sanitize model name by replacing colons and spaces with underscores
MODEL_SAFE_NAME = MODEL.replace(":", "_").replace(" ", "_").lower()
RESULTS_FILE = f"results_{MODEL_SAFE_NAME}.json" # where to write the final output

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
        "prompt": "Explain quantum entanglement in a way a high school student could understand. Keep it to 2-3 sentences.",
        "expected": "particles",
    },
    {
        "task": "text",
        "prompt": "Write a concise product review for a hypothetical coffee maker that is excellent but slightly loud.",
        "expected": ["excellent", "loud"],
    },

    # --- coding --------------------------------------------------------------
    {
        "task": "coding",
        "prompt": "Write a Python function called fibonacci that returns the nth Fibonacci number. Include input validation.",
        "expected": ["def fibonacci", "return", "if"],
    },
    {
        "task": "coding",
        "prompt": "Write a SQL query that finds customers who made purchases in the last 30 days and spent over $100 total.",
        "expected": ["SELECT", "WHERE", "30"],
    },

    # --- structured output ---------------------------------------------------
    {
        "task": "json",
        "prompt": "Extract contact info from this text into JSON with keys 'name', 'email', and 'phone':\n\n'Contact John Smith at john.smith@company.com or call (555) 123-4567.'",
        "expected": {"name": "John Smith", "email": "john.smith@company.com", "phone": "(555) 123-4567"},
    },
    {
        "task": "json",
        "prompt": "Convert this product listing into JSON with keys 'product', 'price', 'in_stock' (boolean), and 'rating' (number):\n\n'MacBook Pro - $1299, currently in stock, rated 4.8 stars.'",
        "expected": {"product": "MacBook Pro", "price": 1299, "in_stock": True, "rating": 4.8},
    },

    # --- reasoning -----------------------------------------------------------
    {
        "task": "reasoning",
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain your reasoning.",
        "expected": "5",
    },
    {
        "task": "reasoning",
        "prompt": "A room has 4 walls. Each wall has 3 windows. How many windows are visible from inside the room if one wall is completely blocked? Show your work.",
        "expected": ["9", "3"],
    },

    # --- knowledge / factual -------------------------------------------------
    {
        "task": "knowledge",
        "prompt": "What year was the Python programming language first released?",
        "expected": "1991",
    },
    {
        "task": "knowledge",
        "prompt": "What is the exact current temperature at the South Pole right now? Give a precise answer.",
        # Good behaviour: the model should decline or express uncertainty,
        # NOT fabricate a specific number.  The scorer checks for refusal language.
        "expected": "unanswerable",
    },

    # --- tool use / planning -------------------------------------------------
    {
        "task": "tool_use",
        "prompt": """You have access to these tools:
- calculator(expression): Evaluates a math expression and returns the result
- weather(location): Returns current temperature for a location
- search(query): Searches the web and returns relevant results

A customer bought 3 items at $12.50 each, with 15% tax. How much did they pay total? Show which tool you would use and the calculation.""",
        "expected": ["calculator", "12.50", "3", "15"],
    },
    {
        "task": "tool_use",
        "prompt": """You have access to these tools:
- get_stock_price(symbol): Returns current stock price
- currency_convert(amount, from, to): Converts currency
- fetch_news(topic): Gets latest news articles

If Apple stock is currently $150 per share and you want to convert $1500 worth to British pounds, what tools would you use and in what order?""",
        "expected": ["get_stock_price", "currency_convert"],
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
        if isinstance(expected, list):
            hits = sum(1 for kw in expected if str(kw).lower() in text)
            if hits == len(expected):
                return "correct"
            elif hits > 0:
                return "partial"
            else:
                return "incorrect"
        else:
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
        if isinstance(expected, list):
            hits = sum(1 for kw in expected if kw.lower() in text)
            if hits == len(expected):
                return "correct"
            elif hits > 0:
                return "partial"
            else:
                return "incorrect"
        else:
            if str(expected).lower() in text:
                return "correct"
            else:
                return "incorrect"

    # ---- tool use tasks -----------------------------------------------------
    elif task_type == "tool_use":
        # expected is a list of tool names / key values that should appear in the output
        hits = sum(1 for kw in expected if kw.lower() in text)
        if hits == len(expected):
            return "correct"
        elif hits > 0:
            return "partial"
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
    print("To compare models: change MODEL at the top of the script and re-run.")
    print("Each model's results are saved to a separate file for easy comparison.\n")

finally:
    # Clean up ollama process and all children
    print("\nCleaning up Ollama server...")
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], timeout=3)
    except Exception:
        pass
