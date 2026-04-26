# =============================================================================
# EVAL UTILS - shared components used by both ollama-eval.py and llama-cpp-eval.py
#
# Exports:
#   TASKS              - list of 12 evaluation tasks across 6 categories
#   score_response()   - scores a model response as "correct", "partial", or "incorrect"
#   print_summary()    - prints a formatted summary table to stdout
#   save_results()     - serialises results to a JSON file
# =============================================================================

import json
import os

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
        # Inner lists are OR-groups: any one word in a group satisfies that slot.
        # Both slots must be satisfied for "correct".
        # Slot 1: positive quality words  |  Slot 2: noise words
        "expected": [
            ["excellent", "great", "amazing", "fantastic", "exceptional",
             "outstanding", "superb", "perfect", "game-changer", "game changer",
             "impressive", "wonderful", "brilliant"],
            ["loud", "noisy", "noise", "sound"],
        ],
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
        # "5" appears throughout any response (prompt echo, wrong answers like "100 minutes").
        # "5 minutes" only appears in a correct answer that identifies the invariant.
        "expected": "5 minutes",
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

You have 10 shares of Apple stock want to convert $1500 worth to British pounds, what tools would you use and in what order?""",
        "expected": ["get_stock_price", "currency_convert"],
    },

]


# =============================================================================
# SCORER
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
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            return "incorrect"

        correct_keys = 0
        for key, val in expected.items():
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
        if expected == "unanswerable":
            refusal_phrases = [
                "cannot", "can't", "don't know", "not possible",
                "impossible", "no way to know", "estimate", "approximate",
                "no exact", "not known", "uncertain", "do not have",
                "don't have", "no access", "not available", "i cannot access",
                "don't have access", "unable to", "lack", "no data",
                # Broader paraphrases observed in practice (e.g. ministral-3:3b)
                "real-time", "real time", "live data", "live weather",
                "current conditions", "not able to", "i'm not able",
                "i am not able", "no real-time", "no real time",
                "don't have access to real", "access to current",
                "access to live", "access to real-time",
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
            # Support nested OR-groups: inner list = any one match satisfies the slot.
            # A plain string item is treated as a single-item OR-group.
            def _slot_hit(slot, t):
                if isinstance(slot, list):
                    return any(s.lower() in t for s in slot)
                return str(slot).lower() in t

            hits = sum(1 for slot in expected if _slot_hit(slot, text))
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
# OUTPUT HELPERS
# =============================================================================

def print_summary(model_name, results):
    """Print a formatted summary table to stdout."""

    score_counts = {"correct": 0, "partial": 0, "incorrect": 0}
    total_latency = 0.0
    total_tps = 0.0

    for r in results:
        score_counts[r["score"]] += 1
        total_latency += r["latency_ms"]
        total_tps     += r["tokens_per_sec"]

    n = len(results)
    print(f"\n{'='*60}")
    print(f"  Summary - {model_name}")
    print(f"{'='*60}")
    print(f"  Correct  : {score_counts['correct']}/{n}")
    print(f"  Partial  : {score_counts['partial']}/{n}")
    print(f"  Incorrect: {score_counts['incorrect']}/{n}")
    print(f"  Avg latency  : {total_latency / n:.0f} ms")
    print(f"  Avg tok/sec  : {total_tps / n:.1f}")
    print(f"{'='*60}\n")

    return score_counts, total_latency, total_tps


def save_results(model_name, temperature, results, filename, script_type=None):
    """Serialise results list to a JSON file in the output directory.
    
    Args:
        model_name: Model name to include in results
        temperature: Temperature setting to include in results
        results: List of result dictionaries
        filename: Filename for the output (without path or script_type prefix)
        script_type: Optional script type ("ollama" or "llamacpp") to prepend to filename
    """

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct filename with script_type prefix if provided
    if script_type:
        output_filename = f"{script_type}_{filename}"
    else:
        output_filename = filename
    
    filepath = os.path.join(output_dir, output_filename)

    score_counts = {"correct": 0, "partial": 0, "incorrect": 0}
    total_latency = 0.0
    total_tps = 0.0

    for r in results:
        score_counts[r["score"]] += 1
        total_latency += r["latency_ms"]
        total_tps     += r["tokens_per_sec"]

    n = len(results)
    output = {
        "model": model_name,
        "temperature": temperature,
        "summary": {
            "correct": score_counts["correct"],
            "partial": score_counts["partial"],
            "incorrect": score_counts["incorrect"],
            "avg_latency_ms": round(total_latency / n, 1),
            "avg_tokens_per_sec": round(total_tps / n, 1),
        },
        "results": results,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filepath}")
