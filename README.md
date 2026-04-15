# local-llm-eval

> Evaluating open-weight LLMs on a standard laptop - locally, privately, for free.

OpenAI and Anthropic are great, but they come with trade-offs: API costs, rate limits, and every prompt you send leaving your machine. This project asks a simpler question: **how well do small open models actually perform on a consumer laptop with no GPU?** We run structured evals across task categories using [Ollama](https://ollama.com), score each response, and save the results to JSON so comparisons are straightforward.

The models here - `llama3.2:3b`, `ministral-3:3b`, `gemma4:e2b` - are all under 4B parameters and run entirely on CPU. No cloud. No billing. No data egress.

---

## Hardware

All results were produced on a standard **HP EliteBook** running Windows 11. No discrete GPU - inference is CPU-only.

| Spec | Detail |
|---|---|
| CPU | Intel Core i7-1185G7 @ ~3.0 GHz (4 cores) |
| RAM | 16 GB |
| OS | Windows 11 (x64) |
| GPU | None - CPU inference only |
| Ollama | Local server, default settings |

Latency numbers throughout this README reflect this specific hardware. A machine with a discrete GPU would see significantly lower values.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the model you want to test
ollama pull llama3.2:3b
ollama pull ministral-3:3b
ollama pull gemma4:e2b

# 3. Set your model and run
#    Edit MODEL at the top of local-llm-eval.py, then:
python local-llm-eval.py
```

Results are written automatically to `results_<model>.json`. To compare models, change the `MODEL` variable and re-run.

**Example terminal output:**

```
============================================================
  Local LLM Evaluation
  Model      : gemma4:e2b
  Temperature: 0.3
  Tasks      : 12
============================================================

Starting Ollama server...
Ollama server is ready!

[01/12] Starting text             task...
    Prompt: Explain quantum entanglement in a way a high school student could unde...
    Running inference... Done!
    Result: correct    |    95272 ms  |    6.4 tok/s
    Tokens: 38 in, 341 out

[02/12] Starting text             task...
    Prompt: Write a concise product review for a hypothetical coffee maker that is...
    Running inference... Done!
    Result: correct    |    56365 ms  |    7.9 tok/s
    Tokens: 33 in, 371 out

...

============================================================
  Summary - gemma4:e2b
============================================================
  Correct  : 10/12
  Partial  : 1/12
  Incorrect: 1/12
  Avg latency  : 72275 ms
  Avg tok/sec  : 10.9
============================================================

Results saved to results_gemma4_e2b.json
```

---

## Eval Suite

12 tasks across 6 categories, 2 per category. Temperature is fixed at `0.3` for all runs (deterministic-leaning, but not fully greedy).

| Category | Tasks | What's tested |
|---|---|---|
| `text` | 2 | Explanation quality, tone, following constraints |
| `coding` | 2 | Python function generation, SQL query construction |
| `json` | 2 | Structured extraction from unstructured text |
| `reasoning` | 2 | Math word problems, spatial logic |
| `knowledge` | 2 | Factual recall, handling unanswerable questions |
| `tool_use` | 2 | Multi-step tool selection and planning |

Scoring is keyword/heuristic-based: responses are labeled `correct`, `partial`, or `incorrect` depending on whether expected values or phrases appear in the output. This approach is intentionally simple - fast to run, easy to audit - but as the results show, it introduces its own blind spots.

---

## Results

### Summary

| Model | Correct | Partial | Incorrect | Avg Latency | Avg Tok/s |
|---|---|---|---|---|---|
| `llama3.2:3b` | **11/12** | 1/12 | 0/12 | **~19.6s** | 10.0 |
| `ministral-3:3b` | **11/12** | 1/12 | 0/12 | ~29.6s | 7.3 |
| `gemma4:e2b` | 10/12 | 1/12 | 1/12 | ~72.3s | 10.9 |

`llama3.2:3b` is the winner in this study: tied for the best accuracy score, fastest by a wide margin, and the most consistently terse output.

---

### Task-by-Task Breakdown

| # | Category | llama3.2:3b | ministral-3:3b | gemma4:e2b |
|---|---|---|---|---|
| 1 | text | ✅ | ✅ | ✅ |
| 2 | text | 🟡 partial | 🟡 partial | ✅ |
| 3 | coding | ✅ | ✅ | ✅ |
| 4 | coding | ✅ | ✅ | ✅ |
| 5 | json | ✅ | ✅ | ✅ |
| 6 | json | ✅ | ✅ | ✅ |
| 7 | reasoning | ✅ | ✅ | ✅ |
| 8 | reasoning | ✅ | ✅ | ✅ |
| 9 | knowledge | ✅ | ✅ | ✅ |
| 10 | knowledge | ✅ | ✅ | ❌ |
| 11 | tool_use | ✅ | ✅ | ✅ |
| 12 | tool_use | ✅ | ✅ | 🟡 partial |

---

### Notable Observations

**Latency is driven by verbosity, not speed**

`gemma4:e2b` runs at a comparable token rate (~10.9 tok/s) to the others - it's just much more verbose. Coding tasks ballooned to 1,500–1,770 output tokens (full docstrings, type hints, multiple usage examples, cross-database SQL variations) compared to ~200 tokens for the same prompt on `llama3.2:3b`. That 3.7× latency gap is almost entirely output length, not inference speed.

**`llama3.2:3b` got the reasoning wrong, but scored correct**

Task 7 - the classic "5 machines, 5 widgets" problem - exposed a scorer limitation. `llama3.2:3b` answered **100 minutes** (incorrect), but the expected keyword was `"5"` and `"5"` appeared elsewhere in the response. It scored `correct`. `gemma4:e2b` and `ministral-3:3b` both answered 5 minutes correctly. Keyword-based scoring can reward the wrong response if it doesn't check context.

**`gemma4:e2b` gave a valid refusal, but still failed**

Task 10 asks for the exact current temperature at the South Pole - a clearly unanswerable question. `gemma4:e2b` responded: *"I do not have access to real-time, live data feeds..."* That's the right behavior. But the scorer's refusal-phrase list only checked for words like `cannot`, `can't`, `don't know`, etc. - none of which appeared. A legitimate, well-phrased refusal was scored `incorrect`. The model was right; the eval was wrong.

**`ministral-3:3b` ingests significantly more input tokens**

Input token counts for `ministral-3:3b` are ~600 per request, compared to ~30–100 for the other two models on identical prompts. This points to a large system prompt baked into the model's Ollama packaging. It has no impact on output quality here, but it's worth noting if you're optimizing context window usage or costs in a production setting.

**The product review partial is a scorer blind spot, not a model failure**

All three models received a `partial` on task 2 (coffee maker review) because none used the exact word *"excellent"*. `llama3.2:3b` wrote "exceeded my expectations," `ministral-3:3b` wrote "game-changer," `gemma4:e2b` correctly used "excellent" and scored `correct`. Two out of three models wrote a genuinely good review - the scorer just couldn't tell.

---

## Key Takeaways

- **`llama3.2:3b` is the practical choice for CPU-only workflows.** Fast, accurate, concise. Gets out of the way.
- **Local inference is viable for structured tasks without a GPU.** JSON extraction, coding, factual QA, and tool planning all work well at 10 tok/s on a 3.0 GHz laptop CPU.
- **Keyword scoring is a useful baseline, but it lies sometimes.** Good evals include human review of partials and edge cases - the numbers alone miss real failures and false negatives.

---

## Limitations

- **No GPU.** All latencies are CPU-bound. A dedicated GPU or Apple Silicon would reduce these by 5–10×.
- **Single-run, no variance.** Each task ran once. Results don't account for stochastic variation across runs.
- **Keyword-based scoring.** The scorer detects presence of expected strings - it doesn't verify accuracy, logical correctness, or response quality holistically.
- **Small model tier only.** These are all sub-4B parameter models. Larger models (7B, 13B+) would need more RAM and significantly more time per task on this hardware.
