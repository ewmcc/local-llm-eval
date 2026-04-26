# local-llm-eval

Evaluating open-weight LLMs on a standard laptop.

OpenAI and Anthropic are great, but they come with trade-offs: API costs, rate limits, and every prompt you send leaving your machine. This project asks a simple question: **how well do small open models actually perform on a consumer laptop with no GPU?** We run structured evals across task categories using two different backends - [Ollama](https://ollama.com) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - score each response, and save results to JSON.

The final comparison is `gemma4:e2b` running on both backends side-by-side. The headline finding: **llama-cpp-python is roughly 1.6× faster than Ollama on the same model and hardware, making it a more suitable backend for local AI agent workflows** - though it requires meaningfully more setup work on Windows.

The models tested - `llama3.2:3b`, `ministral-3:3b`, `gemma4:e2b` - are all under 4B parameters and run entirely on CPU. No cloud. No billing. No data egress.

---

## Hardware

All results were produced on a standard **HP EliteBook** running Windows 11. No discrete GPU - inference is CPU-only.

| Spec | Detail |
|---|---|
| CPU | Intel Core i7-1185G7 @ ~3.0 GHz (4 cores) |
| RAM | 16 GB |
| OS | Windows 11 (x64) |
| GPU | None - CPU inference only |

Latency numbers throughout this README reflect this specific hardware. A machine with a discrete GPU would see significantly lower values.

---

## Setup: Ollama Workflow

Ollama is the simpler path. It handles model management and serves a local HTTP API.

**Prerequisites**

1. Download and install [Ollama for Windows](https://ollama.com/download/windows)
2. Pull the models:

```bash
ollama pull gemma4:e2b
ollama pull llama3.2:3b
ollama pull ministral-3:3b
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Running the eval**

```bash
python ollama-eval.py
```

**Code overview** - `ollama-eval.py` spins up `ollama serve` as a subprocess, waits until the server is responsive, then runs each task through `ChatOllama`:

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma4:e2b", temperature=0.3, base_url="http://localhost:11434")

response = llm.invoke([HumanMessage(content="Write a fibonacci function in Python.")])

# Exact token counts come from Ollama's usage_metadata
usage = response.usage_metadata  # {"input_tokens": 20, "output_tokens": 312, ...}
print(response.content)
```

Results are written to `output/ollama_results_<model>.json`.

---

## Setup: llama-cpp-python Workflow (Windows)

This path is more involved. `llama-cpp-python` gives you direct access to GGUF quantized models without a server daemon, but it compiles a native C++ extension at install time - which requires a C++ compiler on Windows that is not present by default.

### Step 1 - Download the GGUF model file

Go to [HuggingFace](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) and download a quantized GGUF. For this evaluation we used:

- **Model:** `google/gemma-4-E2B-it` (instruction-tuned, 2B effective parameters)
- **Quantization:** `Q4_K_M` (good balance of size and quality for CPU)
- **File:** `gemma-4-E2B-it-Q4_K_M.gguf`

Place the file anywhere accessible, e.g. `C:\models\`.

### Step 2 - Install Microsoft Visual Studio Build Tools ⚠️

`llama-cpp-python` must compile a C++ extension (`llama.cpp`) during `pip install`. Without a C++ compiler, you will get an error like:

```
error: command 'cl.exe' failed: No such file or directory
```
```
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
```

**Fix:** Install the free [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

During installation, on the **Workloads** screen, select **"Desktop development with C++"**. This installs `cl.exe`, CMake, and the Windows SDK - everything the build needs. The full installation is roughly 5–6 GB.

After installing Build Tools, **open a new terminal** (so the updated `PATH` is picked up) before continuing.

### Step 3 - Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `llama-cpp-python`, which will now compile successfully against your MSVC toolchain.

### Step 4 - Configure and run

Edit `MODEL_PATH` at the top of `llama-cpp-eval.py` to point at your GGUF file:

```python
MODEL_PATH = r"C:\models\gemma-4-E2B-it-Q4_K_M.gguf"
```

Then run:

```bash
python llama-cpp-eval.py
```

**Code overview** - `llama-cpp-eval.py` loads the model once at startup and runs each task directly, with no server:

```python
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage

llm = ChatLlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    n_ctx=4096,        # context window in tokens
    n_gpu_layers=-1,   # offload all layers to GPU if available; 0 = CPU-only
    max_tokens=2048,   # cap on output length
    verbose=False,
)

response = llm.invoke([HumanMessage(content="Write a fibonacci function in Python.")])
print(response.content)
```

Results are written to `output/llamacpp_results_<name>.json`.

---

## Eval Suite

12 tasks across 6 categories, 2 per category. Temperature is fixed at `0.3` for all runs.

| Category | Tasks | What's tested |
|---|---|---|
| `text` | 2 | Explanation quality, tone, following constraints |
| `coding` | 2 | Python function generation, SQL query construction |
| `json` | 2 | Structured extraction from unstructured text |
| `reasoning` | 2 | Math word problems, spatial logic |
| `knowledge` | 2 | Factual recall, handling unanswerable questions |
| `tool_use` | 2 | Multi-step tool selection and planning |

Scoring is keyword/heuristic-based via `eval_utils.py`: responses are labelled `correct`, `partial`, or `incorrect` based on whether expected keywords or phrases appear in the output. Latency is measured as clock time (`time.perf_counter()`) around each `invoke()` call.

---

## Results

### All-Model Summary

| Model | Backend | Correct | Partial | Incorrect | Avg Latency |
|---|---|---|---|---|---|
| `llama3.2:3b` | Ollama | 11/12 | 1/12 | 0/12 | **~22.2s** |
| `ministral-3:3b` | Ollama | **12/12** | 0/12 | 0/12 | ~35.8s |
| `gemma4:e2b` | Ollama | 11/12 | 0/12 | 1/12 | ~78.8s |
| `gemma4:e2b` Q4_K_M | llama-cpp | **12/12** | 0/12 | 0/12 | ~50.0s |

`llama3.2:3b` (Ollama) is the fastest model at **~22.2s avg latency**, though its 11/12 result includes one false positive (see Notable Observations). `gemma4:e2b` via llama-cpp runs at **~50.0s avg latency** - roughly 1.6× faster than the **~78.8s** it averages through Ollama - while achieving a perfect 12/12.

---

### gemma4:e2b - Backend Comparison

The primary focus of this evaluation is running the same model (`gemma4:e2b`) through two different LangChain backends on identical hardware.

| Metric | Ollama | llama-cpp (Q4_K_M) |
|---|---|---|
| Correct | 11/12 | **12/12** |
| Partial | 0/12 | 0/12 |
| Incorrect | 1/12 | 0/12 |
| Avg Latency | ~78.8s | **~50.0s** |
| Total wall-clock time | ~946s | **~600s** |

---

### Per-Task Breakdown - gemma4:e2b (Ollama)

| # | Category | Score | Latency | Tokens In | Tokens Out |
|---|---|---|---|---|---|
| 1 | text | ✅ correct | 42.7s | 38 | 356 |
| 2 | text | ✅ correct | 41.1s | 33 | 406 |
| 3 | coding | ✅ correct | 233.7s | 34 | 2,061 |
| 4 | coding | ✅ correct | 161.8s | 42 | 1,306 |
| 5 | json | ✅ correct | 37.7s | 66 | 328 |
| 6 | json | ✅ correct | 50.0s | 68 | 422 |
| 7 | reasoning | ✅ correct | 106.9s | 53 | 975 |
| 8 | reasoning | ✅ correct | 63.6s | 50 | 572 |
| 9 | knowledge | ❌ incorrect | 26.6s | 26 | 225 |
| 10 | knowledge | ✅ correct | 42.1s | 34 | 403 |
| 11 | tool_use | ✅ correct | 68.6s | 103 | 693 |
| 12 | tool_use | ✅ correct | 71.2s | 96 | 747 |

Token counts sourced from Ollama's `usage_metadata` (exact).

---

### Per-Task Breakdown - gemma4:e2b (llama-cpp, Q4_K_M)

| # | Category | Score | Latency | Tokens In\* | Tokens Out\* |
|---|---|---|---|---|---|
| 1 | text | ✅ correct | 20.1s | 17 | 55 |
| 2 | text | ✅ correct | 16.6s | 16 | 66 |
| 3 | coding | ✅ correct | 127.9s | 15 | 209 |
| 4 | coding | ✅ correct | 106.8s | 20 | 309 |
| 5 | json | ✅ correct | 10.7s | 23 | 12 |
| 6 | json | ✅ correct | 10.3s | 25 | 13 |
| 7 | reasoning | ✅ correct | 145.6s | 25 | 353 |
| 8 | reasoning | ✅ correct | 83.2s | 28 | 107 |
| 9 | knowledge | ✅ correct | 2.3s | 9 | 9 |
| 10 | knowledge | ✅ correct | 11.0s | 16 | 64 |
| 11 | tool_use | ✅ correct | 31.4s | 59 | 89 |
| 12 | tool_use | ✅ correct | 34.2s | 48 | 144 |

\* Token counts are **word-split estimates**, not exact. See the analysis section below.

---

## Analysis

### Why We Focus on Latency, Not Tokens/Sec

**Ollama** provides exact token counts. `ChatOllama` reads `prompt_eval_count` and `eval_count` directly from Ollama's response metadata and populates `response.usage_metadata` with real subword token counts:

```python
# From langchain_ollama/chat_models.py
def _get_usage_metadata_from_generation_info(generation_info):
    input_tokens  = generation_info.get("prompt_eval_count")   # exact
    output_tokens = generation_info.get("eval_count")          # exact
    return UsageMetadata(input_tokens=input_tokens, output_tokens=output_tokens, ...)
```

**llama-cpp-python** does not. `ChatLlamaCpp` (from `langchain-community`) does **not** populate `response.usage_metadata`. There is no hook in the `ChatLlamaCpp` implementation that surfaces the internal token counts from the underlying `llama.cpp` runtime. As a result, our `llama-cpp-eval.py` falls back to a word-split approximation:

```python
# llama-cpp-eval.py - word-split fallback (not exact subword tokens)
tokens_out = len(text.split())
tokens_per_sec = round(tokens_out / elapsed_sec, 1)
```

This systematically **undercounts** tokens. Subword tokenisers (BPE/SentencePiece) split words like `"fibonacci"`, `"entanglement"`, or `"SELECT"` into multiple tokens. A word-split count might return 200 for a response that is actually 300–400 real tokens. Dividing elapsed seconds by an undercounted output produces an artificially low tok/s figure.

The reported 2.8 tok/s for llama-cpp is therefore **not comparable** to Ollama's exact 9.1 tok/s. Clock latency (seconds per task) is the only fair cross-backend metric, and it is what all conclusions below are based on.

---

### What Drove the Latency Difference?

The ~1.6× latency gap between Ollama (~78.8s avg) and llama-cpp (~50.0s avg) is almost entirely explained by **output verbosity**, not raw inference speed.

**Coding tasks are the clearest example:**

| Task | Ollama output tokens | llama-cpp output tokens | Ollama latency | llama-cpp latency |
|---|---|---|---|---|
| Task 3 (Python fibonacci) | 2,061 | 209 | 233.7s | 127.9s |
| Task 4 (SQL query) | 1,306 | 309 | 161.8s | 106.8s |

Both backends produced a correct answer for both tasks. The difference is that Ollama's `gemma4:e2b` generated full docstrings, type annotations, multiple usage examples, alternative implementations, and explanatory prose. The llama-cpp version returned a concise, working solution.

---

### Accuracy

The single difference in accuracy (llama-cpp 12/12, Ollama 11/12) is task 9 - knowledge:

> *"What year was the Python programming language first released?"*

Ollama's `gemma4:e2b` answered 1994. The correct answer is 1991. The llama-cpp run answered correctly.

This is a factual error by the model, not a scorer issue.

---

### llama-cpp as a Local Agent Backend

For AI agent workflows - where many sequential LLM calls are chained together - the llama-cpp backend has several practical advantages over Ollama:

- **Lower per-call latency.** At ~50.0s avg vs ~78.8s avg on this hardware, a 10-step agent loop completes in ~8 minutes vs ~13 minutes. The gap compounds with chain length.
- **Concise outputs.** Verbose responses bloat the context window in multi-turn chains. llama-cpp's tendency to give tighter answers is directly useful for agents that pass LLM output downstream.
- **No daemon overhead.** llama-cpp loads the model directly in-process. There is no HTTP round-trip, no Ollama server to manage, and no process to clean up. Startup adds a few seconds once (model load), then subsequent calls are pure inference.
- **Direct GGUF access.** You control exactly which quantization you load (`Q4_K_M`, `Q8_0`, `F16`) and can compare variants without re-pulling from a registry.

The tradeoff is setup complexity, particularly on Windows. The MSVC Build Tools requirement is a real friction point - it adds ~30 minutes of setup and 5–6 GB of disk space before you can even install the package. Ollama, by contrast, is a single installer with a one-command model pull.

---

## Notable Observations (All Models)

**Latency is driven by verbosity, not raw speed**

`gemma4:e2b` via Ollama runs at ~9.1 tok/s (exact, from Ollama metadata) but averages 78.8s per task because it generates far more tokens per response. The same dynamic appears in the llama-cpp vs Ollama comparison: same model, same hardware, but output length varies dramatically by backend.

**`llama3.2:3b` scored `correct` on task 7 despite a wrong answer**

For the "5 machines, 5 widgets in 5 minutes" problem, `llama3.2:3b` gave an incorrect answer, but the expected string `"5 minutes"` appeared elsewhere in the response and the scorer returned `correct`. This false positive is counted in its 11/12 result. Task 8 (room/windows spatial reasoning) scores `partial` — the model calculated 12 − 1 = 11 instead of the correct 9.

**`gemma4:e2b` (Ollama) scored `incorrect` on the Python release year**

Task 9 (knowledge): the model answered 1994 instead of 1991. This is the single incorrect score in the Ollama `gemma4:e2b` run.

**Scorer updates resolved prior false negatives**

The refusal-phrase list was expanded to include `"do not have access"`, `"real-time"`, and related paraphrases, fixing false `incorrect` scores on the South Pole temperature task for both `gemma4:e2b` and `ministral-3:3b`. The product review scorer was also updated to accept `"game-changer"` alongside `"excellent"`, resolving a prior `partial` for `ministral-3:3b`. Both models now score those tasks `correct`.

**`ministral-3:3b` ingests significantly more input tokens**

Input token counts for `ministral-3:3b` are ~570–638 per call, compared to 30–103 for the other models on identical prompts. This reflects a large system prompt baked into the model's Ollama packaging.

---

## Key Takeaways

- **llama-cpp is the better backend for local agent workloads.** ~1.6× faster latency per call, more concise outputs, no server overhead. The verbosity gap is the key driver - not raw inference speed.
- **The Windows setup for llama-cpp is non-trivial.** The MSVC Build Tools requirement will block installation without prior knowledge.
- **Tok/s is not a reliable cross-backend metric here.** Ollama reports exact subword counts; `ChatLlamaCpp` does not expose `usage_metadata`. All throughput comparisons in this project use wall-clock latency.
- **`llama3.2:3b` is the fastest model tested at ~22.2s avg latency**, though its 11/12 result includes a false positive on task 7 (keyword scorer matched `"5 minutes"` in an incorrect response).
- **Keyword scoring is a useful baseline but it lies sometimes.** It scored a wrong answer `correct` (task 7) and missed valid refusals until the phrase list was expanded.

---

## Limitations

- **No GPU.** All latencies are CPU-bound on an Intel i7-1185G7. A discrete GPU or Apple Silicon would reduce these by 5–10×.
- **Single-run, no variance.** Each task ran once. Results don't account for stochastic variation across runs.
- **Keyword-based scoring.** Detects presence of expected strings - does not verify logical correctness, accuracy, or response quality holistically.
- **Token counts for llama-cpp are approximate.** Word-split estimation, not real BPE token counts. Do not compare tok/s figures between backends.
- **Small model tier only.** All models are sub-4B parameters. Larger models (7B, 13B+) need more RAM and time on this hardware.
