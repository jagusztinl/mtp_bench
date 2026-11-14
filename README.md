# MTP / Speculative Decoding Benchmark

`mtp_bench.py` is a Python benchmark tool for OpenAI-compatible inference servers, including **vLLM**, **llama.cpp**, **SGLang**, **Ollama**, and **oMLX**. It can test local servers or compatible remote endpoints that accept Bearer API-key authentication.

It measures:

- output generation throughput;
- prompt/prefill throughput estimates;
- TTFT (time to first token);
- wall-clock benchmark throughput;
- speculative decoding / multi-token prediction (MTP) statistics, when exposed;
- vLLM speculative decoding Prometheus counters;
- llama.cpp Prometheus metrics (best effort);
- SGLang Prometheus speculative-decoding gauge snapshots and counter deltas;
- parallel request throughput.

The script uses the OpenAI-compatible **`/v1/chat/completions`** streaming endpoint. The `--backend` flag labels the server type but does **not** automatically switch protocols, enable speculative decoding, or turn on metrics scraping.

> **Version and naming:** This README documents the updated `mtp_bench_v2.py` implementation. For the commands below, save or rename it to `mtp_bench.py`, or replace `mtp_bench.py` with `mtp_bench_v2.py` in every example. Do not use the older, unmodified script with the new options.

---

## Features

### Core benchmark

The script sends a fixed set of nine benchmark prompts to a server and records per-request and aggregate metrics.

It supports:

- streaming chat completions;
- client-observed TTFT and generation timing;
- configurable model name, maximum output tokens, prompt padding, and parallelism;
- optional random seed and sampling temperature;
- authenticated model listing and inference requests;
- optional, separately authenticated Prometheus scraping;
- JSON output and result comparison using `--diff`;
- server-specific response timing fields, when available;
- graceful handling of missing token usage and MTP statistics.

### API-key authentication

Authentication is optional. When configured, the script sends:

```http
Authorization: Bearer YOUR_API_KEY
```

for both `/v1/models` and `/v1/chat/completions`, including streaming retries.

Supported ways to supply the key, in precedence order:

1. `--api-key YOUR_API_KEY`;
2. the environment variable named by `--api-key-env` (default `MTP_BENCH_API_KEY`);
3. no authentication.

The script does **not** automatically use `OPENAI_API_KEY`, `OLLAMA_API_KEY`, or any other provider-specific variable unless selected with `--api-key-env`.

The `/metrics` endpoint does not automatically receive the inference key. It can use a **separate** metrics key, or it can explicitly reuse the inference key via `--metrics-use-api-key`.

The JSON configuration records only `api_key_configured` and `metrics_api_key_configured` booleans, not the key values. However, avoid putting credentials in URLs, shell history, saved command lines, or generated text.

### Prompt processing speed estimate

The script estimates prompt processing speed as:

```text
prompt_tokens / TTFT
```

This is a **client-side approximation**, not direct device prefill throughput. It is only treated as a meaningful, lower-biased estimate for interpretation at:

```text
--parallel 1
```

At `--parallel > 1`, TTFT includes queueing and scheduling delay. Per-request prompt tok/s is marked with `*`, and `aggregate_prompt_tokens_per_second_estimate` is `null`.

The script also reports:

```text
wall_clock_prompt_tokens_per_second
```

for concurrent tests. This is a useful suite-level **completed-input-token rate**, but it is not pure GPU/CPU prompt-processing throughput because the denominator covers the entire suite, including decoding and idle intervals.

### MTP / speculative decoding stats

The script retains these fields:

```text
predicted_n
predicted_per_second
draft_n
draft_n_accepted
accept_rate
```

and aggregate fields:

```text
total_predicted
total_draft
total_draft_accepted
aggregate_accept_rate
```

`predicted_n` may fall back to the completion-token count. **Missing draft/accepted-token statistics are represented as JSON `null` (displayed as `n/a`), not as measured zeros.** This distinction matters when comparing different engines.

For vLLM, Prometheus counters are generally the better option for inspecting speculative decoding activity. SGLang uses a different metric model, with speculative-decoding **gauges** rather than the same cumulative counters.

### Prometheus metrics support

The script can scrape `/metrics` immediately before and after a benchmark run.

Available metrics backends:

```text
none
vllm
llamacpp
sglang
generic
```

- **`vllm`**: selected speculative-decoding counter deltas, accepted/draft-token ratio, and selected general metrics.
- **`llamacpp`**: best-effort token/request counter detection and interesting metric deltas.
- **`sglang`**: before/after snapshots of known `sglang:spec_*` gauges, plus deltas for `sglang:*` metrics with counter-like suffixes.
- **`generic`**: differences for all available metric names. **Do not interpret all generic differences as counter increments**: the endpoint may expose gauges whose difference is not a meaningful event count.
- **`none`**: no metrics scraping; default and recommended for Ollama/oMLX unless a compatible Prometheus endpoint is independently provided.

Histogram `*_bucket` series are excluded. Other labelled time series are aggregated by metric name, so label-specific detail is not preserved. Prometheus data is server-global and can include unrelated traffic.

---

## Requirements

Python **3.9+** is recommended for the updated script.

Only Python standard-library modules are used:

```text
argparse
concurrent.futures
json
math
os
sys
time
urllib
```

No third-party Python package is required.

The target must expose an OpenAI-compatible `/v1/chat/completions` endpoint with streaming support. `/v1/models` is required for `--list-models`, but is not needed if a valid model ID is supplied directly. Prometheus endpoints are optional.

---

## Supported servers

| Server | OpenAI-compatible streaming | `/v1/models` | Specific speculative-decoding telemetry |
| --- | --- | --- | --- |
| **vLLM** | Yes | Typically | `--metrics-backend vllm`, when the relevant metrics exist |
| **llama.cpp** | Yes, via `llama-server` | Typically | Response `timings`/draft fields when exposed; optional best-effort Prometheus |
| **SGLang** | Yes | Typically | `--metrics-backend sglang`, for `sglang:spec_*` gauge snapshots |
| **Ollama** | Yes, via the OpenAI-compatible API | Typically | No guaranteed draft/acceptance fields through this interface |
| **oMLX** | Yes, for versions exposing the OpenAI API | Version-dependent | Response usage/timings when exposed; no built-in oMLX-specific metrics parser |

The compatibility table describes the **API shape the script expects**, not a guarantee that every release or configuration exposes identical streaming, token usage, seed handling, or speculative decoding fields. Check your installed server version.

### vLLM

Expected endpoints:

```text
/v1/models
/v1/chat/completions
/metrics                   (optional)
```

Example URL:

```text
http://127.0.0.1:8000
```

Use `--backend vllm` to identify the server in the JSON output. Use `--metrics-backend vllm` or `--prometheus` separately to scrape vLLM metrics. The vLLM server must expose the appropriate metrics for detailed speculative decoding analysis.

### llama.cpp

Use an OpenAI-compatible `llama-server` instance. For example, enable its metrics endpoint when supported:

```bash
./llama-server \
  -m model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics
```

Example URL:

```text
http://127.0.0.1:8080
```

Older builds may reject `stream_options.include_usage`. The script retries **once** without that field if the HTTP error explicitly indicates unsupported `stream_options` (HTTP 400, 422, or 500). You may instead disable the field upfront:

```bash
--no-stream-usage
```

The script can fall back to `timings.prompt_n` and `timings.predicted_n` for token counts when `usage` is absent.

### SGLang

SGLang provides an OpenAI-compatible chat-completions API. Its commonly used API port is `30000`.

Example URL:

```text
http://127.0.0.1:30000
```

To scrape server metrics, configure SGLang to expose metrics (for supported versions, for example via `--enable-metrics`) and use:

```bash
--backend sglang --metrics-backend sglang
```

SGLang speculative decoding fields such as `sglang:spec_accept_rate` and `sglang:spec_accept_length` are treated as **gauge snapshots**. The script does **not** subtract them to estimate per-benchmark acceptance, and does not turn them into a run-wide weighted average.

### Ollama

Ollama's local OpenAI-compatible API typically uses:

```text
http://127.0.0.1:11434/v1
```

The script accepts the `/v1` suffix and normalizes it before adding endpoint paths. Local Ollama commonly does not require an API key. If you use a compatible remote or cloud endpoint that requires authentication, set the key explicitly.

Ollama's native API and timing information are **not** automatically queried by this script. Do not assume OpenAI-compatible streaming includes draft/accepted-token counts or a Prometheus endpoint.

### oMLX

For an oMLX installation exposing OpenAI-compatible endpoints, supply its actual API port, for example:

```text
http://127.0.0.1:8000
```

Use `--backend omlx`. The script reads standard OpenAI-style streaming and any compatible usage/timings fields present. It does **not** call version-specific oMLX admin or stats endpoints, and it does not assume a Prometheus endpoint is available.

---

## Installation

Save the updated script as `mtp_bench.py` (or retain the actual name `mtp_bench_v2.py` and use that in all commands).

```bash
chmod +x mtp_bench.py
python mtp_bench.py --help
```

If the script is executed **without any command-line arguments**, it displays the same help text as `--help` and exits successfully without starting a benchmark or connecting to the server:

```bash
python mtp_bench.py
```

Verify the Python syntax:

```bash
python -m py_compile mtp_bench.py
```

---

## Authentication and API-key configuration

### Recommended: environment variable

Avoid passing secrets directly on the command line, where they may appear in shell history and process listings.

**Bash / Linux / macOS:**

```bash
export MTP_BENCH_API_KEY='your-secret-api-key'
python mtp_bench.py \
  --url https://your-inference-host.example \
  --list-models
```

**PowerShell / Windows:**

```powershell
$env:MTP_BENCH_API_KEY = 'your-secret-api-key'
python .\mtp_bench.py `
  --url https://your-inference-host.example `
  --list-models
```

The key is sent as `Authorization: Bearer ...` to `/v1/models` and `/v1/chat/completions`.

### Use a named environment variable

If credentials are already present under a different name, select it explicitly:

```bash
python mtp_bench.py \
  --url https://your-inference-host.example \
  --api-key-env OPENAI_API_KEY \
  --list-models
```

For a compatible Ollama cloud endpoint, the same pattern can be used with `OLLAMA_API_KEY`:

```bash
python mtp_bench.py \
  --backend ollama \
  --url https://ollama.com/v1 \
  --api-key-env OLLAMA_API_KEY \
  --list-models
```

Do not send a provider's credentials to an endpoint that you do not trust.

### Explicit API key (less secure)

```bash
python mtp_bench.py \
  --url https://your-inference-host.example \
  --api-key 'your-secret-api-key' \
  --list-models
```

`--api-key` overrides the configured environment variable. A literal value may be captured in shell history or process inspection; prefer an environment variable.

### Separate authentication for Prometheus

Inference credentials **are not sent to `/metrics` by default**.

For a metrics endpoint protected with a different Bearer token:

```bash
export MTP_BENCH_METRICS_API_KEY='your-metrics-api-key'
python mtp_bench.py \
  --backend sglang \
  --url http://127.0.0.1:30000 \
  --model MODEL_ID \
  --metrics-backend sglang \
  --out sglang-metrics.json
```

Alternatively, specify `--metrics-api-key VALUE` directly. If, and **only if**, the metrics endpoint uses the same credential as inference, specify:

```bash
--metrics-use-api-key
```

Precedence: `--metrics-api-key` / `MTP_BENCH_METRICS_API_KEY` takes priority over `--metrics-use-api-key`.

### Credential safety

- API-key values are not written to the benchmark JSON. Only the booleans `api_key_configured` and `metrics_api_key_configured` are recorded.
- HTTP 401/403 from inference are reported as authentication/authorization failures.
- Error handling attempts to redact configured secrets, but this is **not** a substitute for protecting logs and generated text.
- Use HTTPS for remote endpoints so Bearer credentials are encrypted in transit.
- Do not embed keys in `--url` or `--metrics-url`; these URLs may be logged in JSON.

---

## Quick start

### List available models

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --list-models
```

Example output (illustrative; IDs depend on the deployed server):

```text
Available models:
  MODEL_ID
```

With authentication, set `MTP_BENCH_API_KEY` first or pass `--api-key-env`.

### Basic vLLM benchmark

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --out result.json
```

### Parallel benchmark

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --out result-parallel4.json
```

### Longer prompts

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 4000 \
  --out result-longprompt.json
```

### vLLM benchmark with Prometheus / MTP metrics

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --prometheus \
  --out vllm-prometheus.json
```

Equivalent explicit form:

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --metrics-backend vllm \
  --metrics-url http://127.0.0.1:8000/metrics \
  --out vllm-prometheus.json
```

### llama.cpp benchmark

```bash
python mtp_bench.py \
  --backend llamacpp \
  --url http://127.0.0.1:8080 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --no-stream-usage \
  --out llamacpp-result.json
```

Use `--no-stream-usage` only when necessary; omitting usage requests may reduce the availability of token counts.

### llama.cpp benchmark with metrics

```bash
python mtp_bench.py \
  --backend llamacpp \
  --url http://127.0.0.1:8080 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --metrics-backend llamacpp \
  --no-stream-usage \
  --out llamacpp-metrics.json
```

### SGLang benchmark

```bash
python mtp_bench.py \
  --backend sglang \
  --url http://127.0.0.1:30000 \
  --model MODEL_ID \
  --parallel 4 \
  --metrics-backend sglang \
  --out sglang-metrics.json
```

Omit `--metrics-backend sglang` if the server does not expose Prometheus metrics.

### Ollama benchmark

```bash
python mtp_bench.py \
  --backend ollama \
  --url http://127.0.0.1:11434/v1 \
  --list-models

python mtp_bench.py \
  --backend ollama \
  --url http://127.0.0.1:11434 \
  --model MODEL_ID \
  --out ollama-result.json
```

The default metrics backend is `none`.

### oMLX benchmark

```bash
python mtp_bench.py \
  --backend omlx \
  --url http://127.0.0.1:8000 \
  --list-models

python mtp_bench.py \
  --backend omlx \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --out omlx-result.json
```

Use the port configured in your oMLX deployment.

---

## Command-line options

### Server options

| Option | Default | Description |
| --- | --- | --- |
| `--url` | `http://127.0.0.1:8000` | Server base URL, with or without a trailing `/v1` |
| `--backend` | `openai` | Informational tag: `openai`, `vllm`, `llamacpp`, `sglang`, `ollama`, `omlx` |
| `--model` | none | Exact model ID sent to chat completions |
| `--list-models` | false | Query `/v1/models` and list available model IDs |
| `--api-key` | none | Bearer API key for inference endpoints; explicit value has precedence |
| `--api-key-env` | `MTP_BENCH_API_KEY` | Environment variable name to read the inference API key from |

### Benchmark options

| Option | Default | Description |
| --- | --- | --- |
| `--max-tokens` | `192` | Maximum output tokens per request |
| `--temperature` | `0.0` | Sampling temperature |
| `--seed` | `42` | Base seed; each prompt gets `seed + index` |
| `--parallel` | `1` | Number of concurrent HTTP requests |
| `--pad-tokens` | `1200` | Approximate neutral padding tokens added to each prompt |
| `--timeout` | `300` | HTTP request timeout in seconds |
| `--out` | none | Write benchmark JSON output |
| `--diff A.json B.json` | none | Compare two previous benchmark JSON files; no API call required |

### Streaming options

| Option | Default | Description |
| --- | --- | --- |
| `--no-stream-usage` | false | Omit `stream_options.include_usage` |
| `--include-text` | false | Include generated text in each result record |
| `--include-timings` | false | Include parsed and raw timing chunks in JSON |

### Metrics options

| Option | Default | Description |
| --- | --- | --- |
| `--prometheus` | false | Alias for `--metrics-backend vllm` |
| `--metrics-backend` | `none` | `none`, `vllm`, `llamacpp`, `sglang`, `generic` |
| `--metrics-url` | `<url>/metrics` | Optional custom Prometheus endpoint when enabled |
| `--metrics-api-key` | none | Separate Bearer key for the metrics endpoint |
| `--metrics-use-api-key` | false | Explicitly reuse the inference API key for metrics |
| `--metrics-timeout` | `10` | Prometheus scrape timeout in seconds |
| `--include-metrics-raw` | false | Include raw before/after metric snapshots in JSON |
| `--list-metrics` | false | List scraped metric names and exit |

The separate metrics key can also be supplied through `MTP_BENCH_METRICS_API_KEY`. The metrics key is not automatically inherited from the inference key.

---

## How parallelism works

`--parallel` controls independent, concurrent client-side HTTP requests. For example:

```bash
--parallel 4
```

sends up to four independent streaming requests at the same time. It does **not** split a single prompt across four threads.

For nine prompts and `--parallel 4`, the execution is conceptually:

```text
first four requests start concurrently
as a worker finishes, it starts the next queued prompt
repeat until all nine prompts finish
```

Parallelized by the script:

```text
HTTP request submission
stream reading
client-side timing measurement
waiting for responses
```

Handled by the inference server:

```text
prompt prefill
decode
speculative decoding / MTP
KV-cache management
batch scheduling / continuous batching, if supported
model execution
```

A server may batch requests internally. Parallel load may increase total server throughput while also increasing per-request queueing or latency.

---

## Metrics explained

Each request produces one result row. Example (illustrative values):

```text
code_python        prompt=  1360 out= 192 pred= 192 pred_tok/s=   22.31 draft= n/a acc= n/a rate=   n/a ttft=  1.240s prompt_tok/s=   1096.77 gen_tok/s=   21.88 gen_s=  8.774s wall= 10.014s
```

### Per-request fields

| Field | Meaning |
| --- | --- |
| `prompt` / `prompt_tokens` | Input tokens from response `usage` or supported fallback `timings` |
| `out` / `completion_tokens` | Output tokens from `usage` or supported fallback `timings` |
| `pred` / `predicted_n` | Server predicted-token count, or fallback completion tokens |
| `pred_tok/s` | Server-reported predicted tokens/s, if present, else client estimate |
| `draft` / `draft_n` | Draft tokens proposed; `n/a` if not exposed |
| `acc` / `draft_n_accepted` | Draft tokens accepted; `n/a` if not exposed |
| `rate` / `accept_rate` | Accepted / drafted, when both are available and drafts > 0 |
| `ttft` / `ttft_s` | Client-observed first text or first parsed streaming event |
| `prompt_tok/s` | `prompt_tokens / TTFT` approximation |
| `gen_tok/s` | `completion_tokens / generation_s`, or fallback using wall time |
| `gen_s` | Time from first text/event to request end |
| `wall` / `wall_s` | Total client-observed request duration |
| `token_usage_source` | `usage`, `timings`, or `unavailable` |
| `timings_returned` | Whether the server exposed recognized timing fields |

### Token usage fallback

Token counts are extracted in this order:

1. Standard streaming `usage.prompt_tokens` / `usage.completion_tokens`, when present.
2. Compatible server timing fields such as `prompt_n` / `predicted_n`.
3. Zero values when the source is unavailable; such zeros are **not valid performance observations**.

Inspect `token_usage_source` and `n_requests_missing_token_usage`. If any requests report `unavailable`, token-based throughput comparisons may be incomplete or misleading.

### Important TTFT caveat

TTFT is not pure prompt prefill time. It may include:

```text
HTTP overhead
chat-template processing
server queueing
scheduler delay
prompt prefill
first-token decode
streaming/network overhead
```

Therefore:

```text
prompt_tokens / TTFT
```

is an approximation. It is most useful with:

```text
--parallel 1
--pad-tokens 1000
```

or larger prompts, on an otherwise idle server. With `--parallel > 1`, the `prompt_tok/s` output receives a `*`, for example:

```text
prompt_tok/s=   987.45*
```

Use wall-clock suite throughput for concurrency comparisons.

---

## Aggregate metrics explained

The JSON output contains an `aggregate` object. Example (illustrative):

```json
{
  "n_requests": 9,
  "n_requests_missing_token_usage": 0,
  "n_requests_with_draft_stats": 0,
  "parallel": 4,
  "total_prompt_tokens": 12240,
  "mean_prompt_tokens": 1360.0,
  "pad_tokens_target_approx": 1200,
  "total_completion_tokens": 1728,
  "total_tokens": 13968,
  "total_draft": null,
  "total_draft_accepted": null,
  "aggregate_accept_rate": null,
  "suite_wall_s": 31.421,
  "wall_clock_prompt_tokens_per_second": 389.55,
  "wall_clock_generation_tokens_per_second": 55.0
}
```

### General aggregate fields

| Field | Meaning |
| --- | --- |
| `n_requests` | Number of completed benchmark requests |
| `n_requests_missing_token_usage` | Requests for which both usage and supported timing counts were unavailable |
| `n_requests_with_draft_stats` | Requests reporting a draft-token count |
| `parallel` | Requested client-side concurrency |
| `total_prompt_tokens` | Sum of known/fallback prompt-token values |
| `mean_prompt_tokens` | Mean reported/fallback prompt tokens per request |
| `pad_tokens_target_approx` | Approximate requested prompt padding |
| `total_completion_tokens` | Sum of known/fallback completion-token values |
| `total_tokens` | Total reported/fallback tokens |
| `total_predicted` | Sum of predicted/generated token counts |
| `total_draft` | Sum of draft counts, or `null` if not available |
| `total_draft_accepted` | Sum of accepted draft counts, or `null` if not available |
| `aggregate_accept_rate` | Accepted/drafted if enough data are available; otherwise `null` |
| `sum_request_wall_s` | Sum of individual request wall times |
| `suite_wall_s` | Actual elapsed time for the full benchmark suite |
| `ttft_s_total` | Sum of per-request TTFT values |
| `generation_s_total` | Sum of per-request generation intervals |

### Throughput fields

| Field | Meaning | Interpretation |
| --- | --- | --- |
| `aggregate_prompt_tokens_per_second_estimate` | `total_prompt_tokens / sum(TTFT)` | Only produced for `parallel=1`; approximate |
| `aggregate_generation_tokens_per_second` | `total_completion_tokens / sum(generation_s)` | Only produced for `parallel=1` |
| `aggregate_predicted_per_second` | `total_predicted / sum(generation_s)` | Only produced for `parallel=1` |
| `wall_clock_prompt_tokens_per_second` | `total_prompt_tokens / suite_wall_s` | Suite-level input tokens/s at any concurrency |
| `wall_clock_generation_tokens_per_second` | `total_completion_tokens / suite_wall_s` | Suite-level generated tokens/s at any concurrency |
| `wall_clock_predicted_per_second` | `total_predicted / suite_wall_s` | Suite-level predicted/generated tokens/s |

For concurrent tests, prioritize:

```text
suite_wall_s
wall_clock_prompt_tokens_per_second
wall_clock_generation_tokens_per_second
wall_clock_predicted_per_second
```

Do not mix `predicted_per_second` (which can be server-reported for individual requests) with aggregate client-derived throughput without noting the different denominators.

---

## MTP / speculative decoding metrics

Example response fields when the server does **not** expose draft details:

```json
{
  "predicted_n": 192,
  "predicted_per_second": 22.31,
  "predicted_per_second_server": null,
  "predicted_per_second_client": 22.31,
  "draft_n": null,
  "draft_n_accepted": null,
  "accept_rate": null
}
```

### Response-body MTP fields

| Field | Meaning |
| --- | --- |
| `predicted_n` | Server predicted count if supplied; otherwise fallback completion tokens |
| `predicted_per_second` | Server value when available, otherwise client estimate |
| `predicted_per_second_server` | Direct server-reported value, or `null` |
| `predicted_per_second_client` | Client-estimated generated/predicted tokens/s |
| `draft_n` | Proposed draft tokens, if exposed |
| `draft_n_accepted` | Accepted draft tokens, if exposed |
| `accept_rate` | `draft_n_accepted / draft_n`, when valid |

**`draft_n: null` is not evidence of zero speculative decoding activity.** It often means the compatible response simply omits the statistic. A measured `draft_n: 0` is different, but by itself still does not establish why no drafts were recorded.

For vLLM, prefer Prometheus when available; for SGLang, interpret its gauge snapshots separately. Ollama and oMLX may provide generation token usage without exposing per-request draft/acceptance statistics through the OpenAI-compatible API.

---

## Prometheus metrics

When enabled, metrics are scraped before and after the requests. The resulting data is included under:

```json
{
  "prometheus": {},
  "aggregate": {}
}
```

The selected Prometheus summary fields are also copied into `aggregate`.

### vLLM Prometheus backend

```bash
--prometheus
```

or:

```bash
--metrics-backend vllm
```

The script looks for metric names such as:

```text
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_emitted_tokens_total
```

Example (illustrative; only if those metrics exist):

```json
{
  "metrics_backend": "vllm",
  "prometheus_total_drafts_delta": 576,
  "prometheus_total_draft_tokens_delta": 1152,
  "prometheus_total_accepted_tokens_delta": 840,
  "prometheus_total_emitted_tokens_delta": 1728,
  "prometheus_accept_rate": 0.7292
}
```

Acceptance rate is:

```text
accepted draft tokens delta / proposed draft tokens delta
```

If relevant metric names are absent, the corresponding vLLM count fields are `null`, rather than proof of no MTP activity. Some vLLM releases use other metric names, so inspect `--list-metrics` if data is missing.

### llama.cpp Prometheus backend

```bash
--metrics-backend llamacpp
```

The metrics are matched on a best-effort basis, for example:

```json
{
  "metrics_backend": "llamacpp",
  "llamacpp_prompt_tokens_delta_best_effort": 12240,
  "llamacpp_generation_tokens_delta_best_effort": 1728,
  "llamacpp_requests_delta_best_effort": 9
}
```

The backend also returns `llamacpp_interesting_metric_deltas` for names containing terms such as:

```text
llama
prompt
tokens
decode
prefill
kv
cache
request
```

Version differences mean a zero best-effort count might indicate a missing name rather than zero activity. Check `llamacpp_metric_names_used`.

### SGLang Prometheus backend

```bash
--metrics-backend sglang
```

For known speculative decoding gauges, the script reports **snapshots**, for example:

```json
{
  "metrics_backend": "sglang",
  "sglang_spec_gauges_before": {
    "sglang:spec_accept_rate": 0.72
  },
  "sglang_spec_gauges_after": {
    "sglang:spec_accept_rate": 0.79
  },
  "sglang_counter_deltas": {}
}
```

The numerical values above are illustrative. Supported gauge names in the script include:

```text
sglang:spec_accept_rate
sglang:spec_accept_length
sglang:spec_cap_length
sglang:spec_block_accept_length
sglang:spec_num_steps
sglang:spec_num_draft_tokens
```

Gauge values are **not cumulative counts**. The script does not subtract them to derive a benchmark-wide accept rate. The parser merges label variants by metric name, so a multi-series sum must not be assumed to be a meaningful global rate.

For `sglang:*` metric names ending in `_total`, `_sum`, or `_count`, the script also provides `sglang_counter_deltas`.

### Generic Prometheus backend

```bash
--metrics-backend generic
```

Example:

```json
{
  "metrics_backend": "generic",
  "prometheus_all_nonzero_deltas": {}
}
```

The generic backend subtracts before/after values for every parsed metric (excluding histogram buckets). This is useful for discovery, but values from **gauges are not event counts**, and unlabeled metric-name aggregation can be misleading. Use a specific backend for interpretation.

### Prometheus caveats

- Metrics describe the whole server, not necessarily only this benchmark.
- Other traffic between snapshots contaminates counter deltas.
- A server restart can reset counters; negative counter deltas are clamped to zero with a warning.
- Some metrics are gauges, not counters. This matters especially for SGLang and the generic backend.
- Histogram `_bucket` series are excluded; `_sum` and `_count` are retained.
- Label dimensions are aggregated by metric name and cannot be reconstructed from the result.
- No Prometheus metrics are collected unless explicitly enabled.

---

## Listing available metrics

### vLLM

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --metrics-backend vllm \
  --list-metrics
```

### llama.cpp

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8080 \
  --metrics-backend llamacpp \
  --list-metrics
```

### SGLang

```bash
python mtp_bench.py \
  --url http://127.0.0.1:30000 \
  --metrics-backend sglang \
  --list-metrics
```

### Generic

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --metrics-backend generic \
  --list-metrics
```

Write metrics to JSON:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --metrics-backend generic \
  --list-metrics \
  --out metrics.json
```

If `--list-metrics` is used with `--metrics-backend none`, the script uses the generic backend for that command. Authentication uses the separate metrics-credential rules described above.

---

## Comparing benchmark results

Run a baseline:

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 1 \
  --out baseline.json
```

Run a parallel benchmark:

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --out parallel4.json
```

Compare offline:

```bash
python mtp_bench.py --diff baseline.json parallel4.json
```

The diff prints aggregate differences and per-prompt comparisons. Missing values display as:

```text
n/a
```

rather than fake zeros. For a scientifically meaningful comparison, use the same model, prompt lengths, tokenizer, generation settings, hardware conditions, and server load. A serial-vs-parallel comparison answers a **different question** from a like-for-like engine comparison.

---

## Recommended benchmark methodology

### For single-request latency

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 1 \
  --pad-tokens 1200 \
  --out single.json
```

Focus on:

```text
ttft_s
prompt_tokens_per_second_estimate
generation_tokens_per_second
wall_s
```

### For server throughput

Use increasing parallelism:

```bash
for p in 1 2 4 8 16; do
  python mtp_bench.py \
    --backend vllm \
    --url http://127.0.0.1:8000 \
    --model MODEL_ID \
    --parallel "$p" \
    --pad-tokens 1200 \
    --out "parallel-${p}.json"
done
```

Focus on:

```text
suite_wall_s
wall_clock_prompt_tokens_per_second
wall_clock_generation_tokens_per_second
n_requests_missing_token_usage
```

### For vLLM MTP validation

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --prometheus \
  --out mtp-prometheus.json
```

Focus on:

```text
prometheus_total_draft_tokens_delta
prometheus_total_accepted_tokens_delta
prometheus_accept_rate
```

Nonzero counters demonstrate that speculative activity is visible to the scraper during the interval, not necessarily attributable exclusively to the benchmark if other requests were active.

### For SGLang speculative decoding observability

```bash
python mtp_bench.py \
  --backend sglang \
  --url http://127.0.0.1:30000 \
  --model MODEL_ID \
  --parallel 4 \
  --metrics-backend sglang \
  --out sglang-spec.json
```

Focus on:

```text
sglang_spec_gauges_before
sglang_spec_gauges_after
sglang_counter_deltas
```

Do **not** treat the difference between `sglang:spec_accept_rate` snapshots as the benchmark acceptance rate.

### For Ollama and oMLX

Use the normal OpenAI-compatible stream with `--metrics-backend none` unless a supported Prometheus endpoint is known. Focus on client-side latency/throughput and token-usage availability. Record the actual engine version and runtime settings separately.

---

## JSON output structure

For a normal benchmark:

```json
{
  "config": {},
  "results": [],
  "aggregate": {},
  "prometheus": {}
}
```

`prometheus` is present **only** when metrics scraping is enabled.

### `config`

Illustrative values:

```json
{
  "url": "http://127.0.0.1:30000",
  "backend": "sglang",
  "api_key_configured": true,
  "metrics_api_key_configured": false,
  "model": "MODEL_ID",
  "max_tokens": 192,
  "temperature": 0.0,
  "seed": 42,
  "parallel": 4,
  "pad_tokens_approx": 1200,
  "stream_usage": true,
  "metrics_backend": "sglang",
  "metrics_url": "http://127.0.0.1:30000/metrics"
}
```

The actual `config` object also contains `method` and `method_caveats`. Authentication keys are not recorded.

### `results`

One object per prompt:

```json
{
  "index": 0,
  "name": "code_python",
  "wall_s": 10.014,
  "ttft_s": 1.24,
  "generation_s": 8.774,
  "prompt_tokens": 1360,
  "completion_tokens": 192,
  "total_tokens": 1552,
  "token_usage_source": "usage",
  "predicted_n": 192,
  "predicted_per_second": 21.88,
  "predicted_per_second_server": null,
  "predicted_per_second_client": 21.88,
  "draft_n": null,
  "draft_n_accepted": null,
  "accept_rate": null,
  "prompt_tokens_per_second_estimate": 1096.77,
  "prompt_tps_estimate_reliable": true,
  "generation_tokens_per_second": 21.88,
  "timings_returned": false
}
```

If `--include-timings` is enabled, the result also contains `timings` and `raw_timing_chunks`. If `--include-text` is enabled, it contains the generated `text`.

### `aggregate`

Illustrative:

```json
{
  "n_requests": 9,
  "n_requests_missing_token_usage": 0,
  "n_requests_with_draft_stats": 0,
  "parallel": 4,
  "total_prompt_tokens": 12240,
  "mean_prompt_tokens": 1360.0,
  "pad_tokens_target_approx": 1200,
  "total_completion_tokens": 1728,
  "total_tokens": 13968,
  "total_predicted": 1728,
  "total_draft": null,
  "total_draft_accepted": null,
  "aggregate_accept_rate": null,
  "suite_wall_s": 31.421,
  "aggregate_prompt_tokens_per_second_estimate": null,
  "wall_clock_prompt_tokens_per_second": 389.55,
  "wall_clock_generation_tokens_per_second": 55.0
}
```

The actual object contains additional timing totals, generation rates, and, if enabled, selected Prometheus data.

### `prometheus`

Example vLLM output (illustrative):

```json
{
  "metrics_backend": "vllm",
  "prometheus_total_drafts_delta": 576,
  "prometheus_total_draft_tokens_delta": 1152,
  "prometheus_total_accepted_tokens_delta": 840,
  "prometheus_total_emitted_tokens_delta": 1728,
  "prometheus_accept_rate": 0.7292
}
```

SGLang output instead contains `sglang_spec_gauges_before`, `sglang_spec_gauges_after`, and `sglang_counter_deltas`.

---

## Notes about prompt padding

Each benchmark prompt can be extended with neutral padding designed not to alter the task's meaning:

```text
The following text is neutral benchmark padding. Do not use it to answer the task.
It exists only to increase prompt length for prompt-processing speed measurement.
```

`--pad-tokens` is approximate. The script does not tokenize prompts locally and does not guarantee identical effective token lengths across different tokenizers or chat templates.

Always inspect:

```text
mean_prompt_tokens
```

and `token_usage_source` to determine whether the server supplied usable prompt token counts.

---

## Troubleshooting

### `ERROR: --model is required`

List models first:

```bash
python mtp_bench.py --url http://127.0.0.1:8000 --list-models
```

Then supply the exact returned ID:

```bash
python mtp_bench.py --url http://127.0.0.1:8000 --model MODEL_ID
```

If listing is unavailable but you already know the correct ID, pass `--model` directly.

### `The model does not exist`

The model name differs from the server's model ID. List available models or query the endpoint using the appropriate Bearer token:

```bash
curl -H "Authorization: Bearer ${MTP_BENCH_API_KEY}" \
  http://127.0.0.1:8000/v1/models
```

For an unauthenticated local server, omit the header.

### `HTTP 401` or `HTTP 403`

Check:

- whether the inference server requires an API key;
- the value of `MTP_BENCH_API_KEY` or the variable selected by `--api-key-env`;
- whether the key has permissions for the requested model;
- whether an API gateway expects Bearer authentication or a different scheme.

The script supports the **Bearer** scheme, not arbitrary custom authentication headers.

For a protected metrics endpoint, set `MTP_BENCH_METRICS_API_KEY` or explicitly use `--metrics-use-api-key` when appropriate.

### `unknown url type`

The URL is malformed.

Wrong:

```bash
--url htttp://127.0.0.1:8000
```

Correct:

```bash
--url http://127.0.0.1:8000
```

The script also accepts:

```bash
--url http://127.0.0.1:11434/v1
```

which it normalizes to the API root before appending endpoint paths.

### `HTTP 400` or `HTTP 422` with llama.cpp

Older llama.cpp builds may reject `stream_options.include_usage`. Try:

```bash
python mtp_bench.py \
  --backend llamacpp \
  --url http://127.0.0.1:8080 \
  --model MODEL_ID \
  --no-stream-usage
```

The script's automatic fallback applies only when the returned error explicitly identifies unsupported `stream_options`. Other HTTP errors are not automatically retried.

### `prompt_tokens` is zero

The server did not return usable token counts. Causes may include:

```text
unsupported stream usage
missing usage in final streaming event
missing compatible timing fields
older or customized OpenAI-compatible implementations
```

Inspect:

```bash
--include-timings
```

and check `token_usage_source`. Generic `/metrics` values cannot always replace per-request usage reliably.

### Prometheus metrics are empty

Verify the endpoint, server metrics configuration, and any required authentication. For an unauthenticated endpoint:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

For vLLM speculative metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'spec_decode|draft|accepted'
```

For SGLang:

```bash
curl -s http://127.0.0.1:30000/metrics | grep -E 'sglang:spec_|spec_accept'
```

For llama.cpp, ensure the server has metrics enabled (where supported). If `/metrics` is on another host or port, specify `--metrics-url`.

### `total_draft` is `null` or zero

- `null`: the relevant response fields are not available in any benchmark request.
- `0`: the server reported draft-count values, and their sum is zero.

Neither value alone establishes whether speculative decoding is configured. Inspect vLLM Prometheus counters or SGLang gauge snapshots if available.

### Ollama or oMLX returns no detailed MTP information

This can be expected. The benchmark uses OpenAI-compatible streaming, not the engines' native stats APIs. Token usage and speculative-decoding observability are different capabilities.

### `--metrics-backend sglang` fails

Confirm that SGLang has metrics enabled, the URL is correct, and any `/metrics` authentication is supplied separately. If it is unavailable, rerun without metrics scraping:

```bash
--metrics-backend none
```

### Comparison shows `n/a`

A metric was missing from one result file or unavailable in the response. `n/a` is not a measured zero. Check the two files' model IDs, backends, metrics configuration, and token usage sources.

---

## Interpreting parallel results

With `--parallel > 1`, request latencies overlap.

Example:

```json
{
  "sum_request_wall_s": 90.0,
  "suite_wall_s": 27.0
}
```

`sum_request_wall_s` is the sum of all individual durations; `suite_wall_s` is the real elapsed benchmark time.

For throughput under load, focus on:

```text
wall_clock_prompt_tokens_per_second
wall_clock_generation_tokens_per_second
```

rather than TTFT-derived per-request prompt tok/s. Parallel throughput also depends on the server's queueing, batching, KV-cache, and scheduling behavior.

---

## Recommended result fields

### For prompt/prefill measurement

At `parallel=1`:

```text
prompt_tokens_per_second_estimate
aggregate_prompt_tokens_per_second_estimate
ttft_s
```

At `parallel>1`:

```text
wall_clock_prompt_tokens_per_second
suite_wall_s
```

### For generation throughput

At `parallel=1`:

```text
generation_tokens_per_second
aggregate_generation_tokens_per_second
```

At `parallel>1`:

```text
wall_clock_generation_tokens_per_second
suite_wall_s
```

Always check `token_usage_source` and `n_requests_missing_token_usage` before comparing token/s values.

### For MTP/speculative decoding

Response body:

```text
draft_n
draft_n_accepted
accept_rate
```

vLLM Prometheus:

```text
prometheus_total_draft_tokens_delta
prometheus_total_accepted_tokens_delta
prometheus_accept_rate
```

SGLang Prometheus:

```text
sglang_spec_gauges_before
sglang_spec_gauges_after
sglang_counter_deltas
```

---

## Limitations

This is primarily a client-side benchmark. It cannot directly observe internal scheduling or device kernel timings unless the server publishes usable telemetry.

Important limitations:

- TTFT is not pure prompt prefill time and may include the first decode step.
- Concurrency increases queueing; TTFT-derived per-request prefill estimates become unreliable.
- Wall-clock prompt tokens/s is a suite-level rate, not isolated prefill hardware speed.
- Token usage may be unavailable even when text was generated.
- Missing MTP response fields are not proof that MTP is disabled.
- Prometheus counters are global and can include other traffic.
- SGLang speculative gauges are snapshots, not cumulative accept counts.
- The generic Prometheus mode may subtract gauges and cannot infer their semantic meaning.
- llama.cpp metric names and response timing formats may vary by version.
- The script does not automatically query Ollama native APIs or oMLX admin statistics.
- The `--backend` tag does not change payload shape or API protocol; backend support depends on the installed implementation.
- The approximate prompt padding length differs across tokenizers.
- The benchmark suite has only nine prompts; repeat tests and report variability for meaningful performance comparisons.
- API-key protection depends on transport security, shell/log hygiene, and server configuration.

For clean measurements, benchmark an otherwise idle inference server, warm it up consistently, run multiple trials, and record hardware, quantization, model, speculative-decoding settings, and exact server versions.

---

## Example benchmark workflow

### 1. Start vLLM

Example (adjust model ID and hardware parameters to your deployment):

```bash
vllm serve MODEL_ID \
  --port 8000 \
  --tensor-parallel-size 8
```

### 2. Verify the server

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics | head
```

If the server requires an API key, configure `MTP_BENCH_API_KEY` before running the script. If `/metrics` uses different authentication, configure it separately.

### 3. Run a single-request baseline

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 1 \
  --pad-tokens 1200 \
  --prometheus \
  --out baseline.json
```

### 4. Run a parallel test

```bash
python mtp_bench.py \
  --backend vllm \
  --url http://127.0.0.1:8000 \
  --model MODEL_ID \
  --parallel 4 \
  --pad-tokens 1200 \
  --prometheus \
  --out parallel4.json
```

### 5. Compare results

```bash
python mtp_bench.py --diff baseline.json parallel4.json
```

### 6. Repeat on another backend

For SGLang:

```bash
python mtp_bench.py \
  --backend sglang \
  --url http://127.0.0.1:30000 \
  --model MODEL_ID \
  --parallel 4 \
  --metrics-backend sglang \
  --out sglang-parallel4.json
```

For Ollama or oMLX, use the respective `--backend` tag and compatible API root URL. Leave metrics disabled unless your deployment explicitly supports a compatible endpoint. Compare only after checking token-count availability and equivalent test conditions.

---

## Validation and implementation notes

- The updated script supports Bearer authentication for model listing and chat completion requests, including streaming fallback retries.
- Metric-scraping credentials are separate by default.
- The script supports vLLM, llama.cpp, and SGLang metrics parsers plus a generic discovery mode.
- Draft-token statistics are nullable when the server does not supply them.
- Fallback token usage from response timings is best-effort.
- The existing development package reports syntax checks and simulated HTTP integration tests. **Those are not a substitute for testing against your actual vLLM, llama.cpp, SGLang, Ollama, and oMLX deployments.**

Upstream references:

- [vLLM documentation](https://docs.vllm.ai/)
- [llama.cpp project](https://github.com/ggml-org/llama.cpp)
- [SGLang documentation](https://docs.sglang.ai/)
- [Ollama OpenAI compatibility](https://docs.ollama.com/openai)
- [oMLX project](https://github.com/jundot/omlx)

---

## License

MIT
