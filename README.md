

# MTP / Speculative Decoding Benchmark

`mtp_bench.py` is a Python benchmark tool for OpenAI-compatible local inference servers such as **vLLM** and **llama.cpp**.

It measures:

* output generation throughput
* prompt / prefill throughput estimate
* TTFT, time to first token
* wall-clock benchmark throughput
* speculative decoding / MTP statistics when available
* vLLM Prometheus speculative decoding metrics
* llama.cpp Prometheus metrics, best-effort
* parallel request throughput

The script is designed for testing local inference servers exposing an OpenAI-compatible `/v1/chat/completions` endpoint.

---

## Features

### Core benchmark

The script sends a fixed set of benchmark prompts to the server and records per-request and aggregate metrics.

It supports:

* streaming chat completions
* TTFT measurement
* configurable model name
* configurable output tokens
* configurable prompt padding
* configurable parallelism
* JSON output
* result comparison with `--diff`

### Prompt processing speed estimate

The script estimates prompt processing speed as:

```text
prompt_tokens / TTFT
```

This estimate is only considered reliable when:

```text
--parallel 1
```

When `--parallel > 1`, TTFT includes queueing and scheduling delay, so per-request prompt tok/s is marked with `*` and aggregate prompt tok/s is set to `null`.

The script still reports wall-clock prompt throughput under parallel load:

```text
wall_clock_prompt_tokens_per_second
```

This is the better metric for concurrent tests.

### MTP / speculative decoding stats

The script keeps the classic MTP fields:

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

If the OpenAI-compatible response does not expose these values, the script falls back to completion-token-based estimates where possible.

For vLLM, the recommended way to validate MTP is Prometheus scraping with:

```bash
--prometheus
```

or:

```bash
--metrics-backend vllm
```

### Prometheus metrics support

The script can scrape `/metrics` before and after a benchmark run and calculate deltas.

Supported backends:

```text
none
vllm
llamacpp
generic
```

The `vllm` backend extracts speculative decoding metrics such as draft tokens, accepted tokens, emitted tokens, and accept rate.

The `llamacpp` backend uses best-effort metric detection because llama.cpp metric names may differ between versions.

The `generic` backend returns all non-zero metric deltas.

---

## Requirements

Python 3.8+ is recommended.

The script only uses the Python standard library:

```text
argparse
concurrent.futures
json
math
sys
time
urllib
```

No third-party Python package is required.

---

## Supported servers

### vLLM

The script targets the OpenAI-compatible vLLM API server.

Expected endpoints:

```text
/v1/models
/v1/chat/completions
/metrics
```

The `/metrics` endpoint is used only when Prometheus metrics are enabled in the script.

Example vLLM URL:

```text
http://127.0.0.1:8000
```

### llama.cpp

The script can also run against `llama-server` if OpenAI-compatible chat completion is enabled.

For Prometheus metrics, start llama.cpp server with metrics enabled, for example:

```bash
./llama-server \
  -m model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics
```

Example llama.cpp URL:

```text
http://127.0.0.1:8080
```

Older llama.cpp builds may reject `stream_options.include_usage`. Use:

```bash
--no-stream-usage
```

if needed.

---

## Installation

Save the script as:

```bash
mtp_bench.py
```

Make it executable:

```bash
chmod +x mtp_bench.py
```

Run help:

```bash
python mtp_bench.py --help
```

---

## Quick start

### List available models

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --list-models
```

Example output:

```text
Available models:
  Qwen/Qwen3.6-35B-A3B
```

### Basic vLLM benchmark

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --out result.json
```

### Parallel benchmark

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --out result-parallel4.json
```

### Longer prompts

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --pad-tokens 4000 \
  --out result-longprompt.json
```

### vLLM benchmark with Prometheus / MTP metrics

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --pad-tokens 1200 \
  --prometheus \
  --out vllm-prometheus.json
```

Equivalent explicit form:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --pad-tokens 1200 \
  --metrics-backend vllm \
  --metrics-url http://127.0.0.1:8000/metrics \
  --out vllm-prometheus.json
```

### llama.cpp benchmark

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8080 \
  --model local-model \
  --parallel 4 \
  --pad-tokens 1200 \
  --no-stream-usage \
  --out llamacpp-result.json
```

### llama.cpp benchmark with metrics

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8080 \
  --model local-model \
  --parallel 4 \
  --pad-tokens 1200 \
  --metrics-backend llamacpp \
  --metrics-url http://127.0.0.1:8080/metrics \
  --no-stream-usage \
  --out llamacpp-metrics.json
```

---

## Command-line options

### Server options

| Option          |                 Default | Description                                      |
| --------------- | ----------------------: | ------------------------------------------------ |
| `--url`         | `http://127.0.0.1:8000` | Base URL of the OpenAI-compatible server         |
| `--model`       |                    none | Model name sent in the chat completion request   |
| `--list-models` |                   false | Query `/v1/models` and print available model IDs |

### Benchmark options

| Option                 | Default | Description                                             |
| ---------------------- | ------: | ------------------------------------------------------- |
| `--max-tokens`         |   `192` | Maximum output tokens per request                       |
| `--temperature`        |   `0.0` | Sampling temperature                                    |
| `--seed`               |    `42` | Base seed; each prompt gets `seed + index`              |
| `--parallel`           |     `1` | Number of concurrent client requests                    |
| `--pad-tokens`         |  `1200` | Approximate neutral padding tokens added to each prompt |
| `--timeout`            |   `300` | HTTP request timeout in seconds                         |
| `--out`                |    none | Write benchmark result JSON to file                     |
| `--diff A.json B.json` |    none | Compare two previous benchmark JSON files               |

### Streaming options

| Option              | Default | Description                                       |
| ------------------- | ------: | ------------------------------------------------- |
| `--no-stream-usage` |   false | Do not send `stream_options.include_usage`        |
| `--include-text`    |   false | Include generated text in JSON output             |
| `--include-timings` |   false | Include raw response timing fields in JSON output |

### Metrics options

| Option                  |         Default | Description                                            |
| ----------------------- | --------------: | ------------------------------------------------------ |
| `--prometheus`          |           false | Alias for `--metrics-backend vllm`                     |
| `--metrics-backend`     |          `none` | Metrics backend: `none`, `vllm`, `llamacpp`, `generic` |
| `--metrics-url`         | `<url>/metrics` | Metrics endpoint                                       |
| `--metrics-timeout`     |            `10` | Metrics scrape timeout in seconds                      |
| `--include-metrics-raw` |           false | Include raw before/after metrics in JSON               |
| `--list-metrics`        |           false | List scraped metric names and exit                     |

---

## How parallelism works

The `--parallel` option controls the number of concurrent client-side HTTP requests.

For example:

```bash
--parallel 4
```

means the script sends up to four independent streaming chat-completion requests at the same time.

It does **not** split a single prompt across four threads.

With 9 benchmark prompts and `--parallel 4`, the execution is conceptually:

```text
slot 1: prompts 1, 2, 3, 4 start
slot 2: when one finishes, prompt 5 starts
slot 3: when another finishes, prompt 6 starts
...
until all 9 prompts complete
```

Parallelized by the script:

```text
HTTP request submission
stream reading
client-side timing measurement
waiting for responses
```

Handled by the server:

```text
prompt prefill
decode
speculative decoding / MTP
KV-cache management
batch scheduling
continuous batching
model execution
```

In vLLM, parallel requests may be batched internally by the scheduler. This usually improves total server throughput but may increase per-request latency.

---

## Metrics explained

Each request produces one result row.

Example row:

```text
code_python prompt=  1360 out= 192 pred= 192 pred_tok/s=   22.31 draft=   0 acc=   0 rate=   n/a ttft=  1.240s prompt_tok/s= 1096.77 gen_tok/s=   21.88 gen_s=  8.774s wall= 10.014s
```

### Per-request fields

| Field          | Meaning                                                          |
| -------------- | ---------------------------------------------------------------- |
| `prompt`       | Number of input tokens reported by the server                    |
| `out`          | Number of generated output tokens                                |
| `pred`         | Predicted/generated token count, MTP-compatible field            |
| `pred_tok/s`   | Predicted tokens per second                                      |
| `draft`        | Number of speculative/MTP draft tokens proposed                  |
| `acc`          | Number of speculative/MTP draft tokens accepted                  |
| `rate`         | `acc / draft`                                                    |
| `ttft`         | Client-observed time to first token or first useful stream event |
| `prompt_tok/s` | `prompt_tokens / TTFT`, reliable only at `parallel=1`            |
| `gen_tok/s`    | `completion_tokens / generation_s`                               |
| `gen_s`        | Time from first streamed token/event to request completion       |
| `wall`         | Total client-observed request time                               |

### Important TTFT caveat

TTFT is not pure prompt prefill time.

It includes:

```text
HTTP overhead
chat-template processing
server queueing
scheduler delay
prompt prefill
first-token decode
streaming overhead
```

Therefore:

```text
prompt_tokens / TTFT
```

is an approximation.

It is most useful with:

```text
--parallel 1
--pad-tokens 1000 or higher
```

With short prompts, overhead dominates.

With `--parallel > 1`, TTFT includes concurrency queueing and overlapping request intervals, so per-request prompt tok/s is marked with `*`.

Example:

```text
prompt_tok/s=  987.45*
```

In parallel mode, use wall-clock throughput instead.

---

## Aggregate metrics explained

The JSON output contains an `aggregate` object.

Example:

```json
{
  "n_requests": 9,
  "parallel": 4,
  "total_prompt_tokens": 12240,
  "mean_prompt_tokens": 1360.0,
  "pad_tokens_target_approx": 1200,
  "total_completion_tokens": 1728,
  "total_tokens": 13968,
  "suite_wall_s": 31.421,
  "wall_clock_prompt_tokens_per_second": 389.55,
  "wall_clock_generation_tokens_per_second": 55.0
}
```

### General aggregate fields

| Field                      | Meaning                              |
| -------------------------- | ------------------------------------ |
| `n_requests`               | Number of benchmark requests         |
| `parallel`                 | Client-side concurrency              |
| `total_prompt_tokens`      | Sum of all prompt tokens             |
| `mean_prompt_tokens`       | Average prompt tokens per request    |
| `pad_tokens_target_approx` | Requested approximate padding target |
| `total_completion_tokens`  | Sum of all output tokens             |
| `total_tokens`             | Prompt + output tokens               |
| `sum_request_wall_s`       | Sum of individual request wall times |
| `suite_wall_s`             | Real elapsed benchmark time          |
| `ttft_s_total`             | Sum of per-request TTFT values       |
| `generation_s_total`       | Sum of per-request generation times  |

### Throughput fields

| Field                                         | Reliable when   | Meaning                                       |
| --------------------------------------------- | --------------- | --------------------------------------------- |
| `aggregate_prompt_tokens_per_second_estimate` | `parallel=1`    | `total_prompt_tokens / sum(TTFT)`             |
| `aggregate_generation_tokens_per_second`      | `parallel=1`    | `total_completion_tokens / sum(generation_s)` |
| `aggregate_predicted_per_second`              | `parallel=1`    | `total_predicted / sum(generation_s)`         |
| `wall_clock_prompt_tokens_per_second`         | any parallelism | `total_prompt_tokens / suite_wall_s`          |
| `wall_clock_generation_tokens_per_second`     | any parallelism | `total_completion_tokens / suite_wall_s`      |
| `wall_clock_predicted_per_second`             | any parallelism | `total_predicted / suite_wall_s`              |

For concurrent benchmarks, the most important metrics are:

```text
suite_wall_s
wall_clock_prompt_tokens_per_second
wall_clock_generation_tokens_per_second
wall_clock_predicted_per_second
```

---

## MTP / speculative decoding metrics

The script reports MTP-compatible response fields:

```json
{
  "predicted_n": 192,
  "predicted_per_second": 22.31,
  "predicted_per_second_server": null,
  "predicted_per_second_client": 22.31,
  "draft_n": 0,
  "draft_n_accepted": 0,
  "accept_rate": null
}
```

### Response-body MTP fields

| Field                         | Meaning                               |
| ----------------------------- | ------------------------------------- |
| `predicted_n`                 | Number of generated/predicted tokens  |
| `predicted_per_second`        | Effective predicted tokens per second |
| `predicted_per_second_server` | Server-reported value, if available   |
| `predicted_per_second_client` | Client-calculated estimate            |
| `draft_n`                     | Number of speculative draft tokens    |
| `draft_n_accepted`            | Number of accepted draft tokens       |
| `accept_rate`                 | `draft_n_accepted / draft_n`          |

If `draft_n` and `draft_n_accepted` are zero, it may mean:

```text
MTP is disabled
or
the server does not expose MTP stats in the OpenAI response body
```

For vLLM, Prometheus is usually more reliable for MTP validation.

---

## Prometheus metrics

The script can scrape metrics before and after the benchmark run.

The delta is added to:

```json
{
  "prometheus": {},
  "aggregate": {}
}
```

and selected Prometheus fields are also copied into `aggregate`.

### vLLM Prometheus backend

Use:

```bash
--prometheus
```

or:

```bash
--metrics-backend vllm
```

The script looks for vLLM speculative decoding metrics such as:

```text
vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_emitted_tokens_total
```

Example output:

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

Prometheus accept rate is calculated as:

```text
prometheus_total_accepted_tokens_delta / prometheus_total_draft_tokens_delta
```

### llama.cpp Prometheus backend

Use:

```bash
--metrics-backend llamacpp
```

The llama.cpp backend is best-effort because metric names may differ between versions.

Example output:

```json
{
  "metrics_backend": "llamacpp",
  "llamacpp_prompt_tokens_delta_best_effort": 12240,
  "llamacpp_generation_tokens_delta_best_effort": 1728,
  "llamacpp_requests_delta_best_effort": 9
}
```

The backend also returns:

```json
{
  "llamacpp_interesting_metric_deltas": {}
}
```

with all non-zero deltas for metric names containing strings such as:

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

### Generic Prometheus backend

Use:

```bash
--metrics-backend generic
```

This returns all non-zero metric deltas:

```json
{
  "metrics_backend": "generic",
  "prometheus_all_nonzero_deltas": {}
}
```

This is useful when metric names are unknown.

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

### Generic

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --metrics-backend generic \
  --list-metrics
```

Write metric list to JSON:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --metrics-backend generic \
  --list-metrics \
  --out metrics.json
```

---

## Comparing benchmark results

Run a baseline:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 1 \
  --out baseline.json
```

Run another benchmark:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --out parallel4.json
```

Compare:

```bash
python mtp_bench.py --diff baseline.json parallel4.json
```

The diff prints aggregate deltas and per-prompt comparisons.

If a prompt is missing from one file, the diff prints:

```text
n/a
```

instead of fake zeroes.

---

## Recommended benchmark methodology

### For single-request latency

Use:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
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
    --url http://127.0.0.1:8000 \
    --model Qwen/Qwen3.6-35B-A3B \
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
```

### For vLLM MTP validation

Use Prometheus:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
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

If these are non-zero, speculative decoding/MTP activity is visible through Prometheus.

---

## JSON output structure

Top-level output:

```json
{
  "config": {},
  "results": [],
  "aggregate": {},
  "prometheus": {}
}
```

### `config`

Contains benchmark configuration:

```json
{
  "url": "http://127.0.0.1:8000",
  "model": "Qwen/Qwen3.6-35B-A3B",
  "max_tokens": 192,
  "temperature": 0.0,
  "seed": 42,
  "parallel": 4,
  "pad_tokens_approx": 1200,
  "stream_usage": true,
  "metrics_backend": "vllm",
  "metrics_url": "http://127.0.0.1:8000/metrics"
}
```

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
  "predicted_n": 192,
  "predicted_per_second": 21.88,
  "predicted_per_second_server": null,
  "predicted_per_second_client": 21.88,
  "draft_n": 0,
  "draft_n_accepted": 0,
  "accept_rate": null,
  "prompt_tokens_per_second_estimate": 1096.77,
  "prompt_tps_estimate_reliable": true,
  "generation_tokens_per_second": 21.88,
  "timings_returned": false
}
```

### `aggregate`

Summary across all prompts:

```json
{
  "n_requests": 9,
  "parallel": 4,
  "total_prompt_tokens": 12240,
  "mean_prompt_tokens": 1360.0,
  "pad_tokens_target_approx": 1200,
  "total_completion_tokens": 1728,
  "total_tokens": 13968,
  "total_predicted": 1728,
  "total_draft": 0,
  "total_draft_accepted": 0,
  "aggregate_accept_rate": null,
  "suite_wall_s": 31.421,
  "wall_clock_prompt_tokens_per_second": 389.55,
  "wall_clock_generation_tokens_per_second": 55.0
}
```

### `prometheus`

Only present when metrics are enabled:

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

---

## Notes about prompt padding

Each benchmark prompt is extended with a neutral padding block.

The padding is designed not to change the task meaning:

```text
The following text is neutral benchmark padding. Do not use it to answer the task.
It exists only to increase prompt length for prompt-processing speed measurement.
```

The `--pad-tokens` value is approximate because the script does not use the model tokenizer locally.

Always check:

```text
mean_prompt_tokens
```

in the aggregate output to see the real prompt-token count reported by the server.

---

## Troubleshooting

### `ERROR: --model is required`

List models first:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --list-models
```

Then pass the exact model ID:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model <MODEL_ID>
```

### `The model does not exist`

The model name passed to `--model` does not match the server’s model ID.

Check:

```bash
curl http://127.0.0.1:8000/v1/models
```

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

### `HTTP 400` or `HTTP 422` with llama.cpp

Older llama.cpp builds may reject `stream_options.include_usage`.

Use:

```bash
--no-stream-usage
```

Example:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8080 \
  --model local-model \
  --no-stream-usage
```

### `prompt_tokens` is zero

The server did not return usage information.

Possible causes:

```text
stream_options.include_usage is unsupported
the server does not expose token usage in streaming responses
older llama.cpp build
```

Try:

```bash
--include-timings
```

or:

```bash
--metrics-backend generic --list-metrics
```

### Prometheus metrics are empty

Check manually:

```bash
curl -s http://127.0.0.1:8000/metrics | head
```

For vLLM MTP metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'spec_decode|draft|accepted'
```

For llama.cpp, make sure the server was started with:

```bash
--metrics
```

### `total_draft` is zero

This may mean:

```text
MTP/speculative decoding is disabled
or
the OpenAI response does not expose MTP stats
```

For vLLM, check Prometheus:

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model <MODEL_ID> \
  --prometheus \
  --out mtp-prom.json
```

Then inspect:

```text
prometheus_total_draft_tokens_delta
prometheus_total_accepted_tokens_delta
prometheus_accept_rate
```

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

This means:

```text
sum_request_wall_s
```

is the sum of all individual request durations.

```text
suite_wall_s
```

is the actual elapsed benchmark time.

For throughput under load, use:

```text
wall_clock_prompt_tokens_per_second
wall_clock_generation_tokens_per_second
```

not per-request prompt tok/s.

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

---

## Limitations

This is a client-side benchmark. It cannot directly observe internal server scheduling unless the server exposes metrics.

Important limitations:

* TTFT is not pure prefill time.
* TTFT includes queueing under concurrency.
* Prompt tok/s estimate is reliable only at `parallel=1`.
* Response-body MTP stats may be unavailable.
* Prometheus metrics are server-global counters.
* Prometheus deltas include other traffic if the server is not idle.
* llama.cpp metric names may vary by version.
* Padding token count is approximate.

For clean measurements, run against an otherwise idle inference server.

---

## Example benchmark workflow

### 1. Start vLLM

Example:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8
```

### 2. Verify server

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics | head
```

### 3. Run single-request baseline

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 1 \
  --pad-tokens 1200 \
  --prometheus \
  --out baseline.json
```

### 4. Run parallel test

```bash
python mtp_bench.py \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.6-35B-A3B \
  --parallel 4 \
  --pad-tokens 1200 \
  --prometheus \
  --out parallel4.json
```

### 5. Compare

```bash
python mtp_bench.py --diff baseline.json parallel4.json
```

---

## License

MIT
```

