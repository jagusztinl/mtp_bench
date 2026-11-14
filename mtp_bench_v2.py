#!/usr/bin/env python3
"""
MTP / speculative-decoding benchmark for OpenAI-compatible servers (vLLM, llama.cpp, SGLang, Ollama, oMLX).

Fixes applied relative to v1 (review findings):
  #1  prompt tok/s from TTFT is only valid at parallel=1 (TTFT includes queueing
      under concurrency). Per-request estimates are marked unreliable and the
      aggregate is set to None when parallel > 1.
  #2  Documented: TTFT also includes chat-template processing and the first
      token's decode step, so the prefill-speed estimate is biased slightly low,
      especially with short prompts.
  #3  final_timings is now always a fresh dict copy, so later .update() calls
      can no longer mutate chunks already stored in raw_timing_chunks.
  #4  The Prometheus parser now skips histogram *_bucket series (summing
      cumulative buckets across labels produces garbage). Per-position
      acceptance vectors are summed knowingly; noted in output.
  #5  --no-stream-usage flag, plus automatic one-time retry without
      stream_options if the server rejects it (older llama.cpp builds).
  #6  Per-request output now separates server-reported predicted_per_second
      (from llama.cpp `timings`) from the client-side estimate, so rows and
      aggregates are no longer on silently different bases.
  #7  Worker errors of any Exception type are caught in both the serial and
      thread-pool paths (no raw tracebacks from timeouts / JSON errors).
  #9  Aggregate reports actual mean prompt tokens vs. the requested pad target.
  #10 Counter deltas are clamped at 0 with a warning if after < before
      (server restart between scrapes).
  #11 diff() prints n/a instead of fake zeros when a prompt is missing in A.
  #12 When both `<name>` and `<name>_total` exist, only `_total` is reported
      in the selected-deltas JSON (no double reporting).
"""
import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from urllib import request, error

BASE_PROMPTS = [
    {
        "name": "code_python",
        "prompt": "Write a Python function that returns the n-th Fibonacci number using memoization. Include a docstring.",
    },
    {
        "name": "code_cpp",
        "prompt": "Write a C++ template function `clamp(x, lo, hi)` that returns x clamped to [lo, hi]. No std::clamp.",
    },
    {
        "name": "explain_concept",
        "prompt": "Explain how speculative decoding works in large language model inference, in three short paragraphs.",
    },
    {
        "name": "summarize",
        "prompt": (
            "Summarize in two sentences: The Industrial Revolution began in Britain "
            "in the late 18th century, transforming manufacturing through mechanization, "
            "steam power, and the factory system. It spread to continental Europe and "
            "North America during the 19th century."
        ),
    },
    {
        "name": "qa_factual",
        "prompt": "Q: What are the four fundamental forces of physics?\nA:",
    },
    {
        "name": "translation",
        "prompt": "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    },
    {
        "name": "creative_short",
        "prompt": "Write a four-line poem about an old lighthouse.",
    },
    {
        "name": "stepwise_math",
        "prompt": (
            "Solve step by step: A train leaves station A at 60 km/h. "
            "Two hours later, a second train leaves the same station on the same track "
            "at 90 km/h. How long until the second train catches the first?"
        ),
    },
    {
        "name": "long_code_review",
        "prompt": (
            "You are reviewing a backend service that has been suffering intermittent latency spikes "
            "in production. Produce a structured review with three sections: "
            "(1) likely root causes ranked by probability, "
            "(2) concrete code or configuration changes you would make first, "
            "(3) what telemetry you would add to confirm the diagnosis.\n\n"
            "System description: Python FastAPI service, Postgres 15, four replicas behind nginx. "
            "Each request reads a user record, fetches the last 50 events, computes a score, "
            "updates the user row, inserts score history, and returns JSON. "
            "p50 latency is 35 ms; p99 spikes to 1.8 seconds every 90 seconds. "
            "Spikes correlate with elevated Postgres CPU. PgBouncer is in transaction pooling mode. "
            "The events table has 2B rows and is partitioned monthly. "
            "The score_history table has 800M rows and is unpartitioned. "
            "Begin your review now."
        ),
    },
]

PAD_SENTENCE = (
    "This neutral benchmark padding sentence is intentionally ignored by the task and only increases "
    "the input context length for measuring prompt prefill throughput. "
)

VLLM_SPEC_METRICS = [
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_total",
    # NOTE (#4): per-position acceptance is a labelled vector; the parser sums
    # all positions into one number. Useful as a total, but per-position
    # breakdown is intentionally not preserved.
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:spec_decode_num_emitted_tokens",
    "vllm:spec_decode_num_emitted_tokens_total",
]

VLLM_GENERAL_METRICS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]

LLAMACPP_INTERESTING_SUBSTRINGS = [
    "llamacpp",
    "llama",
    "prompt",
    "tokens",
    "token",
    "generation",
    "decode",
    "prefill",
    "kv",
    "cache",
    "slot",
    "request",
    "processing",
    "t_prompt",
    "t_eval",
    "n_prompt",
    "n_eval",
]

# Metric name suffixes that indicate monotonic counters. Negative deltas for
# these mean the server restarted between scrapes (#10).
COUNTER_SUFFIXES = ("_total", "_sum", "_count")


def make_padding(target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    approx_tokens_per_sentence = 18  # rough; actual prompt_tokens reported in aggregate (#9)
    sentence_count = max(1, int(target_tokens / approx_tokens_per_sentence) + 10)
    body = PAD_SENTENCE * sentence_count
    return (
        "\n\n"
        "----- BEGIN BENCHMARK PADDING -----\n"
        "The following text is neutral benchmark padding. Do not use it to answer the task. "
        "It exists only to increase prompt length for prompt-processing speed measurement.\n\n"
        f"{body}\n"
        "----- END BENCHMARK PADDING -----\n"
    )


def build_prompts(pad_tokens: int):
    padding = make_padding(pad_tokens)
    return [{"name": item["name"], "prompt": item["prompt"] + padding} for item in BASE_PROMPTS]


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}. Use for example: http://127.0.0.1:8000")
    # Accept either http://host:port or http://host:port/v1 as --url.
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def auth_headers(api_key=None, accept=None):
    headers = {}
    if accept:
        headers["Accept"] = accept
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def redact_error(value, *secrets):
    result = str(value)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def get_json(url, timeout=30, api_key=None):
    try:
        req = request.Request(url, headers=auth_headers(api_key, "application/json"))
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(f"HTTP {exc.code}: model listing authentication failed (check API key)") from None
        raise RuntimeError(f"HTTP {exc.code}: model listing failed") from None
    except error.URLError as exc:
        raise RuntimeError(f"Model listing connection failed: {redact_error(exc.reason, api_key)}") from None
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid model listing response: {redact_error(exc, api_key)}") from None


def get_available_models(base_url, api_key=None):
    data = get_json(f"{base_url}/v1/models", api_key=api_key)
    if not data:
        return []
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


def first_present(mapping, keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def safe_float(value, default=0.0):
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def parse_prometheus_text(text: str):
    """
    Very small Prometheus text parser.
    Sums all series with the same metric name, ignoring labels.

    #4: Histogram *_bucket series are skipped entirely. Buckets are cumulative,
    so summing them across labels/le boundaries produces meaningless numbers.
    _sum and _count series are kept (they are ordinary counters).
    """
    metrics = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_with_labels = parts[0]
        value_text = parts[1]
        if "{" in metric_with_labels:
            metric_name = metric_with_labels.split("{", 1)[0]
        else:
            metric_name = metric_with_labels
        if metric_name.endswith("_bucket"):  # (#4)
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        metrics[metric_name] = metrics.get(metric_name, 0.0) + value
    return metrics


def scrape_prometheus_metrics(metrics_url: str, timeout: int = 10, api_key=None):
    try:
        req = request.Request(metrics_url, headers=auth_headers(api_key, "text/plain"))
        with request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        raise RuntimeError(f"Metrics endpoint failed: HTTP {exc.code}") from None
    except error.URLError as exc:
        raise RuntimeError(f"Metrics endpoint connection failed: {redact_error(exc.reason, api_key)}") from None
    return parse_prometheus_text(text), text


_restart_warned = False


def metric_delta(before, after, name):
    """Counter-aware delta. Clamps negative counter deltas to 0 and warns once (#10)."""
    global _restart_warned
    delta = after.get(name, 0.0) - before.get(name, 0.0)
    if delta < 0 and (name.endswith(COUNTER_SUFFIXES) or name not in before):
        if not _restart_warned:
            print(
                f"WARNING: negative delta for counter '{name}' "
                f"(server likely restarted between scrapes); clamping to 0.",
                file=sys.stderr,
            )
            _restart_warned = True
        return 0.0
    return delta


def get_first_delta(before, after, names):
    for name in names:
        if name in before or name in after:
            return metric_delta(before, after, name), name
    return 0.0, None


def selected_metric_deltas(before, after, names):
    """#12: when both `<name>` and `<name>_total` are present, report only `_total`."""
    present = [n for n in names if n in before or n in after]
    present_set = set(present)
    out = {}
    for name in present:
        if (name + "_total") in present_set:
            continue  # skip base variant; _total will be reported
        out[name] = metric_delta(before, after, name)
    return out


def generic_interesting_deltas(before, after, substrings):
    out = {}
    all_names = sorted(set(before.keys()) | set(after.keys()))
    for name in all_names:
        lname = name.lower()
        if any(s.lower() in lname for s in substrings):
            delta = metric_delta(before, after, name)
            if delta != 0:
                out[name] = delta
    return out


def calculate_vllm_prometheus_delta(before, after):
    draft_tokens, draft_tokens_metric = get_first_delta(
        before, after,
        ["vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"],
    )
    accepted_tokens, accepted_tokens_metric = get_first_delta(
        before, after,
        ["vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"],
    )
    drafts, drafts_metric = get_first_delta(
        before, after,
        ["vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"],
    )
    emitted_tokens, emitted_tokens_metric = get_first_delta(
        before, after,
        ["vllm:spec_decode_num_emitted_tokens_total", "vllm:spec_decode_num_emitted_tokens"],
    )
    accept_rate = None
    if draft_tokens > 0:
        accept_rate = accepted_tokens / draft_tokens
    raw_deltas = selected_metric_deltas(before, after, VLLM_SPEC_METRICS + VLLM_GENERAL_METRICS)
    return {
        "metrics_backend": "vllm",
        "prometheus_total_drafts_delta": int(round(drafts)) if drafts_metric else None,
        "prometheus_total_draft_tokens_delta": int(round(draft_tokens)) if draft_tokens_metric else None,
        "prometheus_total_accepted_tokens_delta": int(round(accepted_tokens)) if accepted_tokens_metric else None,
        "prometheus_total_emitted_tokens_delta": int(round(emitted_tokens)) if emitted_tokens_metric else None,
        "prometheus_accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "prometheus_metric_names_used": {
            "drafts": drafts_metric,
            "draft_tokens": draft_tokens_metric,
            "accepted_tokens": accepted_tokens_metric,
            "emitted_tokens": emitted_tokens_metric,
        },
        "prometheus_vllm_selected_deltas": raw_deltas,
        "prometheus_note": "per_pos metrics are summed across positions; _bucket series are excluded",
    }


def calculate_llamacpp_prometheus_delta(before, after):
    interesting = generic_interesting_deltas(before, after, LLAMACPP_INTERESTING_SUBSTRINGS)
    prompt_tokens, prompt_metric = get_first_delta(
        before, after,
        [
            "llamacpp:prompt_tokens_total", "llama:prompt_tokens_total",
            "llamacpp_prompt_tokens_total", "llama_prompt_tokens_total",
            "prompt_tokens_total",
            "llamacpp:n_prompt_total", "llama:n_prompt_total",
            "llamacpp_n_prompt_total", "llama_n_prompt_total",
        ],
    )
    generation_tokens, generation_metric = get_first_delta(
        before, after,
        [
            "llamacpp:generation_tokens_total", "llama:generation_tokens_total",
            "llamacpp_generation_tokens_total", "llama_generation_tokens_total",
            "generation_tokens_total", "completion_tokens_total",
            "llamacpp:n_eval_total", "llama:n_eval_total",
            "llamacpp_n_eval_total", "llama_n_eval_total",
        ],
    )
    requests, request_metric = get_first_delta(
        before, after,
        [
            "llamacpp:requests_total", "llama:requests_total",
            "llamacpp_requests_total", "llama_requests_total",
            "requests_total",
        ],
    )
    return {
        "metrics_backend": "llamacpp",
        "llamacpp_prompt_tokens_delta_best_effort": int(round(prompt_tokens)),
        "llamacpp_generation_tokens_delta_best_effort": int(round(generation_tokens)),
        "llamacpp_requests_delta_best_effort": int(round(requests)),
        "llamacpp_metric_names_used": {
            "prompt_tokens": prompt_metric,
            "generation_tokens": generation_metric,
            "requests": request_metric,
        },
        "llamacpp_interesting_metric_deltas": interesting,
    }


SGLANG_SPEC_GAUGES = (
    "sglang:spec_accept_rate",
    "sglang:spec_accept_length",
    "sglang:spec_cap_length",
    "sglang:spec_block_accept_length",
    "sglang:spec_num_steps",
    "sglang:spec_num_draft_tokens",
)


def calculate_sglang_prometheus_delta(before, after):
    # SGLang's speculative decoding metrics are Gauges, not cumulative counters.
    # Report snapshots, NEVER subtract the previous value or call it test-wide acceptance.
    gauges_before = {key: before[key] for key in SGLANG_SPEC_GAUGES if key in before}
    gauges_after = {key: after[key] for key in SGLANG_SPEC_GAUGES if key in after}
    counters = {}
    for name in sorted(set(before) | set(after)):
        if name.startswith("sglang:") and name.endswith(COUNTER_SUFFIXES):
            counters[name] = metric_delta(before, after, name)
    return {
        "metrics_backend": "sglang",
        "sglang_spec_gauges_before": gauges_before,
        "sglang_spec_gauges_after": gauges_after,
        "sglang_counter_deltas": counters,
        "prometheus_note": (
            "SGLang speculative metrics are gauges. The before/after values are "
            "snapshots, not run-wide or per-request acceptance rates. "
            "Histograms' cumulative buckets are excluded."
        ),
    }


def calculate_generic_prometheus_delta(before, after):
    all_names = sorted(set(before.keys()) | set(after.keys()))
    deltas = {}
    for name in all_names:
        delta = metric_delta(before, after, name)
        if delta != 0:
            deltas[name] = delta
    return {
        "metrics_backend": "generic",
        "prometheus_all_nonzero_deltas": deltas,
        "prometheus_note": "_bucket series are excluded; counters clamped at 0 on restart",
    }


def calculate_prometheus_delta(before, after, backend):
    if backend == "vllm":
        return calculate_vllm_prometheus_delta(before, after)
    if backend == "llamacpp":
        return calculate_llamacpp_prometheus_delta(before, after)
    if backend == "sglang":
        return calculate_sglang_prometheus_delta(before, after)
    if backend == "generic":
        return calculate_generic_prometheus_delta(before, after)
    return {}


def post_chat_stream(url, payload, timeout=300, use_stream_usage=True, api_key=None):
    """
    Streams a chat completion and measures TTFT / generation time.

    #2 NOTE: TTFT includes chat-template application and the decode of the
    first token, not just prefill; the derived prompt tok/s estimate is
    therefore biased slightly LOW, more visibly with short prompts.

    #5: if the server rejects the request while stream_options is set
    (older llama.cpp builds reject unknown fields), retry once without it.
    """
    payload = dict(payload)
    payload["stream"] = True
    if use_stream_usage:
        payload["stream_options"] = {"include_usage": True}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **auth_headers(api_key, "text/event-stream")},
        method="POST",
    )
    t0 = time.perf_counter()
    first_chunk_time = None
    first_text_time = None
    full_text = ""
    final_usage = {}
    final_timings = {}
    raw_timing_chunks = []
    try:
        with request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_line = line[len("data:"):].strip()
                if data_line == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    chunk = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                if first_chunk_time is None:
                    first_chunk_time = now
                usage = chunk.get("usage")
                if usage:
                    final_usage = usage
                timings = chunk.get("timings")
                if timings:
                    final_timings = dict(timings)  # (#3) copy: never alias chunk data
                    raw_timing_chunks.append(dict(timings))
                top_level_timing_keys = [
                    "predicted_n", "predicted_per_second",
                    "draft_n", "draft_n_accepted",
                    "prompt_n", "prompt_per_second",
                ]
                top_level_timings = {k: chunk[k] for k in top_level_timing_keys if k in chunk}
                if top_level_timings:
                    final_timings.update(top_level_timings)  # safe: final_timings is our own copy
                    raw_timing_chunks.append(top_level_timings)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token_text = delta.get("content") or delta.get("reasoning_content") or ""
                if token_text:
                    if first_text_time is None:
                        first_text_time = now
                    full_text += str(token_text)
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # (#5) one-time fallback without stream_options
        mentions_unsupported_stream_options = (
            "stream_options" in body.lower()
            and any(hint in body.lower() for hint in (
                "unsupported", "unexpected", "unknown", "unrecognized", "extra", "invalid"
            ))
        )
        # Do not repeat failed requests on unrelated server errors (OOM etc.).
        if use_stream_usage and e.code in (400, 422, 500) and mentions_unsupported_stream_options:
            return post_chat_stream(url, {k: v for k, v in payload.items() if k != "stream_options"},
                                    timeout=timeout, use_stream_usage=False, api_key=api_key)
        if e.code in (401, 403):
            raise RuntimeError(f"HTTP {e.code}: authentication or authorization failed") from None
        raise RuntimeError(f"HTTP {e.code} from chat completion endpoint\nResponse body:\n"
                           f"{redact_error(body[:2000], api_key)}") from None
    except error.URLError as e:
        raise RuntimeError(f"Chat endpoint connection failed: {redact_error(e.reason, api_key)}") from None

    t_end = time.perf_counter()
    wall_s = t_end - t0
    first_event_time = first_text_time or first_chunk_time
    ttft_s = None if first_event_time is None else first_event_time - t0
    generation_s = None if first_event_time is None else t_end - first_event_time
    return {
        "text": full_text,
        "usage": final_usage,
        "timings": final_timings,
        "raw_timing_chunks": raw_timing_chunks,
        "wall_s": wall_s,
        "ttft_s": ttft_s,
        "generation_s": generation_s,
    }


def run_one_prompt(base_url, args, prompt_item, index):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_item["prompt"]}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.seed is not None:
        payload["seed"] = args.seed + index
    response = post_chat_stream(
        f"{base_url}/v1/chat/completions",
        payload,
        timeout=args.timeout,
        use_stream_usage=not args.no_stream_usage,  # (#5)
        api_key=args.api_key,
    )
    usage = response.get("usage", {}) or {}
    timings = response.get("timings", {}) or {}
    wall_s = response["wall_s"]
    ttft_s = response["ttft_s"]
    generation_s = response["generation_s"]

    # Most OpenAI-compatible servers return usage in the final stream event.
    # llama.cpp sometimes supplies server timings without an OpenAI usage block.
    prompt_from_usage = usage.get("prompt_tokens")
    completion_from_usage = usage.get("completion_tokens")
    prompt_from_timings = first_present(timings, ["prompt_n", "prompt_tokens", "num_prompt_tokens"])
    completion_from_timings = first_present(timings, ["predicted_n", "completion_tokens", "num_predicted_tokens"])
    prompt_tokens = prompt_from_usage if prompt_from_usage is not None else (prompt_from_timings or 0)
    completion_tokens = completion_from_usage if completion_from_usage is not None else (completion_from_timings or 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    token_usage_source = (
        "usage" if prompt_from_usage is not None and completion_from_usage is not None
        else "timings" if prompt_from_timings is not None or completion_from_timings is not None
        else "unavailable"
    )

    predicted_n = first_present(
        timings,
        ["predicted_n", "predicted_tokens", "num_predicted_tokens", "completion_tokens"],
        completion_tokens,
    ) or 0

    predicted_per_second_server = first_present(
        timings,
        ["predicted_per_second", "predicted_tokens_per_second",
         "completion_tokens_per_second", "generation_tokens_per_second"],
        None,
    )
    if predicted_per_second_server is not None:
        predicted_per_second_server = safe_float(predicted_per_second_server)

    draft_n = first_present(
        timings,
        ["draft_n", "draft_tokens", "num_draft_tokens",
         "num_speculative_tokens", "speculative_tokens"],
        None,
    )
    draft_n_accepted = first_present(
        timings,
        ["draft_n_accepted", "accepted_draft_tokens", "num_accepted_tokens",
         "num_accepted_draft_tokens", "accepted_speculative_tokens",
         "speculative_tokens_accepted"],
        None,
    )

    # (#1/#2) prompt tok/s from TTFT: only reliable at parallel=1
    # (TTFT includes queueing under concurrency, plus template + first decode step).
    if ttft_s and prompt_tokens:
        prompt_tokens_per_second = prompt_tokens / ttft_s
    else:
        prompt_tokens_per_second = None

    if generation_s and completion_tokens:
        generation_tokens_per_second = completion_tokens / generation_s
    elif wall_s and completion_tokens:
        generation_tokens_per_second = completion_tokens / wall_s
    else:
        generation_tokens_per_second = 0.0

    # (#6) client-side estimate kept separate from server-reported value
    if generation_s and predicted_n:
        predicted_per_second_client = predicted_n / generation_s
    elif wall_s and predicted_n:
        predicted_per_second_client = predicted_n / wall_s
    else:
        predicted_per_second_client = 0.0
    predicted_per_second = (
        predicted_per_second_server
        if predicted_per_second_server is not None
        else predicted_per_second_client
    )

    accept_rate = None
    if draft_n and draft_n_accepted is not None:
        accept_rate = draft_n_accepted / draft_n

    rec = {
        "index": index,
        "name": prompt_item["name"],
        "wall_s": round(wall_s, 3),
        "ttft_s": round(ttft_s, 6) if ttft_s is not None else None,
        "generation_s": round(generation_s, 6) if generation_s is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_source": token_usage_source,
        "predicted_n": int(predicted_n),
        "predicted_per_second": round(predicted_per_second, 2),
        "predicted_per_second_server": round(predicted_per_second_server, 2)
        if predicted_per_second_server is not None else None,
        "predicted_per_second_client": round(predicted_per_second_client, 2),
        "draft_n": int(draft_n) if draft_n is not None else None,
        "draft_n_accepted": int(draft_n_accepted) if draft_n_accepted is not None else None,
        "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
        "prompt_tokens_per_second_estimate": round(prompt_tokens_per_second, 2)
        if prompt_tokens_per_second is not None else None,
        "prompt_tps_estimate_reliable": args.parallel <= 1,  # (#1)
        "generation_tokens_per_second": round(generation_tokens_per_second, 2),
        "timings_returned": bool(timings),
    }
    if args.include_timings:
        rec["timings"] = timings
        rec["raw_timing_chunks"] = response.get("raw_timing_chunks", [])
    if args.include_text:
        rec["text"] = response["text"]
    return rec


def print_result(rec):
    prompt_tps = (
        f"{rec['prompt_tokens_per_second_estimate']:.2f}"
        if rec["prompt_tokens_per_second_estimate"] is not None else "n/a"
    )
    if not rec.get("prompt_tps_estimate_reliable", True) and prompt_tps != "n/a":
        prompt_tps += "*"  # (#1) unreliable under concurrency
    ttft_txt = f"{rec['ttft_s']:.3f}s" if rec["ttft_s"] is not None else "n/a"
    gen_s_txt = f"{rec['generation_s']:.3f}s" if rec["generation_s"] is not None else "n/a"
    rate_txt = f"{rec['accept_rate']:.3f}" if rec["accept_rate"] is not None else "n/a"
    draft_txt = rec["draft_n"] if rec["draft_n"] is not None else "n/a"
    accepted_txt = rec["draft_n_accepted"] if rec["draft_n_accepted"] is not None else "n/a"
    print(
        f"  {rec['name']:<18} "
        f"prompt={rec['prompt_tokens']:>6} "
        f"out={rec['completion_tokens']:>4} "
        f"pred={rec['predicted_n']:>4} "
        f"pred_tok/s={rec['predicted_per_second']:>8.2f} "
        f"draft={draft_txt:>4} "
        f"acc={accepted_txt:>4} "
        f"rate={rate_txt:>6} "
        f"ttft={ttft_txt:>8} "
        f"prompt_tok/s={prompt_tps:>10} "
        f"gen_tok/s={rec['generation_tokens_per_second']:>8.2f} "
        f"gen_s={gen_s_txt:>8} "
        f"wall={rec['wall_s']:>7.3f}s"
    )


def calculate_aggregate(results, suite_wall_s, parallel, pad_tokens_target):
    total_prompt_tokens = sum(x["prompt_tokens"] for x in results)
    total_completion_tokens = sum(x["completion_tokens"] for x in results)
    total_tokens = sum(x["total_tokens"] for x in results)
    total_predicted = sum(x["predicted_n"] or 0 for x in results)
    draft_records = [x for x in results if x["draft_n"] is not None]
    accepted_records = [x for x in results if x["draft_n"] is not None and x["draft_n_accepted"] is not None]
    total_draft = sum(x["draft_n"] for x in draft_records) if draft_records else None
    total_draft_accepted = (
        sum(x["draft_n_accepted"] for x in accepted_records) if accepted_records else None
    )

    aggregate_accept_rate = None
    if total_draft and total_draft_accepted is not None and len(accepted_records) == len(draft_records):
        aggregate_accept_rate = total_draft_accepted / total_draft

    sum_wall_s = sum(x["wall_s"] for x in results)
    total_ttft_s = sum(x["ttft_s"] or 0 for x in results)
    total_generation_s = sum(x["generation_s"] or 0 for x in results)

    # (#1) TTFT-based prompt tps aggregates are only meaningful at parallel=1;
    # under concurrency, per-request TTFT intervals overlap and include queueing.
    aggregate_prompt_tps = None
    if parallel <= 1 and total_ttft_s > 0 and total_prompt_tokens > 0:
        aggregate_prompt_tps = total_prompt_tokens / total_ttft_s

    aggregate_generation_tps = None
    aggregate_predicted_per_second = None
    if parallel <= 1:
        if total_generation_s > 0 and total_completion_tokens > 0:
            aggregate_generation_tps = total_completion_tokens / total_generation_s
        if total_generation_s > 0 and total_predicted > 0:
            aggregate_predicted_per_second = total_predicted / total_generation_s

    # Wall-clock throughput remains valid at any concurrency.
    wall_clock_prompt_tps = None
    if suite_wall_s > 0 and total_prompt_tokens > 0:
        wall_clock_prompt_tps = total_prompt_tokens / suite_wall_s
    wall_clock_generation_tps = None
    if suite_wall_s > 0 and total_completion_tokens > 0:
        wall_clock_generation_tps = total_completion_tokens / suite_wall_s
    wall_clock_predicted_tps = None
    if suite_wall_s > 0 and total_predicted > 0:
        wall_clock_predicted_tps = total_predicted / suite_wall_s

    mean_prompt_tokens = total_prompt_tokens / len(results) if results else 0  # (#9)

    return {
        "n_requests": len(results),
        "n_requests_missing_token_usage": sum(1 for r in results if r.get("token_usage_source") == "unavailable"),
        "n_requests_with_draft_stats": len(draft_records),
        "parallel": parallel,
        "total_prompt_tokens": total_prompt_tokens,
        "mean_prompt_tokens": round(mean_prompt_tokens, 1),
        "pad_tokens_target_approx": pad_tokens_target,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_predicted": total_predicted,
        "total_draft": total_draft,
        "total_draft_accepted": total_draft_accepted,
        "aggregate_accept_rate": round(aggregate_accept_rate, 4)
        if aggregate_accept_rate is not None else None,
        "sum_request_wall_s": round(sum_wall_s, 3),
        "suite_wall_s": round(suite_wall_s, 3),
        "ttft_s_total": round(total_ttft_s, 6),
        "generation_s_total": round(total_generation_s, 6),
        "aggregate_prompt_tokens_per_second_estimate": round(aggregate_prompt_tps, 2)
        if aggregate_prompt_tps is not None else None,
        "aggregate_generation_tokens_per_second": round(aggregate_generation_tps, 2)
        if aggregate_generation_tps is not None else None,
        "aggregate_predicted_per_second": round(aggregate_predicted_per_second, 2)
        if aggregate_predicted_per_second is not None else None,
        "per_request_aggregates_note": (
            "aggregate_* per-request rates are None when parallel > 1 "
            "(TTFT/generation intervals overlap and include queueing); "
            "use wall_clock_* throughput instead."
        ) if parallel > 1 else None,
        "wall_clock_prompt_tokens_per_second": round(wall_clock_prompt_tps, 2)
        if wall_clock_prompt_tps is not None else None,
        "wall_clock_generation_tokens_per_second": round(wall_clock_generation_tps, 2)
        if wall_clock_generation_tps is not None else None,
        "wall_clock_predicted_per_second": round(wall_clock_predicted_tps, 2)
        if wall_clock_predicted_tps is not None else None,
    }


def resolve_metrics_backend(args):
    if args.prometheus:
        return "vllm"
    return args.metrics_backend


def resolve_metrics_url(base_url, args, backend):
    if args.metrics_url:
        return args.metrics_url.strip()
    if backend == "none":
        return None
    return f"{base_url}/metrics"


def run(args):
    base_url = normalize_base_url(args.url)
    # Select a dedicated key for the metrics endpoint; avoid leaking inference
    # credentials to an unrelated scraper endpoint by default.
    metrics_key = args.metrics_api_key or (args.api_key if args.metrics_use_api_key else None)
    metrics_backend = resolve_metrics_backend(args)
    metrics_url = resolve_metrics_url(base_url, args, metrics_backend)

    if args.list_models:
        models = get_available_models(base_url, api_key=args.api_key)
        if not models:
            print(f"No models returned from {base_url}/v1/models")
            return
        print("Available models:")
        for model in models:
            print(f"  {model}")
        return

    if args.list_metrics:
        if metrics_backend == "none":
            metrics_backend = "generic"
            metrics_url = resolve_metrics_url(base_url, args, metrics_backend)
        metrics, raw = scrape_prometheus_metrics(metrics_url, timeout=args.metrics_timeout,
                                                  api_key=metrics_key)
        print(f"Scraped {len(metrics)} metric names from {metrics_url}")
        for name in sorted(metrics.keys()):
            print(f"{name} {metrics[name]}")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(
                    {"metrics_url": metrics_url, "metrics_backend": metrics_backend, "metrics": metrics},
                    f, indent=2,
                )
            print(f"\nWrote {args.out}")
        return

    if not args.model:
        models = get_available_models(base_url, api_key=args.api_key)
        print("ERROR: --model is required.")
        if models:
            print("\nAvailable models:")
            for model in models:
                print(f"  {model}")
            print("\nExample:")
            print(f"  python mtp_bench.py --url {base_url} --model {models[0]}")
        else:
            print(f"\nCould not fetch models from {base_url}/v1/models")
        sys.exit(2)

    prompts = build_prompts(args.pad_tokens)
    out = {
        "config": {
            "url": base_url,
            "backend": args.backend,
            "api_key_configured": bool(args.api_key),
            "metrics_api_key_configured": bool(metrics_key),
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "parallel": args.parallel,
            "pad_tokens_approx": args.pad_tokens,
            "stream_usage": not args.no_stream_usage,
            "method": "streaming_ttft_prompt_processing_estimate_with_mtp_stats",
            "method_caveats": (
                "TTFT includes chat-template processing and first-token decode, so the "
                "prompt tok/s estimate is biased slightly low. TTFT-based estimates are "
                "only reliable at parallel=1."
            ),
            "metrics_backend": metrics_backend,
            "metrics_url": metrics_url,
        },
        "results": [],
    }

    metrics_before = {}
    metrics_after = {}
    if metrics_backend != "none":
        print(f"Scraping {metrics_backend} metrics before run: {metrics_url}")
        metrics_before, _ = scrape_prometheus_metrics(metrics_url, timeout=args.metrics_timeout,
                                                       api_key=metrics_key)

    print(
        f"Running {len(prompts)} requests with parallel={args.parallel}, "
        f"pad_tokens_approx={args.pad_tokens}, max_tokens={args.max_tokens}"
    )
    if args.parallel > 1:
        print("NOTE: parallel > 1 — TTFT-based prompt tok/s values are marked '*' and "
              "excluded from per-request aggregates; use wall_clock_* throughput.")

    suite_t0 = time.perf_counter()

    def handle_failure(prompt_item, exc):
        print(f"\nRequest failed for prompt: {prompt_item['name']}")
        print(redact_error(exc, args.api_key, metrics_key))
        if "does not exist" in str(exc) or "NotFoundError" in str(exc):
            models = get_available_models(base_url, api_key=args.api_key)
            if models:
                print("\nAvailable models:")
                for model in models:
                    print(f"  {model}")
        sys.exit(1)

    if args.parallel <= 1:
        results = []
        for i, prompt_item in enumerate(prompts):
            try:
                rec = run_one_prompt(base_url, args, prompt_item, i)
            except Exception as e:  # (#7) catch all worker errors, not just RuntimeError
                handle_failure(prompt_item, e)
            results.append(rec)
            print_result(rec)
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_one_prompt, base_url, args, prompt_item, i): prompt_item
                for i, prompt_item in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(futures):
                prompt_item = futures[future]
                try:
                    rec = future.result()
                except Exception as e:  # (#7)
                    handle_failure(prompt_item, e)
                results.append(rec)
                print_result(rec)
        results.sort(key=lambda x: x["index"])

    suite_wall_s = time.perf_counter() - suite_t0

    if metrics_backend != "none":
        print(f"Scraping {metrics_backend} metrics after run: {metrics_url}")
        metrics_after, _ = scrape_prometheus_metrics(metrics_url, timeout=args.metrics_timeout,
                                                      api_key=metrics_key)
        out["prometheus"] = calculate_prometheus_delta(metrics_before, metrics_after, metrics_backend)
        if args.include_metrics_raw:
            out["prometheus"]["before"] = metrics_before
            out["prometheus"]["after"] = metrics_after

    out["results"] = results
    out["aggregate"] = calculate_aggregate(results, suite_wall_s, args.parallel, args.pad_tokens)
    if "prometheus" in out:
        out["aggregate"].update(out["prometheus"])

    print("\nAggregate:")
    print(json.dumps(out["aggregate"], indent=2))
    if "prometheus" in out:
        print("\nPrometheus metrics delta:")
        print(json.dumps(out["prometheus"], indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


def _fmt_cell(value, width, float_fmt="{:>{w}.2f}"):
    """(#11) format a numeric cell or n/a for missing values."""
    if value is None:
        return f"{'n/a':>{width}}"
    if isinstance(value, float):
        return float_fmt.format(value, w=width)
    return f"{value:>{width}}"


def diff(a, b):
    with open(a, "r", encoding="utf-8") as fa:
        A = json.load(fa)
    with open(b, "r", encoding="utf-8") as fb:
        B = json.load(fb)
    print(f"{'metric':<58} {'A':>14} {'B':>14} {'delta':>14}")
    keys = [
        "parallel",
        "total_prompt_tokens",
        "mean_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_predicted",
        "total_draft",
        "total_draft_accepted",
        "aggregate_accept_rate",
        "prometheus_total_drafts_delta",
        "prometheus_total_draft_tokens_delta",
        "prometheus_total_accepted_tokens_delta",
        "prometheus_total_emitted_tokens_delta",
        "prometheus_accept_rate",
        "llamacpp_prompt_tokens_delta_best_effort",
        "llamacpp_generation_tokens_delta_best_effort",
        "llamacpp_requests_delta_best_effort",
        "sum_request_wall_s",
        "suite_wall_s",
        "ttft_s_total",
        "generation_s_total",
        "aggregate_prompt_tokens_per_second_estimate",
        "aggregate_generation_tokens_per_second",
        "aggregate_predicted_per_second",
        "wall_clock_prompt_tokens_per_second",
        "wall_clock_generation_tokens_per_second",
        "wall_clock_predicted_per_second",
    ]
    for k in keys:
        va = A.get("aggregate", {}).get(k)
        vb = B.get("aggregate", {}).get(k)
        if va is None or vb is None:
            print(f"{k:<58} {str(va):>14} {str(vb):>14} {'n/a':>14}")
            continue
        try:
            d = vb - va
            ds = f"{d:+.4f}" if isinstance(d, float) else f"{d:+d}"
        except TypeError:
            ds = "n/a"
        print(f"{k:<58} {va:>14} {vb:>14} {ds:>14}")

    by_a = {x["name"]: x for x in A.get("results", [])}
    print()
    print(
        f"{'prompt':<20} "
        f"{'A pTok/s':>10} {'B pTok/s':>10} "
        f"{'A pred/s':>10} {'B pred/s':>10} "
        f"{'A draft':>10} {'B draft':>10} "
        f"{'A acc':>10} {'B acc':>10} "
        f"{'A rate':>10} {'B rate':>10}"
    )
    for rb in B.get("results", []):
        ra = by_a.get(rb["name"])  # (#11) None if prompt missing in A
        def g(rec, key):
            return rec.get(key) if rec else None
        print(
            f"{rb['name']:<20} "
            f"{_fmt_cell(g(ra, 'prompt_tokens_per_second_estimate'), 10)} "
            f"{_fmt_cell(rb.get('prompt_tokens_per_second_estimate'), 10)} "
            f"{_fmt_cell(g(ra, 'predicted_per_second'), 10)} "
            f"{_fmt_cell(rb.get('predicted_per_second'), 10)} "
            f"{_fmt_cell(g(ra, 'draft_n'), 10)} "
            f"{_fmt_cell(rb.get('draft_n'), 10)} "
            f"{_fmt_cell(g(ra, 'draft_n_accepted'), 10)} "
            f"{_fmt_cell(rb.get('draft_n_accepted'), 10)} "
            f"{_fmt_cell(g(ra, 'accept_rate'), 10, '{:>{w}.3f}')} "
            f"{_fmt_cell(rb.get('accept_rate'), 10, '{:>{w}.3f}')}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000",
                    help="Server URL, with or without /v1 (default: http://127.0.0.1:8000)")
    ap.add_argument("--backend", choices=["openai", "vllm", "llamacpp", "sglang", "ollama", "omlx"],
                    default="openai", help="Informational backend tag; all use OpenAI streaming API")
    ap.add_argument("--api-key", default=None,
                    help="Bearer token for /v1 endpoints; prefer an environment variable")
    ap.add_argument("--api-key-env", default="MTP_BENCH_API_KEY",
                    help="Name of environment variable providing API key (default: MTP_BENCH_API_KEY)")
    ap.add_argument("--model", help="Model name from /v1/models")
    ap.add_argument("--out", help="Write result JSON to file")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--pad-tokens", type=int, default=1200)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--include-text", action="store_true")
    ap.add_argument("--include-timings", action="store_true")
    ap.add_argument(
        "--no-stream-usage",
        action="store_true",
        help="Do not send stream_options.include_usage (for older llama.cpp builds).",  # (#5)
    )
    # Metrics options
    ap.add_argument("--prometheus", action="store_true",
                    help="Alias for --metrics-backend vllm. Scrape /metrics before and after the run.")
    ap.add_argument("--metrics-backend", choices=["none", "vllm", "llamacpp", "sglang", "generic"], default="none",
                    help="Metrics backend. vllm for vLLM /metrics, llamacpp for llama.cpp /metrics, "
                         "generic to dump all non-zero Prometheus deltas.")
    ap.add_argument("--metrics-url", default=None,
                    help="Metrics endpoint. Defaults to <url>/metrics when metrics backend is enabled.")
    ap.add_argument("--metrics-api-key", default=None,
                    help="Optional separate Bearer token for /metrics (or MTP_BENCH_METRICS_API_KEY env)")
    ap.add_argument("--metrics-use-api-key", action="store_true",
                    help="Explicitly reuse inference API key for /metrics")
    ap.add_argument("--metrics-timeout", type=int, default=10,
                    help="Timeout for scraping metrics endpoint.")
    ap.add_argument("--include-metrics-raw", action="store_true",
                    help="Include raw before/after metric snapshots in JSON output.")
    ap.add_argument("--list-metrics", action="store_true",
                    help="Scrape and list available metric names, then exit.")
    # With no CLI arguments, show the same usage information as --help.
    if len(sys.argv) == 1:
        ap.print_help()
        return
    args = ap.parse_args()
    args.api_key = args.api_key or os.environ.get(args.api_key_env) or None
    args.metrics_api_key = args.metrics_api_key or os.environ.get("MTP_BENCH_METRICS_API_KEY") or None
    if args.parallel < 1:
        print("ERROR: --parallel must be >= 1")
        sys.exit(2)
    if args.diff:
        diff(*args.diff)
    else:
        try:
            run(args)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {redact_error(e, args.api_key, args.metrics_api_key)}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()