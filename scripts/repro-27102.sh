#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

server=${SERVER:-"$repo_dir/build-x64-linux-gcc-reldbg/bin/llama-server"}
model=${MODEL:-/mnt/share/gguf/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}
host=${HOST:-127.0.0.1}
port=${PORT:-9999}
duration_min=${DURATION_MIN:-60}
clients=${CLIENTS:-6}
n_predict=${N_PREDICT:-16384}
client_delay_sec=${CLIENT_DELAY_SEC:-20}
ctx_size=${CTX_SIZE:-196608}
prompt_tokens=${PROMPT_TOKENS:-131072,114688,102400,81477,30366,2048}
log_dir=${LOG_DIR:-"$repo_dir/repro-27102-$(date +%Y%m%d-%H%M%S)"}

mkdir -p "$log_dir"

if [[ ! -x "$server" ]]; then
    echo "llama-server not found: $server" >&2
    exit 2
fi

if [[ ! -f "$model" ]]; then
    echo "model not found: $model" >&2
    exit 2
fi

if ! [[ "$client_delay_sec" =~ ^[0-9]+$ ]]; then
    echo "CLIENT_DELAY_SEC must be a non-negative integer" >&2
    exit 2
fi

if ! [[ "$ctx_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "CTX_SIZE must be a positive integer" >&2
    exit 2
fi

server_pid=""
worker_pids=()

cleanup() {
    trap - EXIT INT TERM
    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "logs: $log_dir"
echo "server: $server"
echo "model: $model"
echo "context size: $ctx_size"
echo "prompt tokens: $prompt_tokens"
echo "client delay: $client_delay_sec seconds"

GGML_CUDA_KERNEL_DIAGNOSTICS=1 \
GGML_CUDA_DISABLE_GRAPHS=1 \
CUDA_LAUNCH_BLOCKING=1 \
"$server" \
    -m "$model" \
    --load-mode dio \
    --ctx-size "$ctx_size" \
    --host "$host" \
    --port "$port" \
    --no-ui \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --chat-template-kwargs '{"preserve_thinking": true}' \
    --reasoning-preserve \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    --timeout 120 \
    --cors-origins localhost \
    --no-cors-credentials \
    -ctk q8_0 \
    -ctv q8_0 \
    -v \
    --log-file "$log_dir/server.log" \
    "$@" \
    >"$log_dir/server.stdout.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 600); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "llama-server exited during startup" >&2
        exit 1
    fi
    if curl -fsS "http://$host:$port/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" != 1 ]]; then
    echo "llama-server did not become ready" >&2
    exit 1
fi

python3 - "$repo_dir" "$log_dir" "$host" "$port" "$prompt_tokens" "$n_predict" <<'PY'
import json
import pathlib
import sys
import urllib.request

repo_dir = pathlib.Path(sys.argv[1])
log_dir = pathlib.Path(sys.argv[2])
host = sys.argv[3]
port = int(sys.argv[4])
lengths = [int(value) for value in sys.argv[5].split(",")]
n_predict = int(sys.argv[6])

workloads = [
    (
        "CUDA flash attention investigation",
        "Investigate a CUDA timeout in Q8 KV-cache flash attention. Trace the data flow, identify race and bounds risks, and propose focused diagnostics.",
        [
            "ggml/src/ggml-cuda/fattn.cu",
            "ggml/src/ggml-cuda/fattn-common.cuh",
            "ggml/src/ggml-cuda/fattn-mma-f16.cuh",
            "ggml/src/ggml-cuda/set-rows.cu",
            "ggml/src/ggml-cuda/concat.cu",
            "ggml/src/ggml-cuda/common.cuh",
        ],
    ),
    (
        "KV-cache concurrency analysis",
        "Review KV-cache behavior for concurrent requests with different sequence lengths. Find invariants that should hold during update, defragmentation, and context shifts.",
        [
            "src/llama-kv-cache.cpp",
            "src/llama-kv-cache.h",
            "src/llama-context.cpp",
            "src/llama-memory.cpp",
            "src/llama-memory-recurrent.cpp",
            "src/llama-batch.cpp",
        ],
    ),
    (
        "Server scheduling incident review",
        "Analyze a production incident where several long-running inference requests and new arrivals share server slots. Check scheduling, cancellation, and streaming behavior.",
        [
            "tools/server/README-dev.md",
            "tools/server/server-context.cpp",
            "tools/server/server-queue.cpp",
            "tools/server/server-task.cpp",
            "tools/server/server-stream.cpp",
            "tools/server/server-http.cpp",
        ],
    ),
    (
        "CUDA backend regression test design",
        "Design a minimal backend regression test for a nondeterministic CUDA stall. Explain which tensor shapes, strides, quantization types, and mixed sequence lengths must vary.",
        [
            "tests/test-backend-ops.cpp",
            "tests/testing.h",
            "docs/development/debugging-tests.md",
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            "ggml/src/ggml-cuda/mmq.cuh",
        ],
    ),
    (
        "Qwen model graph review",
        "Review the Qwen model graph for long-context generation with MTP speculative decoding. Trace recurrent state, attention inputs, and concatenation operations.",
        [
            "src/models/qwen.cpp",
            "src/llama-graph.cpp",
            "src/llama-graph.h",
            "src/llama-model.cpp",
            "src/llama-hparams.cpp",
            "src/llama-context.cpp",
        ],
    ),
    (
        "Chat template and parser code review",
        "Audit chat-template and parser handling for a reasoning model. Look for state or tokenization behavior that could affect long multi-turn requests.",
        [
            "common/jinja/README.md",
            "common/jinja/parser.cpp",
            "common/jinja/runtime.cpp",
            "src/llama-chat.cpp",
            "tests/test-chat-template.cpp",
            "docs/development/parsing.md",
        ],
    ),
]

def post_json(endpoint, body):
    request = urllib.request.Request(
        f"http://{host}:{port}{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

def tokenize(content, add_special=False, parse_special=False):
    return post_json("/tokenize", {
        "content": content,
        "add_special": add_special,
        "parse_special": parse_special,
    })["tokens"]

marker = "__LLAMA_CPP_REPRO_27102_USER_CONTENT__"
formatted = post_json("/apply-template", {
    "messages": [
        {"role": "system", "content": "You are a senior software engineer working on llama.cpp."},
        {"role": "user", "content": marker},
    ],
})["prompt"]
template_prefix, template_suffix = formatted.split(marker, 1)

for index, length in enumerate(lengths):
    if length < 1:
        raise ValueError("prompt lengths must be positive")

    title, task, paths = workloads[index % len(workloads)]
    sections = []
    for relative_path in paths:
        path = repo_dir / relative_path
        sections.append(f"\n\nFile: {relative_path}\n\n{path.read_text(errors='replace')}")

    prefix = f"You are working on this llama.cpp engineering task:\n\n{task}\n\nRead the following repository material before answering."
    suffix = f"\n\nNow complete the {title.lower()}. Give a technically precise analysis with concrete references to the supplied code."
    prefix_tokens = tokenize(template_prefix + prefix, parse_special=True)
    corpus_tokens = tokenize("".join(sections))
    suffix_tokens = tokenize(suffix + template_suffix, parse_special=True)

    if length <= len(prefix_tokens) + len(suffix_tokens):
        prompt = tokenize(f"{template_prefix}{prefix}\n\n{suffix}{template_suffix}", parse_special=True)[:length]
    else:
        body_length = length - len(prefix_tokens) - len(suffix_tokens)
        copies, remainder = divmod(body_length, len(corpus_tokens))
        prompt = prefix_tokens + corpus_tokens * copies + corpus_tokens[:remainder] + suffix_tokens

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": True,
        "cache_prompt": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
    with (log_dir / f"payload-{index}.json").open("w") as output:
        json.dump(payload, output, separators=(",", ":"))
    with (log_dir / f"payload-{index}.txt").open("w") as output:
        output.write(f"workload: {title}\ntokens: {len(prompt)}\nfiles:\n")
        output.writelines(f"  {relative_path}\n" for relative_path in paths)
PY

mapfile -t payloads < <(find "$log_dir" -maxdepth 1 -name 'payload-*.json' -print | sort -V)
if ((${#payloads[@]} == 0)); then
    echo "no payloads generated" >&2
    exit 2
fi

deadline=$((SECONDS + duration_min * 60 + client_delay_sec * (clients > 0 ? clients - 1 : 0)))

run_client() {
    local client_id=$1
    local request_id=0
    local payload

    while ((SECONDS < deadline)) && kill -0 "$server_pid" 2>/dev/null; do
        payload=${payloads[$((client_id % ${#payloads[@]}))]}
        echo "$(date --iso-8601=seconds) client=$client_id request=$request_id payload=$(basename "$payload")" >>"$log_dir/client.log"
        curl -sS --no-buffer --max-time 7200 \
            -H 'Content-Type: application/json' \
            --data-binary "@$payload" \
            "http://$host:$port/completion" \
            >/dev/null 2>>"$log_dir/client-errors.log" || true
        request_id=$((request_id + 1))
    done
}

for ((client_id = 0; client_id < clients; ++client_id)); do
    run_client "$client_id" &
    worker_pids+=("$!")
    echo "started client $client_id with $(basename "${payloads[$((client_id % ${#payloads[@]}))]}")"
    if ((client_id + 1 < clients && client_delay_sec > 0)); then
        sleep "$client_delay_sec"
    fi
done

echo "stress running with $clients clients for $duration_min minutes after the final client starts"

result=0
while ((SECONDS < deadline)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "llama-server exited"
        result=1
        break
    fi
    if grep -qE 'launch timed out and was terminated|CUDA error' "$log_dir/server.log" "$log_dir/server.stdout.log" 2>/dev/null; then
        echo "CUDA failure detected"
        result=1
        break
    fi
    sleep 2
done

if [[ "$result" == 0 ]]; then
    echo "stress duration completed without a detected CUDA failure"
else
    grep -nE -B 3 -A 8 'launch timed out and was terminated|CUDA error|launch reporting error|previous successful instrumented' \
        "$log_dir/server.log" "$log_dir/server.stdout.log" 2>/dev/null | tail -80 || true
fi

exit "$result"
