# LEO

An LLM-based agent framework built from scratch on raw LLM APIs.

## Requirements

- Python 3.10+
- A running OpenAI-compatible LLM server. The default target is a local [vLLM](https://github.com/vllm-project/vllm) serving `Qwen/Qwen3.5-35B-A3B-FP8` at `http://localhost:8000/v1`; see `CLAUDE.md` for the `vllm serve` command. Override via `LEO_LLM_BASE_URL` / `LEO_LLM_MODEL` / `LEO_LLM_API_KEY`.
- [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) for sandboxing tool-invoked shell commands.

## Setup

### 1. Allow unprivileged user namespaces (Ubuntu 24.04+)

Recent Ubuntu kernels ship AppArmor with unprivileged user namespace creation restricted by default, which blocks `bwrap` with `setting up uid map: Permission denied`. Disable the restriction:

```bash
sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0
echo 'kernel.apparmor_restrict_unprivileged_userns = 0' | sudo tee /etc/sysctl.d/60-apparmor-userns.conf
```

The `/etc/sysctl.d/` line makes it persist across reboots.

### 2. Install Leo

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 3. Start vllm

```bash
docker run -d --name vllm --privileged --gpus all --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node \
  bash -c "vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 262144 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.5 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder"
```

The `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` flags are required for native tool calling (OpenAI-style `tool_calls`) with Qwen3. The `qwen3_coder` parser matches Qwen3's XML-tagged `<tool_call><function=NAME>…</function></tool_call>` format (despite the "coder" name, it handles the whole Qwen3 family, not just coder variants). Using `hermes` here leaves tool-call text unparsed in the assistant content.

### 4. Run

```bash
leo                       # start the chatbot
leo -sysprompt my.txt     # use a custom system prompt
```

Type `/help` inside the REPL for commands.

### Environment variables

Leo loads a `.env` file from the current working directory on startup (via `python-dotenv`). Values already set in the shell environment take precedence over the file. Recognized keys:

- `TAVILY_API_KEY` — required for the `web_search` tool.
- `LEO_LLM_BASE_URL` — defaults to `http://localhost:8000/v1`.
- `LEO_LLM_MODEL` — defaults to `Qwen/Qwen3.5-35B-A3B-FP8`.
- `LEO_LLM_API_KEY` — defaults to `EMPTY` (vLLM ignores it but the SDK requires something).

`.env` is gitignored.

## Skills

Leo discovers skills at startup from `~/.leo/skills/<name>/SKILL.md`. Each SKILL.md starts with a YAML frontmatter block, followed by instructions the model reads after calling `load_skill`:

```markdown
---
name: my-skill
description: One sentence summary the model sees upfront.
---

# Instructions

Step-by-step guidance here. Reference scripts or assets in the same directory;
the model invokes them via the `bash` tool.
```

At startup, Leo injects `name: description` for each skill into the system prompt. When the model decides a skill is relevant, it calls `load_skill(name)` to read the full body. Use `/skills` in the REPL to list what's installed.
