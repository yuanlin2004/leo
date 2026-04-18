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

### 3. Run

```bash
leo                       # start the chatbot
leo -sysprompt my.txt     # use a custom system prompt
```

Type `/help` inside the REPL for commands.
