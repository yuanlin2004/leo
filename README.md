# LEO

An LLM-based agent framework built from scratch on raw LLM APIs.

## Requirements

- Python 3.10+
- A running OpenAI-compatible LLM server. The default target is a local [vLLM](https://github.com/vllm-project/vllm) serving `Qwen/Qwen3.6-35B-A3B-FP8` at `http://localhost:8000/v1`; see `CLAUDE.md` for the `vllm serve` command. Override via `LEO_LLM_BASE_URL` / `LEO_LLM_MODEL` / `LEO_LLM_API_KEY`.
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
  bash -c "vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
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

Leo loads `~/.env` first, then a `.env` in the current working directory (via `python-dotenv`). Later sources do not override keys already set — so shell env wins over project `.env`, which wins over `~/.env`. Recognized keys:

- `TAVILY_API_KEY` — required for the `web_search` tool.
- `LEO_LLM_BASE_URL` — defaults to `http://localhost:8000/v1`.
- `LEO_LLM_MODEL` — defaults to `Qwen/Qwen3.6-35B-A3B-FP8`.
- `LEO_LLM_API_KEY` — defaults to `EMPTY` (vLLM ignores it but the SDK requires something).

`.env` is gitignored.

## Examples

### Example 1: Extract Information from a Webpage that Embeds a Google Sheet

Most cloud based chatbot (e.g. ChatGPT, Gemini, Claude, Grok 2026-04) failed this task. 

```
$ leo
...
you> /think-on
(thinking: on)

you> Find who the webmaster is from [mvmb.org - About Us](https://www.mvmb.org/Booster-Board)or the pages it links to. 

leo> I found the webmaster information for mvmb.org!

Based on the [MVMB Members Contact List](https://docs.google.com/spreadsheets/d/1t4vG4sSDMVNGBixn1XKh40S4BB_4YGiPU_soLpgYqPY) embedded on the Booster Board page, the webmaster is:

**Yuan Lin** - Media Chair and Webmaster
- **Email:** media_admin@mvmb.org

This information is from the 2025-2026 Booster Board contact list for Monta Vista High School Music Boosters (MVMB).
```

### Example 2: Build a Parpahty LLM Wiki 

Create the schema file. 

```
$ leo
...
you> Read the article at https://gist.githubusercontent.com/karpathy/442a6bf555914893e9891c11519de94f/raw/ac46de1ad27f92b28ac95459c782c07f6b8c964a/llm-wiki.md and create a llm-wiki.md file that works at the system prompt for an LLM agent that works as describe in the article.

leo> Now I have the full article. Let me create a system prompt for an LLM that embodies the "LLM Wiki" pattern described by Karpathy.

leo> Done! I've created `llm-wiki.md` — a system prompt file that turns an LLM into a **Wiki Maintainer Agent** as described in Karpathy's article. ...
```

Launch the agent

```
$ leo -systemprompt llm-wiki.md
...
you> Inject the raw sources from ~/raw and build the wiki.

...
leo>  The wiki is fully built. Here's the summary: ...
```


## Skills

Leo can use skills that follow the spec on https://agentskills.io.

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

Each skill's directory is read-only mounted into the bash sandbox at its own absolute path, so scripts and assets that live alongside `SKILL.md` (e.g. `~/.leo/skills/my-skill/helper.py`) can be invoked directly.

A set of skills that have been tested for Leo can be found in the `skills-repo` folder. Copy the folder of a skill to `~/.leo/skills`. Some skills need setup, see the `README.md` file that comes with a skill. 

## Tracing (LangSmith)

Leo emits [LangSmith](https://smith.langchain.com/) traces when `LANGSMITH_TRACING=true`. Each user turn is a parent span with child spans for every LLM call (via `wrap_openai`) and every tool invocation, giving a tree like `turn → [llm_call, tool_call, llm_call, …]`.

`LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` are typically kept in `~/.env` as user-level defaults. Flip tracing on per project by setting `LANGSMITH_TRACING=true` in the project `.env` (or in the shell). Leave it unset — or set to anything other than `true` — and the wrappers no-op with zero runtime cost.


# Acknowledgements

- Part of `CLAUDE.md` is modified from https://github.com/forrestchang/andrej-karpathy-skills