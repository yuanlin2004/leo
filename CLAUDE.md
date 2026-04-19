# General Principles

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

# About This Project

LEO is an LLM-based agent framework. 


## Directory Structure

- src: python source root 
- docs: documents
- venv: python virtual environment
- artifacts: run outputs and logs (gitignored)
- tests: unit tests
- scripts: various scripts
- prompts: varous prompts for experimental purposes

## Other Useful Info

### Default LLM Model
Qwen3.5-35B-A3B-FP8

### How to start vllm

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

### Long-running batches: use tmux

The development setup is a Mac running a remote SSH session to the Linux box. When the Mac sleeps, the SSH session breaks and any background processes started by Claude Code receive SIGHUP and die — even though they run on the remote Linux box, they are children of the Claude Code session.

For runs longer than ~30 min, launch via tmux so the work survives session disconnects:

```bash
# Start (replace <name> and <command>)
tmux new -d -s <name> 'source venv/bin/activate && <command> 2>&1 | tee /tmp/<name>.log'


