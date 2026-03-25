# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .          # Install in editable mode
pytest                    # Run all tests
pytest test/test_react_agent.py  # Run a specific test file
leo ask "..."             # One-shot agent query
leo chat                  # Interactive multi-turn chat
```

## Architecture

Leo is a CLI-based agentic framework. The main layers are:

**CLI → AgentRuntimeBuilder → Agent → ToolsRegistry → LLM**

### Entry Point
`src/leo/cli/main.py` handles argument parsing and dispatches to one of five commands: `ask`, `chat`, `run`, `evaluate`, `replay`. It builds the runtime via `AgentRuntimeBuilder`.

### Agents (`src/leo/agents/`)
Three agent implementations inherit from `Agent` base class:
- **ReActAgent** – structured JSON reasoning loop (`thought`, `content`, `code`, `tool_calls`); primary agent
- **SimpleAgent** – minimal overhead, iterative tool loop
- **PlanExecuteAgent** – two-phase plan-then-execute

`AgentSession` wraps any agent with multi-turn conversation state and supports save/load.

### Tool System (`src/leo/tools/`)
`ToolsRegistry` aggregates multiple `ToolProvider` implementations:
- `LocalToolProvider` – built-in tools (file I/O, shell, tmux, Python execution, search)
- `SkillToolProvider` – discovers and activates skills from `.leo/skills/`
- `MCPToolProvider` – Model Context Protocol server integration
- `EnvironmentToolProvider` – environment-specific tools for evaluation

Capability profiles (`profiles.py`) control which tools are visible. The `benchmark-environment` profile hides file/shell/tmux tools for safe evaluation.

### Skills (`src/leo/skills/`, `.leo/skills/`)
Skills are discoverable tool bundles. They live under `~/.leo/skills/` (user-global) or `.leo/skills/` (project-local). Activated per agent instance; exportable tools, resources, and requirements.

### LLM Client (`src/leo/core/llm.py`)
`LeoLLMClient` wraps OpenAI-compatible APIs. Supports OpenRouter and Ollama. Handles schema sanitization, retries, and backoff.

### Agent Spec (`src/leo/agent_spec.py`)
`AgentSpec` (frozen dataclass) loaded from YAML defines an agent's capability profile, skills, plugins, and model defaults. The builtin `leo.generic` spec lives in `src/leo/builtin_agent_specs/generic.yaml`.

### Environments (`src/leo/environments/`, `src/leo_plugins/`)
`EnvironmentIntegration` is the plugin base for task-backed evaluation (e.g., AppWorld). Plugins are loaded via `module:function` syntax. The AppWorld plugin (`src/leo_plugins/appworld/`) adapts Leo agents to the AppWorld SDK for benchmarking.

### Execution Tracing (`src/leo/runs/`)
Full execution traces are recorded to `.jsonl` files and can be replayed via `leo replay --trace <file>`.

## Configuration

Environment variables (shell or `.env`):
- `LEO_PROVIDER` – `openrouter` or `ollama`
- `LEO_MODEL` – model identifier
- `LEO_AGENT` – agent type (`react`, `simple`, `plan_execute`)
- `LEO_LOG_LEVEL`
- `OPENROUTER_API_KEY`, `OLLAMA_API_KEY`, `OLLAMA_BASE_URL`, `TAVILY_API_KEY`

## External Dependencies

- **AppWorld repo**: https://github.com/stonybrooknlp/appworld/ — local copy with data at `/Users/yuan/Documents/GitHub/appworld`

## Key Conventions

- `src/` is the Python source root (set in `pyproject.toml` `pythonpath`)
- Test files follow `test_*.py` naming; `tsimple_agent.py` and `treact_agent.py` are also discovered
- `artifacts/` holds run outputs (gitignored)
- Plugin loading uses `module:class_or_function` string syntax

## Git Commits

- Do not mention Claude or AI assistance in commit messages or co-author lines.
