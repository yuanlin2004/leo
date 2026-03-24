# Leo Skill/Tool Support TODO

Date: 2026-03-13

## P0

### A. Finish Milestone 1 Provider Refactor

Status: completed on 2026-03-13.

Goal: complete the provider-based tool architecture required by [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md), instead of continuing to grow `ToolsRegistry` as a single direct owner of every tool.

Why this is P0:
- The current behavior mostly satisfies Milestone 1 acceptance criteria, but the required architecture is still missing.
- Later AppWorld milestones assume provider composition exists.
- MCP support was added, but it was added directly into `ToolsRegistry`, not through a generic provider model.

Implemented:
- Added `ToolProvider` and concrete providers for local tools, MCP tools, and skill-contributed tools.
- Refactored `ToolsRegistry` into a composition layer over providers instead of a monolithic direct owner.
- Added `LocalToolProvider` for core tools, meta tools, and directly registered tools.
- Added `SkillToolProvider` for skill discovery, activation, and active skill tool execution.
- Added `MCPToolProvider` so MCP tools participate through the same provider abstraction.
- Preserved agent-facing semantics so `ReActAgent` and `SimpleAgent` did not need public API changes.
- Preserved `/tools`, `/skills`, and `/skill <name>` behavior.

Candidate files:
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/tools/providers.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/providers.py)
- [src/leo/tools/__init__.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/__init__.py)
- [src/leo/agents/react_agent.py](/Users/yuan/Documents/GitHub/leo/src/leo/agents/react_agent.py)
- [src/leo/agents/simple_agent.py](/Users/yuan/Documents/GitHub/leo/src/leo/agents/simple_agent.py)
- [docs/leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md)

Verification:
- `ToolProvider`, `LocalToolProvider`, `SkillToolProvider`, and `MCPToolProvider` exist.
- `ToolsRegistry` aggregates providers instead of owning every tool path directly.
- Existing chat, ask, skill, and MCP tests still pass with only minimal fixture updates.
- `pytest` passed with `67 passed, 3 skipped` when this item was implemented.

### B. Implement Milestone 2 Capability Profiles

Status: completed on 2026-03-13.

Goal: complete Milestone 2 from [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md) so Leo can expose different tool sets and prompt supplements by profile without changing the base agent contract.

Implemented:
- Added `CapabilityProfile` and builtin profiles in [profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py).
- Added profile resolution and profile-aware registry construction in [main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py).
- Added tool tagging and deterministic profile-based visibility filtering in [registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py).
- Kept the default `generic` profile behavior aligned with current Leo behavior.
- Added a `benchmark-environment` profile that hides file, shell, tmux, and skill/MCP meta tools while preserving the base agent contract.
- Wired optional profile prompt supplements through normal agent creation rather than creating a new benchmark-only agent.

Candidate files:
- [src/leo/tools/profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/tools/providers.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/providers.py)
- [src/leo/cli/main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py)
- [test/test_cli.py](/Users/yuan/Documents/GitHub/leo/test/test_cli.py)
- [test/test_core_tools.py](/Users/yuan/Documents/GitHub/leo/test/test_core_tools.py)

Verification:
- Default `leo ask` and `leo chat` continue using the generic profile.
- Profiles can hide or expose tool groups deterministically.
- Agent construction stays backward compatible for current callers.
- `pytest` passed with `69 passed, 3 skipped` when this item was implemented.

### C. Implement Milestone 3 Stateful Execution Context

Status: completed on 2026-03-13.

Goal: complete Milestone 3 from [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md) by adding reusable, session-scoped Python execution with persistent state and structured failures.

Implemented:
- Added `ExecutionContext` in [execution.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/execution.py).
- Added `execute_python` as a core tool with persistent namespace semantics inside one Leo session.
- Added bounded stdout/stderr capture and bounded traceback formatting for failures.
- Reset execution state on session reset through the normal core runtime reset path.
- Gated execution through capability profiles so the default `generic` profile stays lightweight.
- Enabled execution in the `benchmark-environment` profile without changing the base agent contract.

Candidate files:
- [src/leo/tools/execution.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/execution.py)
- [src/leo/tools/core.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/core.py)
- [src/leo/tools/profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [test/test_execution_context.py](/Users/yuan/Documents/GitHub/leo/test/test_execution_context.py)
- [test/test_core_tools.py](/Users/yuan/Documents/GitHub/leo/test/test_core_tools.py)

Verification:
- Multiple execution calls in one run can reuse variables and imports.
- Failures return structured error information instead of crashing the loop.
- Execution state is isolated per session/run via registry reset behavior.
- `pytest` passed with `73 passed, 3 skipped` when this item was implemented.

### D. Implement Milestone 4 Environment Adapter Framework

Status: completed on 2026-03-13.

Goal: complete Milestone 4 from [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md) by adding a generic environment adapter interface, task-scoped tool binding, and a first AppWorld adapter that exposes only public task data.

Implemented:
- Added `EnvironmentAdapter`, `EnvironmentToolSpec`, and `EnvironmentAdapterError` in [adapters.py](/Users/yuan/Documents/GitHub/leo/src/leo/environments/adapters.py).
- Added `AppWorldTaskContext` and `AppWorldEnvironmentAdapter` for local AppWorld-style task payloads, including hidden-field filtering and output evaluation hooks.
- Added `EnvironmentToolProvider` so environment-scoped tools participate through the same provider architecture as local, MCP, and skill tools.
- Updated `ToolsRegistry` to attach and detach environment adapters, expose public environment context, and keep task-scoped tools available only while an adapter is active.
- Added runtime context injection so agents receive public environment task context as transient system messages without direct AppWorld-specific logic in the agent loop.
- Split transient run-state reset from full session teardown so one-shot environment-backed runs do not discard the active environment before the first model call.
- Enabled the environment provider in both builtin capability profiles, while keeping environment tools invisible unless an adapter is actually attached.

Candidate files:
- [src/leo/environments/__init__.py](/Users/yuan/Documents/GitHub/leo/src/leo/environments/__init__.py)
- [src/leo/environments/adapters.py](/Users/yuan/Documents/GitHub/leo/src/leo/environments/adapters.py)
- [src/leo/tools/providers.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/providers.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/tools/profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py)
- [src/leo/agents/react_agent.py](/Users/yuan/Documents/GitHub/leo/src/leo/agents/react_agent.py)
- [src/leo/agents/simple_agent.py](/Users/yuan/Documents/GitHub/leo/src/leo/agents/simple_agent.py)
- [test/test_environment_adapters.py](/Users/yuan/Documents/GitHub/leo/test/test_environment_adapters.py)

Verification:
- Environment-backed runs initialize and tear down cleanly through the registry lifecycle.
- Task-scoped environment tools are available only while an environment adapter is attached.
- Hidden AppWorld fields are excluded from public task context, tool results, and injected model context.
- `pytest` passed with `78 passed, 3 skipped` when this item was implemented.

### E. Implement Milestone 5 AppWorld Tooling And Run Harness

Status: completed on 2026-03-13.

Goal: complete Milestone 5 from [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md) by adding a generic environment-backed run/evaluate CLI, AppWorld task tooling, artifact recording, and a reproducible sequential run harness.

Implemented:
- Expanded [adapters.py](/Users/yuan/Documents/GitHub/leo/src/leo/environments/adapters.py) so `AppWorldEnvironmentAdapter` supports both local payloads and the real `appworld.AppWorld` package runtime.
- Added AppWorld task-scoped tools for task context, live code execution, public doc search, output saving, and evaluation.
- Added [appworld.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/appworld.py) with `AppWorldRunConfig`, `AppWorldTaskResult`, `AppWorldRunSummary`, and sequential run orchestration.
- Added [trace.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/trace.py) for stable JSONL trace recording.
- Added `leo run --environment appworld` and `leo evaluate --environment appworld` in [main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py).
- Added an AppWorld-specific prompt supplement and benchmark-profile defaults for environment-backed runs.
- Added stable artifact outputs per task: final answer, saved output payload, evaluation payload, task result JSON, and run summary JSON.
- Added AppWorld root handling through the official `appworld.update_root(...)` path so Leo can run against downloaded AppWorld data outside the repo root.

Candidate files:
- [src/leo/environments/adapters.py](/Users/yuan/Documents/GitHub/leo/src/leo/environments/adapters.py)
- [src/leo/runs/__init__.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/__init__.py)
- [src/leo/runs/appworld.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/appworld.py)
- [src/leo/runs/trace.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/trace.py)
- [src/leo/cli/main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py)
- [test/test_appworld_run.py](/Users/yuan/Documents/GitHub/leo/test/test_appworld_run.py)

Verification:
- The fake-package integration test proves Leo can initialize an AppWorld task, execute multiple environment code steps, save outputs, and evaluate through the generic harness.
- The batch/result path writes stable per-task artifacts and a run-level summary.
- A real smoke run completed end-to-end on public AppWorld train task `82e2fac_1` using downloaded AppWorld data under `/tmp/appworld-data`, with Leo artifacts written under `/tmp/leo-appworld-smoke/leo-real-smoke/82e2fac_1`.
- `pytest` passed with `83 passed, 3 skipped` after this item landed.

### F. Implement Milestone 6 AppWorld MCP And Competitive Hardening

Status: completed on 2026-03-13.

Goal: complete Milestone 6 from [leo-generic-first-appworld-plan-milestones.md](/Users/yuan/Documents/GitHub/leo/docs/leo-generic-first-appworld-plan-milestones.md) by supporting AppWorld MCP through the same provider architecture and adding enough tracing/replay support to debug benchmark runs.

Implemented:
- Enabled MCP providers in the `benchmark-environment` capability profile in [profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py).
- Added registry-level event emission in [registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py) for tool calls, tool results, environment attach/detach, save, and evaluate events.
- Added tracing LLM support in [appworld.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/appworld.py) so prompts and model responses are captured alongside tool and environment events.
- Added `leo replay --trace <path>` in [main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py) for single-trace replay/debug summaries.
- Added AppWorld MCP configuration support in the run harness via `--appworld-mcp-url` or `--appworld-mcp-command`, reusing the existing generic `MCPToolProvider`.
- Verified an AppWorld MCP-backed run path in [test/test_appworld_run.py](/Users/yuan/Documents/GitHub/leo/test/test_appworld_run.py) without changing agent semantics.

Candidate files:
- [src/leo/tools/profiles.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/profiles.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/runs/appworld.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/appworld.py)
- [src/leo/runs/trace.py](/Users/yuan/Documents/GitHub/leo/src/leo/runs/trace.py)
- [src/leo/cli/main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py)
- [test/test_appworld_run.py](/Users/yuan/Documents/GitHub/leo/test/test_appworld_run.py)
- [test/test_cli.py](/Users/yuan/Documents/GitHub/leo/test/test_cli.py)

Verification:
- AppWorld MCP tools are discovered and invoked through the same registry path as other MCP tools.
- Leo can run AppWorld tasks through either the direct adapter-backed path or an MCP-backed path without changing the agent loop.
- Saved JSONL traces contain model requests, model responses, tool calls, tool results, environment events, and final result metadata.
- `leo replay --trace` summarizes saved traces for single-task debugging.
- `pytest` passed with `83 passed, 3 skipped` after this item landed.

## Foundation

### 0. Add Core Coding-Agent Tools

Status: completed on 2026-03-13.

Goal: give Leo a minimal primitive toolset for coding workflows before layering higher-level skills on top.

Rationale:
- `read`, `write`, `edit`, and shell execution are universal runtime primitives, not domain skills.
- Skills should describe workflows and domain procedures, not replace basic filesystem and process access.
- `tmux` should be exposed as a persistent session primitive after basic shell/process support exists.

Implemented:
- Added `read_file` with workspace path checks, optional line ranges, and truncation limits.
- Added `edit_file` with guarded search/replace semantics.
- Added `write_file` with explicit overwrite protection.
- Added `run_shell` for workspace-scoped command execution.
- Added managed tmux session tools:
- `tmux_start_session`
- `tmux_send_keys`
- `tmux_capture_pane`
- `tmux_kill_session`
- Registered these as core tools in the runtime, independent of skill activation.

Candidate files:
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/tools/__init__.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/__init__.py)
- [src/leo/tools/core.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/core.py)
- [src/leo/skills/runtime.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/runtime.py)
- [test/test_simple_agent.py](/Users/yuan/Documents/GitHub/leo/test/test_simple_agent.py)
- [test/test_react_agent.py](/Users/yuan/Documents/GitHub/leo/test/test_react_agent.py)
- [test/test_core_tools.py](/Users/yuan/Documents/GitHub/leo/test/test_core_tools.py)

Verification:
- Leo can inspect files, modify files, and execute shell commands without relying on skills.
- Persistent terminal workflows can be managed through dedicated `tmux` session tools.
- Tests cover path restrictions, truncation behavior, edit safety, shell failures, and tmux session lifecycle.
- `pytest` passed with `55 passed, 2 skipped` when this item was implemented.

## Priority 1

### 1. Add Real MCP Runtime Support

Status: completed on 2026-03-13.

Goal: close the gap between declaring MCP dependencies and actually using MCP-backed tools.

Implemented:
- Added an `MCPToolRuntime` and server config model with stdio and HTTP transport support.
- Added HTTP response handling for both JSON and SSE (`text/event-stream`) MCP responses.
- Added MCP config loading from `--mcp-config`, `LEO_MCP_CONFIG`, or inline `LEO_MCP_SERVERS`.
- Registered discovered MCP tools through the normal tool registry path with `mcp:<server>` provenance.
- Added `list_mcp_servers` to expose connection status, discovered tools, and server metadata.
- Surface connection, protocol, and tool-call failures as normal runtime errors.
- Added an opt-in live integration test against `https://mcp.deepwiki.com/mcp`.
- Kept `get_skill_requirements` as the skill-side dependency surface.

Candidate files:
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [src/leo/tools/__init__.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/__init__.py)
- [src/leo/tools/mcp.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/mcp.py)
- [src/leo/cli/main.py](/Users/yuan/Documents/GitHub/leo/src/leo/cli/main.py)
- [test/test_mcp_tools.py](/Users/yuan/Documents/GitHub/leo/test/test_mcp_tools.py)
- [docs/leo-skills-full-support-requirements-and-design.md](/Users/yuan/Documents/GitHub/leo/docs/leo-skills-full-support-requirements-and-design.md)

Verification:
- MCP-backed tools can be discovered and invoked from the normal agent tool loop.
- Missing or misconfigured MCP servers produce actionable errors.
- Tests cover discovery, invocation, and failure handling.
- `pytest` passed with `63 passed, 3 skipped` after HTTP support and live integration coverage were added.
- The live DeepWiki MCP integration test passed when run with `LEO_RUN_LIVE_MCP_TESTS=1`.

### 2. Add Skill Readiness / Preflight Checks

Status: completed on 2026-03-13.

Goal: let Leo answer whether a skill is runnable in the current environment before attempting execution.

Implemented:
- Added `check_skill_readiness` as a core tool.
- Added side-effect-free skill inspection so Leo can assess readiness without activating the skill.
- Validate inferred or declared binaries for skill commands.
- Validate required environment variables.
- Validate required MCP servers against current MCP runtime status.
- Surface compatibility, auth, and platform constraints as warnings or manual checks when they are not machine-verifiable.
- Return a structured report with `ready`, `blocking_issues`, `warnings`, `commands`, and `suggested_remediation`.

Candidate files:
- [src/leo/skills/catalog.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/catalog.py)
- [src/leo/skills/runtime.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/runtime.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)
- [test/test_skills_registry.py](/Users/yuan/Documents/GitHub/leo/test/test_skills_registry.py)

Verification:
- The model can call one tool to assess whether a skill can run here.
- Reports distinguish hard blockers from optional compatibility notes.
- Tests cover binary/env/MCP failure cases.
- `pytest` passed with `67 passed, 3 skipped` when this item was implemented.

## Priority 2

### 3. Promote Skill Installation to a Core Leo Feature

Goal: make remote skill discovery and install/update a native product capability rather than only a skill workflow.

Tasks:
- Add a native command/API to list installable curated skills.
- Add a native command/API to install a skill from a curated source or GitHub path.
- Record origin, channel, and install source in catalog-visible metadata.
- Support update/reinstall semantics explicitly instead of only install-if-missing.
- Reuse the existing `skill-installer` workflow logic where practical rather than duplicating behavior.

Candidate files:
- [src/leo/skills/catalog.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/catalog.py)
- New install module under `src/leo/skills/`
- CLI entrypoints if present
- [skill-installer](/Users/yuan/.codex/skills/.system/skill-installer/SKILL.md)

Definition of done:
- Users can list and install external skills without manually invoking a helper skill.
- Installed skills show clear provenance in discovery output.
- Network and auth failures are surfaced clearly.

### 4. Ship a Curated Built-In Skill Pack

Goal: make Leo immediately useful with a small set of high-value operational skills.

Tasks:
- Create a `.system` or `.curated` starter pack for workflows such as `github`, `tmux`, `web-research`, and `repo-review`.
- Keep each skill aligned with Leo’s current activation/resource/command model.
- Use these bundled skills as end-to-end fixtures for runtime tests.
- Document expected environment requirements for each skill.

Candidate files:
- New skill folders under `.leo/skills/`
- [test/test_skills_registry.py](/Users/yuan/Documents/GitHub/leo/test/test_skills_registry.py)
- [test/test_react_agent.py](/Users/yuan/Documents/GitHub/leo/test/test_react_agent.py)

Definition of done:
- Leo ships with a minimal, opinionated built-in skill set.
- At least one bundled skill exercises commands and one exercises resource loading.
- Tests validate auto-activation and explicit activation flows.

## Priority 3

### 5. Harden Skill Manifest Semantics

Goal: reduce heuristic discovery and make third-party skill compatibility more deterministic.

Tasks:
- Define explicit metadata for activation-time resources.
- Define explicit metadata for commands and execution mode.
- Define explicit metadata for requirements beyond current inference.
- Prefer declared metadata over regex or file-structure inference.
- Preserve heuristic inference as a fallback for legacy or partially specified skills.

Candidate files:
- [src/leo/skills/catalog.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/catalog.py)
- [src/leo/skills/runtime.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/runtime.py)
- [docs/leo-skills-full-support-requirements-and-design.md](/Users/yuan/Documents/GitHub/leo/docs/leo-skills-full-support-requirements-and-design.md)

Definition of done:
- Explicit manifests override inference cleanly.
- Skill discovery output becomes more deterministic across repos.
- Tests cover declared-vs-inferred precedence.

## Suggested Phase Plan

### Phase 1
- Add real MCP runtime support.
- Add skill readiness / preflight checks.

### Phase 2
- Promote skill installation to a core Leo feature.
- Ship a curated built-in skill pack.

### Phase 3
- Harden skill manifest semantics.
