# Leo Skill/Tool Support TODO

Date: 2026-03-13

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

Goal: let Leo answer whether a skill is runnable in the current environment before attempting execution.

Tasks:
- Add a tool such as `check_skill_readiness`.
- Validate binaries declared or inferred for skill commands.
- Validate required env vars.
- Validate MCP dependencies once MCP runtime support exists.
- Include auth/platform compatibility hints where available.
- Return a compact report with `ready`, `blocking_issues`, and `suggested_remediation`.

Candidate files:
- [src/leo/skills/catalog.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/catalog.py)
- [src/leo/skills/runtime.py](/Users/yuan/Documents/GitHub/leo/src/leo/skills/runtime.py)
- [src/leo/tools/registry.py](/Users/yuan/Documents/GitHub/leo/src/leo/tools/registry.py)

Definition of done:
- The model can call one tool to assess whether a skill can run here.
- Reports distinguish hard blockers from optional compatibility notes.
- Tests cover binary/env/MCP failure cases.

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
- New skill folders under `.agents/skills/`
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
