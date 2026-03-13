# Leo Generic-First AppWorld Plan

## Summary
Keep Leo as a general-purpose agent framework and add the minimum new abstractions needed for benchmark-style environments like AppWorld. The core idea is to introduce generic capabilities Leo currently lacks, then implement AppWorld as one adapter on top of them rather than baking benchmark logic into the main agent loop.

This keeps Leo broadly useful for normal tool-using tasks while making it competitive on AppWorld through better stateful execution, environment-scoped tools, benchmark orchestration, and reproducible eval output.

## Key Changes
### 1. Upgrade Leo from static tool calling to capability-based agents
- Add a generic `ToolProvider` interface so Leo can consume tools from:
  - local Python callables
  - lazy-loaded skills
  - external protocols such as MCP
  - environment adapters such as AppWorld
- Add a `CapabilityProfile` for agents so different runs can expose different tool sets and prompts without forking the entire agent implementation.
- Keep `ReActAgent` as the default generic agent, but allow specialized profiles such as `default-chat`, `code-execution`, and `benchmark-environment`.

### 2. Add generic stateful execution support
- Introduce a reusable `ExecutionContext` abstraction for long-running, stateful work.
- Provide a generic Python execution tool that supports persistent variables across steps within one run, with bounded stdout/stderr capture and structured errors.
- Add optional workspace-scoped file access tools for generic coding tasks, but make them capability-gated so benchmark profiles can disable them when inappropriate.
- This becomes the foundation Leo needs for AppWorld's iterative coding style without hardcoding AppWorld into the core loop.

### 3. Add generic environment adapters
- Introduce an `EnvironmentAdapter` interface with lifecycle hooks such as:
  - initialize task/session
  - expose task-scoped tools
  - provide public instructions/context
  - save outputs
  - evaluate results
- Implement `AppWorldEnvironmentAdapter` as the first adapter.
- The AppWorld adapter should expose only public task information and task-scoped tools backed by the AppWorld SDK or APIs.
- The agent should interact with the adapter through generic interfaces, not through AppWorld-specific logic embedded in `ReActAgent`.

### 4. Add external tool protocol support
- Add `MCPToolProvider` with tool discovery, schema normalization, and invocation routing.
- Expose MCP server configuration generically so Leo can use external tool ecosystems beyond AppWorld.
- Treat AppWorld MCP as one concrete server configuration, not as a special case in Leo's architecture.
- Preserve existing chat and skill flows when no MCP servers are configured.

### 5. Add a generic run/eval harness with an AppWorld preset
- Add a generic non-interactive command surface for environment-backed runs:
  - `leo run --environment <name>`
  - `leo evaluate --environment <name>`
- Add an AppWorld preset under that interface rather than a benchmark-only top-level CLI.
- Required AppWorld-specific config should include dataset, task selection, experiment name, and environment endpoints/root path.
- The generic runner should:
  - initialize the environment adapter
  - build the correct capability profile
  - execute the agent loop
  - persist artifacts and traces
  - optionally call environment evaluation hooks
- This keeps the CLI generic while still supporting leaderboard-ready AppWorld outputs.

## Public Interfaces
- New generic interfaces:
  - `ToolProvider`
  - `CapabilityProfile`
  - `ExecutionContext`
  - `EnvironmentAdapter`
- New generic CLI:
  - `leo run --environment <name>`
  - `leo evaluate --environment <name>`
- New first-party implementations:
  - `MCPToolProvider`
  - `AppWorldEnvironmentAdapter`
  - `appworld` environment preset/config

## Test Plan
- Unit tests for `ToolProvider` composition so current skills and registered tools still behave the same.
- Unit tests for `ExecutionContext` state persistence across multiple tool calls.
- Unit tests for `EnvironmentAdapter` lifecycle and capability scoping.
- Unit tests ensuring the AppWorld adapter never exposes hidden evaluation or answer fields.
- Integration test for a generic `leo run --environment appworld` path on a public train task.
- Integration test for AppWorld save/evaluate artifact generation.
- MCP integration test using a fake server to verify discovery and invocation work independently of AppWorld.
- Regression tests confirming `leo ask` and `leo chat` remain unchanged for standard users.

## Assumptions And Defaults
- Leo remains a general-purpose agent first; AppWorld support is implemented as an adapter and preset, not as the dominant execution model.
- `ReActAgent` stays the main loop unless benchmark testing proves a small profile-specific variant is needed.
- Generic stateful execution is useful beyond AppWorld and should therefore live in core Leo.
- AppWorld optimization happens through prompts, tool exposure, and run configuration, not through benchmark-specific hacks in the core architecture.
- The initial benchmark target is leaderboard-ready artifact generation while preserving existing chat workflows.
