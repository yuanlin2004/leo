# Leo Generic-First AppWorld Plan With Milestones

## Summary
Extend Leo's core architecture so it remains a general-purpose agent while gaining the capabilities needed for AppWorld: composable tool sources, stateful code execution, environment adapters, external tool protocols, and a generic run/evaluate harness. Implement AppWorld as the first environment adapter and tuning target, not as a benchmark-specific fork.

The work should land in six milestones. Each milestone is independently testable and leaves the codebase in a usable state. The order matters: later milestones assume the abstractions from earlier ones already exist.

## Milestone 1: Core Tool Abstractions
### Goal
Replace Leo's current "single registry of Python callables" model with a generic, composable tool system that still preserves all existing behavior.

### Changes
- Introduce `ToolProvider` as the common interface for tool discovery and execution.
- Keep `ToolsRegistry`, but reposition it as the composition layer that aggregates providers instead of owning every tool directly.
- Preserve existing local tools and skill-backed actions by wrapping them in provider implementations rather than rewriting their behavior.
- Keep current agent-facing semantics unchanged: agents still request tool schemas and invoke tools by name.

### Step-by-step
1. Define `ToolProvider` interface with methods for listing schemas, listing descriptions, and executing by tool name.
2. Split current `ToolsRegistry` responsibilities into:
   - provider registration/composition
   - legacy compatibility wrapper methods
   - meta-tool support for skills
3. Add a `LocalToolProvider` for tools registered directly in code.
4. Add a `SkillToolProvider` for lazy-loaded skill actions.
5. Rewire `ReActAgent` and `SimpleAgent` to consume the provider-backed registry without changing their public APIs.
6. Preserve current `/tools`, `/skills`, and `/skill <name>` CLI behavior.

### Acceptance criteria
- Existing chat and ask flows behave the same.
- Existing skill discovery and lazy loading still work.
- Existing tests around `ToolsRegistry`, `ReActAgent`, and `SimpleAgent` still pass with only minimal fixture updates.

## Milestone 2: Capability Profiles And Agent Configuration
### Goal
Allow Leo to run the same agent loop with different tool sets, prompts, and execution capabilities depending on the environment.

### Changes
- Introduce `CapabilityProfile` to describe which providers, prompts, and execution features are enabled for a run.
- Add profile-aware agent construction so Leo can stay generic while still having an AppWorld-optimized mode.
- Avoid creating a completely separate benchmark-only agent unless testing proves the main loop is insufficient.

### Step-by-step
1. Define `CapabilityProfile` with:
   - profile name
   - enabled providers
   - optional extra system prompt
   - execution toggles for code/file/environment tools
2. Update agent creation in the CLI layer so profiles can be selected programmatically.
3. Keep the default profile as today's generic Leo behavior.
4. Add a `benchmark-environment` profile that can be used later by environment-backed runs.
5. Ensure profile selection changes available tools without changing the base agent contract.

### Acceptance criteria
- Default `leo ask` and `leo chat` continue using the generic profile.
- Profiles can hide or expose tool groups deterministically.
- Agent construction stays backward compatible for current callers.

## Milestone 3: Stateful Execution Context
### Goal
Give Leo a reusable, generic execution substrate for iterative coding tasks such as AppWorld.

### Changes
- Introduce `ExecutionContext` for stateful Python execution across multiple tool invocations within one run.
- Add a generic Python execution tool with persistent globals/locals, bounded output capture, and structured failures.
- Make execution capability profile-gated so generic chat can remain lightweight.

### Step-by-step
1. Define `ExecutionContext` lifecycle:
   - create per run
   - execute code snippet
   - preserve state between calls
   - reset on run completion/failure
2. Implement Python execution with:
   - persistent namespace
   - captured stdout/stderr
   - structured return payload containing output, error, and summary
3. Add limits for output size and exception formatting to prevent runaway context growth.
4. Expose the execution tool through a provider rather than hardcoding it into agents.
5. Add profile-level control so normal chat does not automatically get code execution unless enabled.
6. Document the expected tool contract for agents so prompts can rely on stable behavior.

### Acceptance criteria
- Multiple execution calls in one run can reuse variables and imported modules.
- Failures return structured error information rather than crashing the loop.
- Execution state is isolated per session/run.

## Milestone 4: Generic Environment Adapter Framework
### Goal
Create a reusable interface for task-backed environments so AppWorld is just the first implementation.

### Changes
- Introduce `EnvironmentAdapter` abstraction for environments that provide instructions, scoped tools, persistence, and evaluation hooks.
- Define generic run lifecycle hooks rather than putting environment logic into the agent loop.
- Implement `AppWorldEnvironmentAdapter` as the first adapter.

### Step-by-step
1. Define `EnvironmentAdapter` interface with methods to:
   - initialize environment/session
   - provide public task context
   - expose task-scoped tool providers
   - save outputs
   - optionally evaluate outputs
   - clean up resources
2. Define `AppWorldTaskContext` as the public context object returned to the agent.
3. Implement `AppWorldEnvironmentAdapter` on top of AppWorld SDK or supported remote endpoints.
4. Expose only public task data to the agent; do not surface hidden evaluation/answer fields.
5. Bind the adapter's task-scoped tools into the active capability profile for the duration of a run.
6. Keep the agent unaware of AppWorld internals beyond the tool schemas and task context it receives.

### Acceptance criteria
- Environment-backed runs can initialize and tear down cleanly.
- Task-scoped tools are only available for the active environment run.
- Hidden AppWorld data is not exposed through context or tool results.

## Milestone 5: AppWorld Tooling And Run Harness
### Goal
Make Leo capable of executing AppWorld tasks end-to-end in a reproducible way while still using the generic architecture.

### Changes
- Add AppWorld-specific tool providers and a generic environment-backed CLI.
- Add a non-interactive run harness that records artifacts, traces, and evaluation outputs.
- Tune the profile and prompt for stateful task solving without forking the whole system.

### Step-by-step
1. Add AppWorld task-scoped tools exposed through the adapter:
   - get task/instruction context
   - execute stateful code against the live world
   - search public AppWorld docs
   - save outputs
   - complete/evaluate task when allowed
2. Add generic CLI commands:
   - `leo run --environment appworld`
   - `leo evaluate --environment appworld`
3. Define `AppWorldRunConfig` with:
   - dataset
   - task selection
   - experiment name
   - model/provider
   - step/time limits
   - local root or remote endpoint configuration
4. Build the run harness flow:
   - parse config
   - initialize adapter
   - build capability profile
   - create execution context
   - run agent loop
   - save outputs and trace
   - evaluate if requested
5. Emit `AppWorldTaskResult` and run-level summaries with stable fields for debugging and leaderboard packaging.
6. Keep batch execution sequential in v1 for correctness and simpler state isolation.
7. Add an AppWorld-specific prompt supplement focused on:
   - short verify/fix loops
   - disciplined use of task docs
   - explicit save/complete behavior
   - avoiding irrelevant exploration

### Acceptance criteria
- Leo can run at least one public AppWorld train task end-to-end.
- Outputs are saved in a layout AppWorld tooling can evaluate directly.
- Batch runs continue after per-task failures and produce a summary.

## Milestone 6: MCP Support And Competitive Hardening
### Goal
Generalize Leo's external tool support and harden the AppWorld path for benchmark iteration and repeatable evaluation.

### Changes
- Add `MCPToolProvider` as another generic provider implementation.
- Support AppWorld MCP configuration through the same provider architecture.
- Add observability and replay facilities needed for benchmark tuning.

### Step-by-step
1. Implement `MCPToolProvider` with:
   - server configuration
   - tool discovery
   - schema normalization
   - tool invocation routing
2. Support both stdio and HTTP transports if feasible in the same abstraction; if not, land HTTP first and note stdio as follow-up.
3. Allow capability profiles to include one or more MCP providers.
4. Add trace capture for:
   - prompts
   - tool calls
   - execution snippets
   - environment events
   - final result metadata
5. Add a single-task replay/debug mode using saved traces and verbose logging.
6. Validate that AppWorld can be run either through direct adapter-backed tools or through an MCP-backed provider without changing agent semantics.
7. Use public train/dev tasks to iterate on prompt and profile tuning only after the infrastructure is stable.

### Acceptance criteria
- MCP tools can be discovered and invoked through the same registry path as local tools.
- Leo can support AppWorld MCP without a benchmark-specific code path in the core agent.
- Debug traces are sufficient to reproduce and analyze failures.

## Public Interfaces
- New core interfaces:
  - `ToolProvider`
  - `CapabilityProfile`
  - `ExecutionContext`
  - `EnvironmentAdapter`
- New AppWorld/environment types:
  - `AppWorldRunConfig`
  - `AppWorldTaskContext`
  - `AppWorldTaskResult`
- New provider implementation:
  - `MCPToolProvider`
- New generic CLI surface:
  - `leo run --environment <name>`
  - `leo evaluate --environment <name>`

## Test Plan
### Core regression
- Existing `leo ask`, `leo chat`, skills, and tool execution behavior remain unchanged under the default profile.
- Existing local-tool and skill-tool tests still pass after the provider refactor.

### New unit coverage
- `ToolProvider` composition and name resolution.
- `CapabilityProfile` gating of tool visibility.
- `ExecutionContext` state persistence, isolation, and structured error returns.
- `EnvironmentAdapter` lifecycle and cleanup behavior.
- `MCPToolProvider` schema loading and invocation with a fake server.

### AppWorld integration
- Public-task test proving Leo can initialize an AppWorld task, receive public instructions, execute multiple stateful code steps, save outputs, and evaluate.
- Batch-run test where one task fails and later tasks still execute.
- Safety test proving hidden evaluation fields are never exposed to the agent.
- Artifact test proving run outputs are AppWorld-readable.

### Competitive validation
- Smoke benchmark run on a small public train/dev slice.
- Trace review workflow proving failures can be diagnosed from saved artifacts alone.
- Prompt/profile regression tests to ensure tuning changes do not silently degrade generic Leo behavior.

## Assumptions And Defaults
- Leo remains generic-first; environment-specific behavior is added through profiles, providers, and adapters.
- `ReActAgent` remains the primary loop unless concrete benchmark evidence shows it must be replaced.
- Stateful Python execution is a core Leo capability, not an AppWorld-only feature.
- AppWorld is the first environment adapter and first benchmark target, but the architecture must support additional environments later without rework.
- V1 batch execution is sequential.
- Competitive tuning is performed on public train/dev tasks only; blind test sets are treated as evaluation-only.
