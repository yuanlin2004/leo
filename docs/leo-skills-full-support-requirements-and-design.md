# Leo Full External Skills Support: Requirements And Design

Date: 2026-03-13

## Goal

Leo must support the skill packages in:

- `openai/skills`
- `google-gemini/gemini-skills`
- `anthropics/skills`

"Support" means Leo can:

1. discover the skill
2. activate the skill
3. inject the skill instructions into protected model context
4. load bundled companion resources on demand
5. surface missing requirements and dependencies
6. execute the skill workflow when that workflow requires bundled scripts or external CLIs

This is broader than the current milestone-1 implementation, which already supports instruction injection and on-demand resource loading for many skills, but does not yet support the full external execution and dependency model.

## Repo Study Summary

### OpenAI

- Repo: <https://github.com/openai/skills/tree/main>
- Current structure includes channel folders such as `.system` and `.curated`.
- Skills often include:
  - `references/`
  - `scripts/`
  - `agents/openai.yaml`
  - `assets/`
  - occasional config files such as `config.yaml`

Representative examples:

- `openai-docs` uses bundled references plus MCP dependency metadata.
- `playwright` uses references plus a bundled shell wrapper script.
- `gh-fix-ci` and `imagegen` rely on Python helper scripts.
- `vercel-deploy` relies on deployment scripts and external auth/CLI state.

### Anthropic

- Repo: <https://github.com/anthropics/skills>
- Skills are mostly:
  - `SKILL.md`
  - companion markdown guides
  - helper scripts
  - occasional binary assets

Representative examples:

- `frontend-design` is instruction-only.
- `pdf`, `docx`, `pptx`, and `xlsx` are instruction + guide + script packages.

### Gemini

- Repo: <https://github.com/google-gemini/gemini-skills>
- Skills are mostly instruction/reference packages.
- Frontmatter can include extra fields such as `compatibility`.

Representative examples:

- `gemini-api-dev` is instruction-only.
- `vertex-ai-api-dev` is instruction + references + `compatibility`.

## Key Product Conclusion

Most external skills are **not** "call a single registered tool" packages.

Most of them are one of these:

1. instruction-only
2. instruction + references
3. instruction + scripts/CLIs
4. instruction + MCP dependencies

Leo therefore needs a full skill runtime, not just:

- `activate_skill`
- `get_skill_resource`
- Leo-native Python `actions.py`

## Requirements

### 1. Discovery

Leo must:

- discover nested skill roots and repo channel folders
- preserve the package origin and channel
- support deterministic precedence rules
- treat same-name collisions as an explicit policy decision, not filesystem-order luck

Required metadata per discovered skill:

- `canonical_id`
- `name`
- `description`
- `scope`
- `origin_repo`
- `channel`
- `path`
- `loadable`
- `validation_error`

### 2. YAML And Package Parsing

Leo must parse real YAML frontmatter and preserve fields beyond `name` and `description`, including:

- `metadata`
- `compatibility`
- `license`
- future-compatible fields that should not be dropped silently

Leo must also inspect package files beyond `SKILL.md`, including:

- `references/`
- `scripts/`
- `templates/`
- `assets/`
- `agents/*.yaml`
- top-level config files

### 3. Activation And Protected Context

On activation, Leo must:

- load the `SKILL.md` body into protected system context
- keep that context available across the session
- not force every skill to contribute runtime tools

Instruction-only skills such as Anthropic `frontend-design` should work with no additional runtime tool registration.

### 4. Resource Loading

Leo must expose bundled resources on demand without exposing generic file reads.

The model needs a way to retrieve:

- focused references
- helper scripts
- templates
- config files

This already exists in part via `get_skill_resource`, but the long-term requirement is broader package awareness, not just regex-based file detection from `SKILL.md`.

### 5. Requirements And Dependencies

Leo must surface what a skill needs in order to work:

- MCP servers
- environment variables
- external binaries
- platform constraints
- auth requirements
- network/escalation expectations

Examples:

- OpenAI `openai-docs` needs the `openaiDeveloperDocs` MCP server.
- OpenAI `playwright` needs `npx`.
- OpenAI `vercel-deploy` needs Vercel auth and network access.
- Gemini `vertex-ai-api-dev` declares `compatibility` requirements.

### 6. Workflow Execution

To support all skills, Leo must be able to execute declared skill workflows.

This does **not** mean every skill must be converted into a runtime tool by name.

It **does** mean Leo needs a safe execution layer for bundled commands and scripts.

### 7. Execution Policy

Policy:

- do not use bash as the runtime orchestrator
- do not use `bash -c`
- use direct subprocess argv execution when a command can be invoked directly
- use `tmux` for persistent, interactive, or shell-oriented workflows

`tmux` is the default session mechanism when a workflow:

- spans multiple steps
- needs persistent terminal state
- launches a server
- streams output over time
- depends on a shell wrapper or long-running process

This policy is required because many external skills assume ongoing command-line workflows, not single pure function calls.

## Proposed Runtime Design

### Skill Model

Add or extend the following types:

- `SkillManifest`
- `SkillSummary`
- `SkillResource`
- `SkillRequirement`
- `SkillCommand`
- `ActivatedSkill`

### Skill Resource Model

Each resource should track:

- `path`
- `kind` such as `reference`, `script`, `template`, `asset`, `config`
- `text_loadable`
- `binary`
- `size_bytes`

### Skill Requirement Model

Each requirement should track:

- `kind` such as `mcp`, `env_var`, `binary`, `platform`, `auth`
- `name`
- `value`
- `required`
- `source`

### Skill Command Model

Each runnable workflow command should track:

- `name`
- `execution_mode` (`direct` or `tmux`)
- `argv_template`
- `cwd_mode`
- `allowed_env`
- `timeout_ms`
- `produces_artifacts`
- `source`

## Model-Visible Tools

Keep:

- `list_available_skills`
- `activate_skill`
- `get_skill_resource`

Add:

- `get_skill_requirements`
- `list_skill_commands`
- `run_skill_command`

### `get_skill_requirements`

Returns the dependency model for an activated skill:

- MCP servers
- env vars
- binaries
- auth notes
- platform notes

### `list_skill_commands`

Returns the safe, declared command workflows that Leo can execute for the activated skill.

### `run_skill_command`

Runs a declared skill command with validated arguments.

Rules:

- no bash orchestration
- direct argv execution when possible
- `tmux` for persistent/session-based workflows
- explicit allowlist only
- stdout/stderr captured
- artifacts normalized and returned

## How Command Support Should Work

Leo should not automatically expose every file in `scripts/` as a callable model tool.

Instead:

1. define repo-aware adapters for known ecosystems
2. declare safe command entry points explicitly
3. map those entry points into `SkillCommand`

This is safer and more realistic than trying to infer arbitrary script interfaces from prose.

## Repo Adapter Requirement

Because the three repos do not currently share a single executable manifest standard, Leo needs compatibility adapters for:

- `openai/skills`
- `anthropics/skills`
- `google-gemini/gemini-skills`

Repo adapters should:

- understand repo-specific packaging
- extract dependency metadata
- identify safe command entry points
- set channel precedence rules where needed

## Acceptance Criteria

Leo should be considered to fully support a skill when it can:

1. discover the skill
2. activate it
3. inject its instructions into the model prompt
4. load referenced resources on demand
5. report its requirements and dependencies
6. execute its declared workflow commands using direct subprocesses or `tmux`
7. return useful outputs and artifact paths

## Initial Rollout Order

1. extend discovery/indexing with richer package metadata
2. add requirements extraction
3. add `list_skill_commands`
4. add `run_skill_command`
5. implement repo adapters for OpenAI, Anthropic, and Gemini
6. add compatibility tests against cloned copies of all three repos

## Notes

- Instruction-only skills are already mostly supported.
- Resource-heavy skills are now partly supported.
- The largest remaining gap is safe execution of external skill workflows plus dependency awareness.
- The runtime should prefer direct argv execution and `tmux`, not bash, for skill workflow support.
