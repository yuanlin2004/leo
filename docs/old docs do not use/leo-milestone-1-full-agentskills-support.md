# Revised Milestone 1: Full AgentSkills Support In Leo

## Summary
Replace the original "Core Tool Abstractions" milestone with a full AgentSkills-compliance milestone. In this milestone, Leo should implement AgentSkills-style skill discovery, progressive disclosure, activation, and context handling as first-class behavior, while keeping tools and skills clearly separated.

This milestone uses the following decisions:
- Conformance: strict spec only
- Activation: dedicated model-visible skill lifecycle tools
- Discovery scope: project-level plus user-level skills, with deterministic precedence

After this milestone, Leo should be able to discover AgentSkills-compliant skills, expose only compact metadata by default, activate a skill on demand, register any tools/resources that skill contributes, and preserve activated skill context correctly across the session.

The full spec is at https://agentskills.io/home

## Key Changes
### 1. Separate skills from tools completely
- Define `Skill` as a spec-compliant package/recipe, not a callable capability.
- Define `Tool` as a model-callable runtime capability.
- Remove the conceptual role of "skill actions" from the runtime model.
- Eliminate `execute_skill_action` from the target architecture for compliant skills.
- Require that once a skill is activated, any contributed tools are exposed as normal tools by name.

### 2. Add an AgentSkills-compliant skill catalog
- Introduce a `SkillsCatalog` responsible only for:
  - discovery
  - metadata indexing
  - activation
  - activated-skill state
  - loading skill content/resources according to disclosure rules
- Support two discovery roots in v1:
  - project scope under the repo
  - user scope under the user skill directory
- Define deterministic precedence:
  - project skill with the same canonical id overrides user skill
- Reject non-compliant skills in strict mode with explicit load errors.

### 3. Implement progressive disclosure
- On discovery, Leo should load only the minimum skill metadata needed for listing and routing.
- Before activation, the model should only receive compact skill summaries, not full skill bodies.
- On activation, Leo should load the skill's full behavioral instructions and any declared resources needed at activation time.
- Activated skill content must remain protected from normal prompt compaction/truncation rules so the model does not silently lose the recipe mid-session.
- If the session resets, activated skill state resets too.

### 4. Add dedicated skill lifecycle tools
- Keep skill activation separate from file reading.
- Expose model-visible lifecycle tools:
  - `list_available_skills`
  - `activate_skill`
- Optionally add `get_activated_skills` if needed for observability, but do not require it for v1.
- `list_available_skills` should return compact metadata only.
- `activate_skill` should:
  - validate the requested skill exists
  - validate it is AgentSkills-compliant
  - load the full skill package
  - register any contributed tools/resources into the current session runtime
  - return a compact activation result summarizing what became available
- Do not expose generic file-read-based skill activation in this milestone.

### 5. Add tool registration from activated skills
- Define a skill activation result object that can contribute:
  - behavioral instructions
  - tool definitions/handlers
  - optional non-tool resources Leo needs to keep available
- Route contributed tools into `ToolsRegistry` as normal tools with provenance metadata such as `skill:<skill_name>`.
- Enforce unique tool names across the active session.
- Duplicate tool names should be a hard error, not last-write-wins.
- Only activated skills contribute tools; discovered-but-inactive skills do not.

### 6. Update agent and CLI behavior
- `ReActAgent` and `SimpleAgent` should continue to consume only normal tool schemas plus agent-internal tools like `final_answer`.
- Agent prompts should instruct the model to discover and activate skills before using skill-contributed tools.
- `/skills` should show discovered skills from the catalog, not active tool names.
- `/skill <name>` should display skill metadata and activation status; if you want parity with model behavior, it may optionally activate the skill, but the default should be inspect-only unless explicitly asked.
- `/tools` should show only currently registered runtime tools, including any from activated skills.

## Public Interfaces
- New types:
  - `SkillManifest`
  - `SkillSummary`
  - `ActivatedSkill`
  - `SkillActivationResult`
  - `SkillsCatalog`
- Updated interfaces:
  - `ToolsRegistry` becomes runtime-tool-only and accepts dynamic tool registration from activated skills
- Model-visible tools:
  - `list_available_skills`
  - `activate_skill`
- Removed target pattern for spec-compliant skills:
  - `execute_skill_action`

## Implementation Details
### Skill loading model
- Discovery pass:
  - scan project and user skill roots
  - parse only the spec fields needed for listing, compatibility validation, and routing
  - index by canonical skill id/name
- Activation pass:
  - load the skill body and activation-time resources
  - build a `SkillActivationResult`
  - append skill instructions into the session's protected skill context
  - register any contributed tools into the session tool runtime

### Session model
- Activated skills are session-scoped, not process-global.
- A new chat session starts with zero activated skills.
- Saving/loading conversations should either:
  - persist activated skill ids and reactivate them on load, or
  - explicitly reject transcript restoration without reconstructing skill state.
- For this milestone, prefer persisting activated skill ids and reactivating them during transcript load so sessions remain usable.

### Error handling
- Missing skill: return a normal tool error from `activate_skill`.
- Non-compliant skill: return a strict-spec validation error with the first blocking reason.
- Duplicate contributed tool name: fail activation and keep prior runtime state unchanged.
- Partial activation must roll back; activation is all-or-nothing.

## Test Plan
### Discovery and validation
- Discover compliant skills from both project and user roots.
- Confirm project scope overrides user scope on name collision.
- Reject legacy Leo-only skills when required spec fields/structure are missing.
- Verify invalid skills appear as invalid/not-loadable rather than silently activating.

### Progressive disclosure
- `list_available_skills` returns summary metadata only, not full skill bodies.
- Activating a skill loads its full instructions and keeps them in protected session state.
- Non-activated skills do not contribute tools.
- Session reset clears activated skill state and removes contributed tools.

### Activation and tool registration
- Activating a compliant skill registers its contributed tools and makes them callable by name.
- Duplicate tool names fail activation atomically.
- Deactivated/nonexistent skills never affect runtime tools.
- `execute_skill_action` is no longer required for compliant-skill flows.

### Agent and CLI regression
- `leo ask` and `leo chat` still work with no activated skills.
- `/skills` lists discovered skills correctly.
- `/tools` reflects the active runtime tool set.
- Transcript save/load restores activated skills correctly.
- `final_answer` remains agent-internal and unaffected by the skill redesign.

## Assumptions And Defaults
- AgentSkills is the only accepted skill format in this milestone.
- Legacy Leo skills must be migrated rather than carried forward in compatibility mode.
- Dedicated activation tools are the only model-facing activation path in v1.
- Project and user discovery roots are both supported in milestone 1.
- Activated skills may contribute tools, but skills themselves are never model-callable tools.
- Protected activated-skill context is a session concern and must survive prompt compaction within that session.
