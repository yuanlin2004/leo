# Leo AgentSpec Mental Model And Packaging

## Purpose

This document defines Leo's long-term mental model and packaging boundaries.

Leo is a generic runtime, not the agent definition itself. A usable, specialized Leo agent is formed by combining the runtime with a reusable agent definition, then instantiating that definition with memory and session state.

This document is the canonical reference for:

- the core ontology of Leo agents
- the relationship among skills, environments, plugins, and memory
- the top-level packaging model
- the default composition and trust rules

## Core Concepts

### LeoCore

`LeoCore` is the generic runtime.

It owns:

- the agent loop
- tool execution plumbing
- provider composition
- session mechanics
- model client integration
- runtime safety and lifecycle management

`LeoCore` is not a specialized agent by itself.

### AgentSpec

`AgentSpec` is the reusable definition of an agent type.

It specifies:

- what kind of agent this is
- what capabilities it has
- what environment it operates in
- what policies, prompts, defaults, and plugins shape its behavior

`AgentSpec` is the top-level reusable artifact for agent definition.

### SkillPack

`SkillPack` is a reusable procedural or domain capability package.

It contributes things like:

- instructions
- references
- commands
- optional tool-facing capability requirements

A skill teaches the agent how to do something. It does not define the whole agent.

### EnvironmentPack

`EnvironmentPack` is a reusable package describing the world/interface the agent operates in.

It contributes things like:

- environment-facing guidance
- context exposure rules
- task/output/evaluation contracts
- references to runtime adapter behavior

An environment defines where and how the agent operates. It does not define the whole agent.

### Plugin

`Plugin` is an executable extension package.

Plugins are the extension mechanism for new executable behavior. In v1, plugins may contribute tools directly.

Plugins are distinct from declarative packages such as `SkillPack` and `EnvironmentPack`.

### CapabilityProfile

`CapabilityProfile` is a policy/exposure layer.

It controls which runtime capabilities are exposed for a particular agent configuration, such as:

- tool groups
- provider visibility
- execution features
- prompt supplements tied to capability mode

It is part of the policy/configuration layer, not a top-level package type.

### Memory

`Memory` is persistent knowledge attached to a live agent instance.

Examples include:

- user or workspace preferences
- durable facts
- episodic summaries
- learned recurring patterns

Memory is instance-bound, not spec-bound.

### AgentInstance

`AgentInstance` is a live runtime instantiation of an `AgentSpec`.

It combines:

- the reusable agent definition
- persistent memory
- session state
- active environment/runtime state

## Composition Model

The core composition model is:

`LeoCore + AgentSpec -> AgentType`

`AgentType + Memory + SessionState -> AgentInstance`

This means:

- `LeoCore` is the engine
- `AgentSpec` is the reusable build sheet
- `AgentInstance` is the live running copy

## Relationship Model

### Skills And Environments Are Siblings

`SkillPack` and `EnvironmentPack` are sibling inputs into `AgentSpec`.

They solve different problems:

- `SkillPack`: how to do something
- `EnvironmentPack`: where and how the agent operates

Neither should subsume the other.

This means:

- environments are not a special kind of skill
- skills are not environment attachments
- both are composed by the `AgentSpec`

### Memory Is Instance-Bound

Memory belongs to the live agent instance, not the reusable agent definition.

This keeps `AgentSpec` portable and shareable, while allowing each deployed or user-bound instance to accumulate its own persistent state.

### One Primary Environment Per AgentSpec

For now, an `AgentSpec` should bind to at most one primary `EnvironmentPack`.

Multiple environments may be supported later as orchestration, but they are not part of the base mental model.

## Packaging Strategy

Leo should use separate top-level package formats, not one universal package format.

The top-level package types are:

- `AgentSpec`
- `SkillPack`
- `EnvironmentPack`
- `Plugin`

These package types should share a common metadata vocabulary, including fields such as:

- `id`
- `version`
- `display_name`
- `description`
- `author`
- `license`
- `compatibility`
- `dependencies`
- provenance or signing metadata

However, each package type should have its own schema and validation rules.

This separation is preferred because it improves:

- trust-boundary clarity
- validation quality
- lifecycle clarity
- user/operator understanding

### Bundle Format

A higher-level bundle format may be added later for distribution convenience, but it should not replace the separate internal package types.

## Plugin Strategy

Plugins are allowed.

In v1:

- plugins may contribute tools directly
- plugin loading is explicit
- plugin APIs are part of Leo's compatibility surface

### Dependency Ownership

`SkillPack` and `EnvironmentPack` may declare plugin requirements.

However, `AgentSpec` is the final explicit authority on which plugins are authorized and loaded.

This means:

- packages may state what they need
- `AgentSpec` decides what is actually enabled
- transitive plugin activation should not happen implicitly

This keeps executable authority visible and auditable at the `AgentSpec` layer.

## Recommended AgentSpec Structure

Conceptually, an `AgentSpec` should compose:

- prompt layers
- zero or more `SkillPack` references
- zero or one primary `EnvironmentPack` reference
- one `CapabilityProfile`
- zero or more explicitly authorized `Plugin` references
- tool, output, and policy defaults
- optional model defaults

An `AgentSpec` should not contain:

- conversation history
- transient execution state
- persistent user memory
- task-local artifacts

## Implications For Leo

This mental model implies the following architectural direction:

- `AgentSpec` should become a first-class runtime/config object in Leo
- current hardcoded environment metadata should move toward `EnvironmentPack`
- `EnvironmentAdapter` should remain as runtime mechanics, but increasingly load declarative environment-pack data
- skills should remain capability packages rather than becoming the top-level agent abstraction
- plugin loading should become an explicit part of `AgentSpec` evaluation and startup

## Defaults Chosen

The current chosen defaults are:

- skills and environments are siblings
- memory is instance-bound
- separate top-level package formats are preferred
- plugins are allowed
- plugins may contribute tools directly in v1
- `AgentSpec` explicitly authorizes plugins
- one primary environment per `AgentSpec` for now

## Summary

Leo's long-term architecture should be understood as:

- `LeoCore`: generic runtime
- `AgentSpec`: reusable agent definition
- `SkillPack` and `EnvironmentPack`: sibling package inputs
- `Plugin`: executable extension mechanism
- `Memory`: persistent instance-bound state
- `AgentInstance`: a live instantiated agent

This model keeps the runtime generic, the packaging clean, and the extension model scalable.
