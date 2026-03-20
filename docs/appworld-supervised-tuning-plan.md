# AppWorld-Supervised Prompt/Context Tuning For Leo

## Summary

Use AppWorld `train` as offline supervision to tune Leo's existing control surface, not to fine-tune the base model. The loop is:

1. Mine `train` task specs, `solution.py`, `compiled_solution.py`, `required_apis.json`, and `api_calls.json`.
2. Distill them into reusable prompt rules and sanitized strategy exemplars keyed only by public task features.
3. Sweep prompt recipes, context-injection recipes, and a small low-temperature grid through Leo's existing `evaluate` harness.
4. Select the winning recipe by highest `dev` pass rate.
5. Keep `test_*` completely untouched.

Primary objective: maximize `dev` pass rate.

Supervision policy: distill patterns only; never expose solution code, answers, or hidden fields at runtime.

## Implementation Changes

### 1. Add an offline AppWorld tuning pipeline

Create a non-runtime script or module that reads AppWorld from an explicit `appworld_root` and emits a sanitized tuning dataset with, for each `train` task:

- Public features Leo may legally use at runtime: instruction text, task id pattern/family, `required_apps`, public data fields, answer shape, mutation vs QA class, difficulty.
- Offline-only supervision features: required API sequence, login pattern, pagination usage, aggregation/ranking pattern, save/finalization behavior, common failure-recovery pattern.
- A short natural-language strategy summary derived from the reference solution and API-call trace.
- No copied answer, no private data, no raw solution code in runtime artifacts.

Output artifacts:

- `strategy_library.jsonl`: sanitized exemplars and summaries.
- `prompt_findings.json`: mined global rules and family-specific rules.
- `train_manifest.json`: task metadata and labels for sweep analysis.

### 2. Introduce a config-driven tuning recipe

Define a small recipe format Leo can load for AppWorld runs. Each recipe should specify:

- `system_rules`: extra prompt supplement blocks.
- `context_policy`: whether to inject retrieved strategy exemplars, how many, and retrieval keys.
- `temperature`: one global value.
- `task_family_overrides`: optional prompt/context overrides by app or task family.

Start with three recipe dimensions only:

- Prompt variants: baseline current supplement, mined global rules, mined global + family rules.
- Context variants: no exemplars, top-1 exemplar, top-2 exemplars.
- Temperature grid: `0.0`, `0.05`, `0.1`, `0.2`.

Default retrieval keys:

- `required_apps`
- normalized instruction pattern
- mutation vs QA
- ranking/filtering keyword family from public text

### 3. Extend the AppWorld run path for dynamic context engineering

Add one AppWorld-specific hook in the run path so a recipe can inject extra runtime context before the user prompt is sent.

Behavior:

- Keep the current public task context injection unchanged.
- Keep the current benchmark prompt supplement unchanged as the baseline layer.
- If a recipe enables exemplar retrieval, fetch sanitized strategy summaries using only public task features and inject them as an additional system message or prompt section.
- Log which recipe and exemplars were used into run artifacts for reproducibility.

Do not change the core ReAct loop. Keep this in the AppWorld tuning layer and agent-spec/prompt assembly.

### 4. Add a sweep/evaluation harness

Build a repeatable tuner that:

- Generates candidate recipes from the recipe grid.
- Runs `leo evaluate` over `train` for coarse fitting.
- Keeps per-task artifacts already produced by Leo and adds a run-level comparison table.
- Ranks recipes by:
  1. `dev` pass rate
  2. `train` to `dev` generalization gap
  3. median iterations/tool calls as tiebreakers

Recommended search procedure:

- Stage 1: sweep on all `train`.
- Stage 2: keep top 3 recipes and run them on all `dev`.
- Stage 3: select one winning recipe and freeze it as the new AppWorld default.

### 5. Distillation rules for prompt/context content

Prompt and exemplar distillation should produce only reusable reasoning patterns such as:

- auth flow ordering
- when to inspect docs vs execute code
- pagination handling
- exact metric interpretation
- answer formatting and save/final-answer behavior
- common mutation completion checks
- app-specific API naming pitfalls

Explicitly exclude from prompts/exemplars:

- final answers
- private data
- concrete task-specific identifiers not present in public context
- verbatim long code blocks from `solution.py`
- hidden required API lists at runtime

## Public Interfaces / Config Additions

Add these stable interfaces:

- AppWorld tuning recipe format:
  - `id`
  - `system_rules`
  - `context_policy`
  - `temperature`
  - `task_family_overrides`
- Sanitized strategy record format:
  - `task_id`
  - `public_features`
  - `strategy_summary`
  - `app_family`
  - `task_family`
  - `tags`
- Optional AppWorld run config fields:
  - `tuning_recipe_path`
  - `strategy_library_path`

No change to the base agent contract or tool schemas.

## Test Plan

Validate in four layers:

- Unit tests for the miner:
  - strategy records contain no answer/private fields
  - retrieval keys are derived only from public task data
  - long solution code is not copied into runtime artifacts
- Unit tests for recipe loading:
  - recipe parsing, defaults, and override resolution
  - temperature and context policy are applied deterministically
- Integration tests for AppWorld run assembly:
  - retrieved exemplar context appears when enabled
  - run artifacts record recipe id and selected exemplars
  - baseline runs still work with no recipe
- Benchmark validation:
  - current baseline recipe on `train` and `dev`
  - full sweep over the candidate grid
  - freeze the best recipe only if `dev` pass rate improves over baseline

## Assumptions

- Leo is being tuned by prompt/context optimization only, not model fine-tuning.
- `train` solutions and traces may be used offline for distillation.
- `dev` is used only for model-selection-style validation, not for fitting prompt rules.
- `test_normal` and `test_challenge` remain unused during tuning.
- A single global temperature is sufficient for v1; per-family temperature overrides are deferred unless the sweep shows a clear benefit.
