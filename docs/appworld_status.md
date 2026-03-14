# AppWorld Status

This file tracks Leo's AppWorld status after the architecture reset to LLM-authored AppWorld code.

## Current Baseline

Leo no longer uses adapter-authored task-family solution code or `execute_appworld_task_strategy`.
The AppWorld adapter now provides:

- public task context
- API discovery via `list_appworld_apis`
- API schema lookup via `describe_appworld_api`
- execution via `execute_appworld_code`
- auth hints and task-plan guidance only

The model must write the Python snippet itself.

## LLM-Authored Code Baseline

| Task ID | App | Task Type | Status | Model | Latest Live Artifact |
| --- | --- | --- | --- | --- | --- |
| `82e2fac_1` | `spotify` | Question answering | Passing | `nvidia/nemotron-3-super-120b-a12b:free` | `/tmp/leo-appworld-runs/appworld-llm-code-baseline-82e2fac_1/82e2fac_1/result.json` |
| `302c169_1` | `phone` | State mutation, null answer | Failing | `nvidia/nemotron-3-super-120b-a12b:free` | `/tmp/leo-appworld-runs/appworld-llm-code-baseline-302c169_1/302c169_1/result.json` |

## Historical Strategy-Assisted Baseline

The tasks below passed before the refactor, when Leo still had adapter-authored task-family strategy code. They are historical results only and must be revalidated before being counted again under the new architecture.

- `82e2fac_1`
- `27e1026_1`
- `23cf851_1`
- `23cf851_2`
- `23cf851_3`
- `cf6abd2_1`
- `cf6abd2_2`
- `cf6abd2_3`
- `302c169_1`
- `302c169_2`
- `302c169_3`
- `68ee2c9_1`
- `68ee2c9_2`
- `68ee2c9_3`
- `27e1026_2`
- `27e1026_3`

## Notes

- `82e2fac_1` passed with the model generating and executing its own AppWorld code.
- `302c169_1` failed because the model entered a repeated mutation loop and never reached `final_answer`.
- This suggests the current default model is viable for at least some retrieval tasks, but not yet reliable enough for all mutation tasks under the new architecture.

## Revalidation Rule

When AppWorld behavior changes:

1. Run a small no-strategy baseline set first.
2. Record both passes and failures in this file.
3. Only promote a task back into the passing set after it passes live with LLM-authored code.
4. If the model repeatedly loops or fails to synthesize correct AppWorld code, treat that as a model-quality issue first, not a reason to reintroduce adapter-authored task code.
