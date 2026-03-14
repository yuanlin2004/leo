# AppWorld Status

This file tracks AppWorld tasks that Leo currently passes in live evaluation runs.

## Passing Tasks

| Task ID | App | Task Type | Status | Latest Live Artifact |
| --- | --- | --- | --- | --- |
| `82e2fac_1` | `spotify` | Question answering | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/82e2fac_1/result.json` |
| `27e1026_1` | `spotify` | Question answering | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/27e1026_1/result.json` |
| `23cf851_1` | `venmo` | Question answering | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/23cf851_1/result.json` |
| `cf6abd2_1` | `simple_note` | State mutation, null answer | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/cf6abd2_1/result.json` |
| `302c169_1` | `phone` | State mutation, null answer | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/302c169_1/result.json` |
| `68ee2c9_1` | `file_system` | State mutation, null answer | Passing | `/tmp/leo-appworld-runs/codex-appworld-regression-20260314b/68ee2c9_1/result.json` |

## Live Evaluation Baseline

- Provider: `openrouter`
- Model: `nvidia/nemotron-3-super-120b-a12b:free`
- Temperature: `0`
- Run mode: `leo.cli.main evaluate`
- Latest full passing sweep: `codex-appworld-regression-20260314b`

## Regression Rule

Whenever a new AppWorld task is enabled:

1. Add or update the task-specific Leo strategy/tooling.
2. Run the new task live until it passes.
3. Rerun every task listed in `Passing Tasks`.
4. Fix any regressions before committing.
5. Update this file with the new passing task and latest artifact path.

Do not commit a new AppWorld task enablement unless the full passing set still passes live.
