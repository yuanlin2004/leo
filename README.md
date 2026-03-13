# Leo

CLI for interacting with Leo agents in one-shot and multi-turn chat modes.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
leo ask "Summarize latest AI infra news"
leo chat
```

## Chat Commands

- `/help`
- `/exit` or `/quit`
- `/reset`
- `/skills`
- `/skill <name>`
- `/tools`
- `/config`
- `/save <file>`
- `/load <file>`

## Transcript Persistence

Save and restore chat state:

```text
/save transcripts/session.json
/load transcripts/session.json
```

The saved file is JSON with `schema_version`, `messages`, and `activated_skill_ids`.

## Banner

`leo chat` shows a LEO banner by default.

Use `--no-banner` to skip it:

```bash
leo chat --no-banner
```

## Environment Variables

You can set keys in shell env or `.env`:

- `OPENROUTER_API_KEY`
- `TAVILY_API_KEY`
- `LEO_PROVIDER`
- `LEO_MODEL`
- `LEO_AGENT`
- `LEO_LOG_LEVEL`
