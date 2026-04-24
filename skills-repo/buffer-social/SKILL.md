---
name: buffer-social
description: create and manager social media posts using buffer.com

---

## Instructions 


### `buffer profiles`

List all connected profiles.

### `buffer post <text>`

Create content.

Options:

- `--profile <id>`: single target profile
- `--profiles <ids>`: comma-separated profile IDs
- `--all`: all connected profiles
- `--time <datetime>`: ISO 8601 scheduled time
- `--queue`: add to queue
- `--image <path>`: attach local image path (validated; upload flow limited by current API docs)
- `--draft`: save as idea/draft instead of post

### `buffer queue`

View scheduled/queued posts.

Options:

- `--profile <id>`: filter by profile
- `--limit <n>`: max results

### `buffer ideas`

List saved ideas.

Options:

- `--limit <n>`: max results

## Common Use Cases

```bash
# Post to one profile
node ./buffer.js post "Just shipped 🚀" --profile <id>

# Schedule for tomorrow
node ./buffer.js post "Tomorrow update" --profile <id> --time "2026-03-03T14:00:00Z"

# Multi-channel post
node ./buffer.js post "New blog live" --profiles id1,id2

# Save draft
node ./buffer.js post "Draft concept" --profile <id> --draft
```

## Troubleshooting

- **Auth errors (401/403):** check `BUFFER_API_KEY`, regenerate key if needed.
- **Rate limits (429):** wait ~60s and retry.
- **Invalid date:** use ISO format like `2026-03-03T14:00:00Z`.
- **Image path error:** verify file exists and path is correct.

