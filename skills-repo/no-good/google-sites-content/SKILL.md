---
name: google-sites-content
description: >-
  Extract content from a Google Sites page (sites.google.com or a custom
  domain hosted on Google Sites). These pages are JavaScript-rendered so
  web_fetch returns only boilerplate; the real content lives in embedded
  Google Drive/Docs/Sheets/PDFs whose IDs are in the raw HTML. Trigger on
  any of: URL on sites.google.com; web_fetch output under ~500 characters
  or dominated by navigation/"click a link" text; raw HTML containing
  DOCS_timing, drive.google.com/viewer/main, or script nonce blobs with
  little visible body text.
---

# When to use this

**Load this skill BEFORE running any bash command, as soon as either of
these is true:**

- `web_fetch` on the URL returned mostly navigation / welcome text, a "stay
  informed by clicking" message, or otherwise obviously incomplete content.
- The URL is `sites.google.com/...`, or the raw HTML contains
  `DOCS_timing`, `drive.google.com/viewer/main`, or `<script nonce="...">`
  with very little visible body text.

Do not attempt ad-hoc grep/curl exploration first — it almost always wastes
turns on JavaScript blobs. Load the skill and follow the recipe.

# Recipe

### Step 1. Pull the raw HTML and find embedded Google resources

```bash
curl -sL "<page-url>" > /tmp/page.html
grep -oE 'drive\.google\.com/(embeddedfolderview|file/d/|folderview)[^"]+' /tmp/page.html | sort -u
grep -oE 'docs\.google\.com/(document|spreadsheets|presentation|forms)/d/[0-9a-zA-Z_-]+' /tmp/page.html | sort -u
grep -oE 'https?://[^"]+\.pdf' /tmp/page.html | sort -u
```

These three greps cover the common embeds. Inspect what comes out; pick the
resource that matches the user's question.

### Step 2a. Drive folder (embeddedfolderview)

```bash
FOLDER_ID="..."
curl -s "https://drive.google.com/embeddedfolderview?id=$FOLDER_ID" > /tmp/folder.html
```

**Extract (file_id, filename) pairs — do not skip this step.** The folder
HTML lists each entry with both an embedded link and a visible title; you
need both so downstream steps and your final answer can reference files
by their real names.

```bash
python3 - <<'PY'
import re
html = open("/tmp/folder.html").read()
titles = re.findall(r'flip-entry-title[^>]*>([^<]+)<', html)
ids = list(dict.fromkeys(re.findall(r'file/d/([0-9a-zA-Z_-]+)', html)))
for fid, title in zip(ids, titles):
    print(f"{fid}\t{title.strip()}")
PY
```

**Warning: do NOT make up filenames from the order of file IDs.** The real
filenames in the folder HTML are authoritative — e.g., `2026-03 MVHS
Newsletter March 2026`. If you save a downloaded PDF to `/tmp/<something>.pdf`,
name it from the real title (e.g. `/tmp/2026-03.pdf`), not from a positional
guess. Mislabeling the file will cause you to cite the wrong source in
your final answer.

### Step 2b. Google Doc (plain text export)

```bash
DOC_ID="..."
curl -sL "https://docs.google.com/document/d/$DOC_ID/export?format=txt" > /tmp/doc.txt
```

### Step 2c. Google Sheet (CSV export)

```bash
SHEET_ID="..."
curl -sL "https://docs.google.com/spreadsheets/d/$SHEET_ID/export?format=csv" > /tmp/sheet.csv
```

### Step 2d. PDF (on Drive or a direct URL)

```bash
FILE_ID="..."   # from Step 1 or 2a
curl -L -s "https://docs.google.com/uc?export=download&id=$FILE_ID" | pdftotext - /tmp/out.txt
# For a direct PDF URL:
# curl -sL "<pdf-url>" | pdftotext - /tmp/out.txt
```

### Step 3. Search the extracted text

```bash
grep -i -C 2 "<search terms>" /tmp/out.txt
```

### Step 4. Quote correctly in the final answer

**When you quote the source in your reply to the user, pull full paragraphs,
not grep line-matches.** After finding the hit in Step 3, re-run with wider
context (`grep -B 2 -A 10 "<term>"`) or just dump the whole extracted file
and read the surrounding paragraph. Never stitch `grep | head` output into
a quote — it produces fragmented text that truncates mid-sentence and
misrepresents the source.

# Tips

- Files in `/tmp` persist across bash calls within a session; reuse them.
- If a Drive folder has many files, loop and grep each one for the user's
  search term until you find the right document.
- If all of Step 1 returns nothing, the page may not actually be Google
  Sites — double-check the URL and fall back to generic scraping.
