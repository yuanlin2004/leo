---
name: google-sites-content
description: >-
  Extract content from a Google Sites page (sites.google.com or a custom
  domain hosted on Google Sites). These pages are JavaScript-rendered so
  web_fetch returns only boilerplate; the real content lives in embedded
  Google Drive/Docs/Sheets/PDFs whose IDs are in the raw HTML.
---

# When to use this

Load this skill when `web_fetch` on a page returns mostly navigation text,
a welcome message, or obviously incomplete content, AND the page is hosted
on Google Sites. Signals:

- URL is `sites.google.com/...` or the page source contains `google.com/sites`,
  `DOCS_timing`, or `drive.google.com/viewer/main`.
- `curl -s <url> | head` shows a lot of obfuscated JavaScript and `<script nonce="...">`.

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
# Each entry is a file ID + filename:
grep -oE 'file/d/[0-9a-zA-Z_-]+' /tmp/folder.html | sort -u
```

To see filenames alongside IDs, grep for the surrounding `<div class="flip-entry">`
blocks in `/tmp/folder.html`.

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

# Tips

- Files in `/tmp` persist across bash calls within a session; reuse them.
- If a Drive folder has many files, loop and grep each one for the user's
  search term until you find the right document.
- If all of Step 1 returns nothing, the page may not actually be Google
  Sites — double-check the URL and fall back to generic scraping.
