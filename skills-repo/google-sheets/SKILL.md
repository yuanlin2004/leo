---
name: Google Sheets
description: read, write, update and manage Google Sheets
---

## Instructions

Use the `gog` CLI tool to interact with Google Sheets. The tool supports a wide range of operations, including reading and writing cell values, formatting, managing named ranges, inserting rows/columns, handling notes and links, creating new spreadsheets, and managing tabs.

```
# Read
gog sheets metadata <spreadsheetId>
gog sheets get <spreadsheetId> 'Sheet1!A1:B10'
gog sheets get <spreadsheetId> MyNamedRange

# Export (via Drive)
gog sheets export <spreadsheetId> --format pdf --out ./sheet.pdf
gog sheets export <spreadsheetId> --format xlsx --out ./sheet.xlsx

# Write
gog sheets update <spreadsheetId> 'A1' 'val1|val2,val3|val4'
gog sheets update <spreadsheetId> 'A1' --values-json '[["a","b"],["c","d"]]'
gog sheets update <spreadsheetId> 'Sheet1!A1:C1' 'new|row|data' --copy-validation-from 'Sheet1!A2:C2'
gog sheets update <spreadsheetId> MyNamedRange 'new|row|data'
gog sheets update <spreadsheetId> 'Sheet1!A1:C1' 'new|row|data' --copy-validation-from MyValidationNamedRange
gog sheets append <spreadsheetId> 'Sheet1!A:C' 'new|row|data'
gog sheets append <spreadsheetId> 'Sheet1!A:C' 'new|row|data' --copy-validation-from 'Sheet1!A2:C2'
gog sheets find-replace <spreadsheetId> "old" "new"
gog sheets find-replace <spreadsheetId> "old" "new" --sheet Sheet1 --regex
gog sheets update-note <spreadsheetId> 'Sheet1!A1' --note ''
gog sheets append <spreadsheetId> MyNamedRange 'new|row|data'
gog sheets clear <spreadsheetId> 'Sheet1!A1:B10'
gog sheets clear <spreadsheetId> MyNamedRange

# Format
gog sheets format <spreadsheetId> 'Sheet1!A1:B2' --format-json '{"textFormat":{"bold":true}}' --format-fields 'userEnteredFormat.textFormat.bold'
gog sheets format <spreadsheetId> MyNamedRange --format-json '{"textFormat":{"bold":true}}' --format-fields 'userEnteredFormat.textFormat.bold'
gog sheets format <spreadsheetId> 'Sheet1!A1:B2' --format-json '{"borders":{"top":{"style":"SOLID"}}}' --format-fields 'userEnteredFormat.borders.top.style'
gog sheets merge <spreadsheetId> 'Sheet1!A1:B2'
gog sheets unmerge <spreadsheetId> 'Sheet1!A1:B2'
gog sheets number-format <spreadsheetId> 'Sheet1!C:C' --type CURRENCY --pattern '$#,##0.00'
gog sheets freeze <spreadsheetId> --rows 1 --cols 1
gog sheets resize-columns <spreadsheetId> 'Sheet1!A:C' --auto
gog sheets resize-rows <spreadsheetId> 'Sheet1!1:10' --height 36
gog sheets read-format <spreadsheetId> 'Sheet1!A1:B2'
gog sheets read-format <spreadsheetId> 'Sheet1!A1:B2' --effective

# Named ranges
gog sheets named-ranges <spreadsheetId>
gog sheets named-ranges get <spreadsheetId> MyNamedRange
gog sheets named-ranges add <spreadsheetId> MyNamedRange 'Sheet1!A1:B2'
gog sheets named-ranges add <spreadsheetId> MyCols 'Sheet1!A:C'
gog sheets named-ranges update <spreadsheetId> MyNamedRange --name MyNamedRange2
gog sheets named-ranges delete <spreadsheetId> MyNamedRange2

# Insert rows/cols
gog sheets insert <spreadsheetId> "Sheet1" rows 2 --count 3
gog sheets insert <spreadsheetId> "Sheet1" cols 3 --after

# Notes
gog sheets notes <spreadsheetId> 'Sheet1!A1:B10'
gog sheets links <spreadsheetId> 'Sheet1!A1:B10'   # Includes rich-text links

# Create
gog sheets create "My New Spreadsheet" --sheets "Sheet1,Sheet2"

# Tab management
gog sheets add-tab <spreadsheetId> <tabName>
gog sheets rename-tab <spreadsheetId> <oldName> <newName>
gog sheets delete-tab <spreadsheetId> <tabName>          # use --force to skip confirmation
```