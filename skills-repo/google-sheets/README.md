# Google Sheets skill

This skill drives Google Sheets through the [`gogcli`](https://github.com/steipete/gogcli) command `gog`. Because leo's `bash` tool runs inside a bubblewrap sandbox with `--unshare-all`, gogcli cannot reach the system keyring (GNOME Keyring / Secret Service) or open a browser. Use the **file-backed keyring** instead.

## One-time setup (on the host)

1. Install `gog` and make sure it is on `PATH` (leo resolves symlinks under `~/.local/bin`, so a symlink into e.g. `~/git/others/gogcli/bin/gog` works).

2. Create an OAuth client in the Google Cloud Console (Credentials → Create Credentials → OAuth client ID → **Desktop app**) and download the JSON. Register it with gogcli:

   ```bash
   gog auth credentials ~/Downloads/client_secret_*.json
   ```

   This writes `~/.config/gogcli/credentials.json`.

3. Authenticate your account using the file keyring backend (not the system keychain):

   ```bash
   export GOG_KEYRING_BACKEND=file
   export GOG_KEYRING_PASSWORD='<choose a strong passphrase>'
   gog auth add <you>@gmail.com
   gog auth list   # verify
   ```

   This writes encrypted tokens under `~/.config/gogcli/keyring/`.

## Environment variables leo forwards into the sandbox

The bash tool forwards these from the host shell into bwrap when set:

| Variable | Purpose |
| --- | --- |
| `GOG_KEYRING_BACKEND` | Must be `file` for sandbox use. |
| `GOG_KEYRING_PASSWORD` | Required — decrypts the refresh token. Without a TTY, gog cannot prompt. |
| `GOG_ACCOUNT` | Optional — default account when you have more than one. |

Export them in the shell that launches `leo` (e.g. in your shell rc). Inside the sandbox, `XDG_CONFIG_HOME` is set to your real `~/.config` so gog finds `credentials.json` and the keyring directory even though `HOME=/tmp`.

## Quick check

```bash
gog auth list
gog sheets metadata <spreadsheetId>
```

If you see `no TTY available for keyring file backend password prompt`, `GOG_KEYRING_PASSWORD` is not set in leo's environment. If you see `OAuth client credentials missing`, `credentials.json` is not under `~/.config/gogcli/`.
