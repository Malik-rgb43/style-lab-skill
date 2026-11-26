# The global store: layout and schemas

Everything the user makes with this skill lives in one folder outside the skill, so updating or
reinstalling the skill never wipes a style, a client or a setting:
```
~/.claude/style-lab/            (or the folder in STYLE_LAB_HOME)
  styles.json                   style registry v2
  styles.json.bak-YYYYMMDD      backup written once before a v1 registry is upgraded
  styles/<slug>/                default home for NEW style packs (older packs stay where they are)
  clients.json                  client registry
  clients/<slug>/
    client.json                 the brand profile
    logo/  fonts/               files the user gave (copies, never links)
  capabilities.json             the connections check (Stage 0)
  brand.md                      optional: the user's own default brand
```
Later versions add `ideas/`, `lessons/`, `projects.json`, `taste/` and `studio/` here; nothing
in the skill folder holds user data. Footage, builds and renders stay in the user's own project
folders.

Rules for every script that writes here (`store.py` does this for all of them):
- one writer at a time: a `<file>.lock` next to the file, taken with retry; a lock older than
  60 s is treated as stale;
- atomic writes: a temp file in the same folder, then replace, so a crash never leaves half a
  file; JSON is UTF-8, Hebrew kept as is;
- `STYLE_LAB_HOME=<folder>` moves the whole store (tests, a portable setup). For styles only,
  `--registry <file>` or `STYLE_LAB_REGISTRY` points at another registry file.
- Never edit these files by hand while a script may run; use the commands.

## styles.json (registry v2)
```json
{"version": 2, "styles": [ { ...one entry per style... } ]}
```
| Field | Values | Meaning |
|---|---|---|
| slug | latin, digits, dashes | the id; made from the name when not given (Hebrew is transliterated) |
| name | any text, Hebrew fine | the name the user chose |
| description | text | one line |
| creator | text | whose references trained it (optional) |
| pack | absolute path | the style pack folder |
| skill | absolute path | the generated `style-<slug>/SKILL.md` |
| fonts / colors | `brand` \| `reference` | brand tokens or the reference's own |
| theme | `dark` \| `light` \| `both` | |
| status | `training` \| `ready` \| `archived` | only `ready` styles are offered for editing |
| kind | `trained` \| `described` \| `derived` | from videos, from a description, from another style |
| aspect | `9:16` \| `16:9` \| `1:1` \| `4:5` | |
| name_source | `user` \| `claude` | who chose the name |
| version | whole number | bump it when the rules change |
| rules_count | whole number | numbered rules in STYLE.md |
| aliases | list | other names that find it |
| board_url, notes | text | |
| created / updated | ISO dates | set by the script |

**Linked clients are not stored here.** Links live on the client side (`client.json` →
`styles`), one source of truth; `styles.py list` and `styles.py show` compute them on read.

`python scripts/styles.py migrate` upgrades a v1 registry in place: it sets `version: 2`, adds
the missing fields with defaults (`kind trained`, `aspect 9:16`, `name_source user`,
`version 1`, `rules_count 0`) and writes `styles.json.bak-YYYYMMDD` first. Reading a v1 file
upgrades it in memory only; any write through `styles.py` leaves a v2 file with the backup.
`styles.py check` lists a missing pack or skill (a problem for `ready` styles, expected while
`training`); offer to relink (`set <slug> pack=...`) or archive it.

## clients.json and client.json
`clients.json` only lists who exists:
```json
{"version": 1, "clients": [{"slug": "example-shop", "name": "...", "dir": "clients/example-shop"}]}
```
`clients/<slug>/client.json` is the profile:
| Field | Type | Meaning |
|---|---|---|
| slug, name | text | id and display name |
| aliases | list | other names that find the client |
| industry, audience, tone | text | tone = a few words ("חם, ישיר, בלי סלנג") |
| languages | list | e.g. `he, en` |
| logo | list | logo files inside `clients/<slug>/logo/` |
| fonts | `{heading, body, mono}` | Google Fonts names or files in `clients/<slug>/fonts/`; default Karantina 700 / Heebo / Consolas |
| brand_colors | list of `#RRGGBB` | what the client gave; the first one feeds the palette |
| palette | object or null | the picked proposal from `palette.py` (`references/palette.md`) plus `source_hex` and `picked` |
| theme | `dark` \| `light` \| `both` | |
| styles | list of `{slug, default}` | linked styles; at most one `default: true` |
| dos / donts | lists | brand rules ("no red", "never show money falling") |
| assets | list | product shots, UI captures (paths) |
| notes | text | |
| created / updated | ISO dates | set by the script |

`clients.py remove` takes the client out of `clients.json` only; the folder stays, and
`clients.py restore <slug>` brings it back. `add` never overwrites a kept folder.

## capabilities.json (Stage 0)
Written by `doctor.py`; the `session` block is written by `doctor.py session` and kept across
re-runs of the local check.
```json
{
  "checked": "2026-01-01T09:00:00",
  "local": {
    "ffmpeg": {"ok": true, "version": "ffmpeg version ...", "path": "...", "note": "", "hint": ""},
    "torch":  {"ok": true, "version": "", "hint": "", "cuda": "not probed"},
    "RUNPOD_API_KEY": {"ok": false, "version": "", "hint": "set the RUNPOD_API_KEY environment variable (paid cloud ASR)"}
  },
  "session": {"higgsfield": {"ok": true, "note": "MCP, paid credits"}},
  "stages": {"cut": {"uses": ["ffmpeg", "torch", "torchaudio", "faster_whisper"], "missing": [], "paid": []}}
}
```
- `local`: binaries (ffmpeg, ffprobe, node, blender, yt-dlp, gh), HyperFrames, Python modules
  (numpy, cv2, torch, torchaudio, faster_whisper, PIL) and env keys. Env keys are checked for
  presence only; a value is never read into the output or the file. `cuda` is `not probed`
  unless `doctor.py --deep` ran.
- `session`: MCP servers and skills Claude saw in its own tool list, recorded by Claude.
- `stages`: per stage, what it uses, what is missing and what costs money. The stage map is
  `STAGES` at the top of `scripts/doctor.py`: `train`, `train_profile`, `cut`, `frames`,
  `build`, `3d`, `cloud_asr` (paid), `publish`.
