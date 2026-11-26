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
    lessons.md                  generated: the client's approved rules
  lessons/lessons.json          every interpreted note (the learning loop)
  craft-rules.learned.md        generated: approved craft rules
  settings.json                 learning-loop settings (auto_promote)
  taste/<client slug>.jsonl     every pick and rejection (`_none.jsonl` for no client)
  skill-feedback.md             problems in the skill's own process, as a checklist
  capabilities.json             the connections check (Stage 0)
  lexicon/<lang>.json           spelling fixes for the cut pipeline {"heard word": "spelling"},
                                managed by `cutlang.py lexicon add|list`
  brand.md                      optional: the user's own default brand
```
Later versions add `ideas/`, `projects.json` and `studio/` here; nothing in the skill folder
holds user data. Footage, builds and renders stay in the user's own project folders.

**Style pack and project folders** (where phase A and phase B write):
```
<pack>/  refs/ sources.md  analysis/  board/ (board.json board.html thumbs/)  animations.json
         library/<V01-A06 title>/  STYLE.md (numbered rules, tokens, motion, sound)  kit/
<project>/edits/<clip>/  raw/ words.json edit_ready.mp4 edit_ready.words.json plan.json
         plan.html build/ assets/ renders/ revisions.md SOURCES.md
~/.claude/skills/style-<slug>/SKILL.md   the generated style skill
```
`refs/` videos are local study copies, never published.

Rules for every script that writes here (`store.py` does this for all of them):
- one writer at a time: a `<file>.lock` next to the file, taken with retry; a lock older than
  60 s is treated as stale. A writer that waits 10 s exits with code 2 and one line naming the
  lock: another session is writing, so retry; delete the lock only if no session is running;
- atomic writes: a temp file in the same folder, then replace, so a crash never leaves half a
  file; JSON is UTF-8, Hebrew kept as is;
- `STYLE_LAB_HOME=<folder>` moves the whole store (tests, a portable setup); a relative folder
  is made absolute from the current folder. For styles only,
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
| version | whole number | bump it when the rules change; `lessons.py` adds 1 whenever it rewrites the style's learned rules |
| rules_count | whole number | approved style-layer rules (set by `lessons.py`) |
| aliases | list | other names that find it; never another style's slug, name or alias |
| board_url, notes | text | |
| created / updated | ISO dates | set by the script |

**Linked clients are not stored here.** Links live on the client side (`client.json` →
`styles`), one source of truth; `styles.py list` and `styles.py show` compute them on read.

Lookup (`show`, `set`, `remove`, `clients.py link`) matches the slug first across all styles,
then the name, then the aliases, case-insensitive, so a slug always wins. A name must not be
empty, and an alias that is another style's slug, name or alias is refused.

`python scripts/styles.py migrate` upgrades a v1 registry in place: it sets `version: 2`, adds
the missing fields with defaults (`kind trained`, `aspect 9:16`, `name_source user`,
`version 1`, `rules_count 0`) and writes `styles.json.bak-YYYYMMDD` first. Reading a v1 file
upgrades it in memory only; any write through `styles.py` leaves a v2 file with the backup.
`styles.py check` lists a missing pack or skill (a problem for `ready` styles, expected while
`training`); offer to relink (`set <slug> pack=...`) or archive it. `check` (on stdout) and
`list` (on stderr, after the rows) also print one line per client link to a style that is not in
the registry: `dangling: client <name> -> style <slug> (missing style)`. A dangling link never
changes the exit code; only a missing path on a `ready` style fails `check`. `styles.py remove`
says `N client(s) still link to it`; offer `clients.py unlink <client> <slug>` or a relink.

## clients.json and client.json
`clients.json` only lists who exists:
```json
{"version": 1, "clients": [{"slug": "example-shop", "name": "...", "dir": "clients/example-shop"}]}
```
`clients/<slug>/client.json` is the profile:
| Field | Type | Meaning |
|---|---|---|
| slug, name | text | id and display name |
| aliases | list | other names that find the client; never another client's slug, name or alias |
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

Lookup matches the slug first, then the name, then the aliases (as for styles). A name must
not be empty.

`clients.py remove` takes the client out of `clients.json` only; the folder stays, and
`clients.py restore <slug>` brings it back. `add` never overwrites a kept folder. `restore`
registers the folder's name as the slug: when `client.json` inside says another slug (a copied
or renamed folder), it is rewritten to the folder name; a slug that is already registered is
refused. A restored client with the same name as another is registered with a note to rename one.
`clients.py set <client> brand_colors=...` that changes the first colour of a client with a saved
palette prints a note to rerun `clients.py palette <client> --pick <kind>`.

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

## lessons/lessons.json (the learning loop)
Written only by `scripts/lessons.py`; how it is used: `references/learning-loop.md`.
```json
{"version": 1, "next_id": 3, "lessons": [ { ...one entry per lesson... } ]}
```
| Field | Values | Meaning |
|---|---|---|
| id | `L0001`, `L0002` … | never reused |
| text | text | the owner's note in his words (for an exception: generated from the reason) |
| measurable | text | Claude's measurable reading; this is the rule text once approved |
| layer | `craft` \| `style` \| `client` \| `video` | section 2 of the learning loop |
| target | slug or null | the style (style layer) or client (client layer); null for craft and video |
| kind | `quality` \| `taste` \| null | craft lessons only |
| scope | text | where it applies ("all entrances"); empty = everywhere |
| evidence | list | `{project, video, t, round, note, before, after, added}`; one (project, video, t, round) counts once |
| status | `candidate` \| `proposed` \| `rule` \| `rejected` \| `merged` \| `one-off` | video lessons are `one-off` |
| rule_no | whole number or null | its number within its layer and target, set on approval |
| exception_of | lesson id or null | the taste craft rule a style exception overrides |
| reason | text | why a lesson was rejected, or why an exception differs |
| merged_into | lesson id or null | set when merged |
| created / updated / promoted | ISO dates | `promoted` = the day it became a rule |

Generated rule files: `craft-rules.learned.md`, `clients/<slug>/lessons.md` (whole files) and,
for a style, its own block between `<!-- style-lab:learned-rules:start <slug> -->` and
`<!-- style-lab:learned-rules:end <slug> -->` in its STYLE.md and style skill (one block per
style, so styles sharing a pack keep separate blocks). Each written file gets a
`<file>.bak-YYYYMMDD` backup once a day.

## settings.json
`{"auto_promote": false}`. `lessons.py settings auto_promote=true` (only on the owner's request)
turns proposed lessons into rules at once, without the conflict check; `lessons.py settings`
prints the values.

## taste/<client slug>.jsonl
One JSON object per line, appended by `taste.py log`; `_none.jsonl` holds picks made for no
client:
```json
{"date": "2026-01-01", "client": "example-shop", "style": "paper-3d", "step": "plan", "project": "launch-reel",
 "picked": "b03-opt2", "rejected": ["b03-opt1", "b03-opt3"],
 "tags": {"picked": ["warm", "paper"], "rejected": ["neon"]}, "note": ""}
```
`step` is `plan` | `frames` | `motion` | `palette` | `idea`. `taste.py summary` reads the last 50
entries (`--last`), optionally of one style (`--style`), and returns `prefers`, `avoids` and
`rejected_twice`.

## skill-feedback.md
A Markdown checklist written by `lessons.py feedback add|done`: `- [ ] <date> (step: B4,
project: P) <text>` for an open item, `- [x] …` once done. `feedback list` numbers the open
items; `feedback done <n>` uses that number.
