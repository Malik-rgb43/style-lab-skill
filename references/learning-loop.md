# The learning loop: notes, lessons, rules, taste

Every note the owner gives is read once, measured, and filed at the right layer, so the next
video starts better without the owner repeating himself. Two scripts hold it, both writing to
the global store (`references/store-schema.md`):
- `scripts/lessons.py`: interpreted notes (lessons), their evidence, rules, exceptions,
  conflicts, settings and the skill feedback list;
- `scripts/taste.py`: every pick and rejection, and a summary that recommendations read.
Talk to the owner in Hebrew; the commands and the rule text stay as written here.

## 1. Read every note into one line, before any fix
After a round's notes arrive (`references/review-loop.md`, "Reading notes"), show the owner one
line per note, all in one message, before fixing anything:
```
<n>. "<note, his words>" → <what I understood, measurable> → <where it applies> → <layer>
```
- **What I understood** is a change someone else could check: numbers, curve names, frames,
  px, dB, a named element. Never "better", "smoother", "more alive".
- **Where** is a time or range, an element, "all entrances in this video", "every video in
  this style", "all of this client's videos".
- **Layer** is `video`, `style <slug>`, `client <slug>` or `craft quality|taste` (section 2).
- **Unsure?** Say so on the line and give the two readings: `קריאה א: … / קריאה ב: …`. Fix
  that note only after the owner picks; fix the clear ones meanwhile.
- The fix follows the stated reading and nothing else. If the owner corrects a line, the
  corrected line is the reading (and the lesson text).

Worked example (the owner's own note):
```
1. "המהירות ליניארית, צריך להתחיל מהר ולסיים לאט" → entrances of text and objects use an
   ease-out curve (fast start, long settle, e.g. expo.out over 12-18 frames), not linear →
   all entrances in this video → craft / taste
```

## 2. Routing: which layer a note belongs to
Precedence, lowest to highest: craft baseline (`references/craft-rules.md`) < craft learned
(`craft-rules.learned.md` in the store) < style ("Review-round rules" in the style's STYLE.md
and skill) < client (`clients/<slug>/lessons.md`) < this video. A higher layer wins where two
disagree.
| The note is about | Layer | Examples |
|---|---|---|
| this content only | `video` | "move this chip", "delete the dog", "the caption at 0:12 is late" |
| a brand's taste or content | `client` | "no red", "this client hates emojis", "never show money falling" |
| how this style looks or moves, where another style could rightly differ | `style` | "headlines enter from the bottom edge", "paper texture at 12 %" |
| good editing anywhere | `craft` | linear curves, pops, clipped syllables, unreadable contrast |
Craft lessons also get a **kind**: `quality` (objectively wrong anywhere: a clipped syllable,
text under 4.5:1 contrast, a one-frame pop) or `taste` (a craft default a style may rightly
change, such as ease-out entrances). Only taste craft rules can have style exceptions.
When the layer is unclear, say so on the note line and ask; never file a client's taste as craft.

## 3. Saving a reading: `add` or `match`
Before saving, look for the same lesson (by meaning, not words):
```
python scripts/lessons.py list --layer <layer> [--target <style|client>]   # also --status, --json
python scripts/lessons.py show <id>
```
- **Same meaning as a live lesson** (candidate, proposed or rule): `match` it, and tell the
  owner which lesson and why ("זו אותה הערה כמו L0007: שתיהן על כניסות ליניאריות"):
  ```
  python scripts/lessons.py match <id> --project <P> --video <V> --t <sec> --round <N> --note "<his words>" [--before <frame.png>] [--after <frame.png>]
  ```
- **New**: `add` it:
  ```
  python scripts/lessons.py add --text "<his words>" --measurable "<the reading>" --layer craft|style|client|video [--target <style|client>] [--kind quality|taste] --scope "<where>" --project <P> --video <V> --t <sec> --round <N> [--before ..] [--after ..]
  ```
  `craft` needs `--kind` and no target; `style` and `client` need `--target`; `video` has
  neither and is saved as `one-off` (history only, never a rule; save it only when it may
  recur, and change its layer later with `edit <id> layer=...` if it does).
- `add` warns when the same text was rejected before: do not propose it again; tell the owner.
- Two lessons that turn out to mean the same: `lessons.py merge <src> <into>` (evidence moves,
  `src` becomes `merged`; a rejected `src` cannot be merged). A `match` or `merge` into a rule
  re-renders its rule file (the evidence count), all or nothing.
- Fix a wrong reading with `lessons.py edit <id> measurable="..."`. `text`, `measurable` and
  `scope` can be edited on any lesson, rules too (a rule's file is re-rendered); `target` and
  `kind` until it is a rule; `layer` only on a candidate or one-off. Drop a wrong candidate with
  `lessons.py reject <id> --reason "..."` (kept, so it is not proposed again).

## 4. Candidate → proposed → rule
- A new non-video lesson is a **candidate**. It becomes **proposed** when its evidence spans 2
  different (project, video) pairs or 2 different rounds. The same note recorded twice in one
  round (same project, video, time, round) counts once.
- Only the owner makes a rule. At the end of each round, list
  `lessons.py list --status proposed`, show each as one line (reading, scope, layer, evidence
  count) and ask. On his yes:
  ```
  python scripts/lessons.py approve <id>
  ```
- A **candidate** (one note so far) becomes a rule only when the owner explicitly says so
  ("make it a rule now"): `lessons.py approve <id> --one-note`.
- **Conflicts.** `approve` first prints possibly conflicting rules and exits 1 when there are
  any. Listed: rules of a lower layer or of the same layer and target that have an empty scope
  (everywhere), share a scope word, or have the same style or client target. Judge each: if none really
  clashes, run `approve <id> --yes-no-conflict` and say so; if one does, never pick a side
  yourself: show both to the owner and ask (reword the new lesson with `edit`, reject it, or,
  against a taste craft rule, make it a style exception). `lessons.py conflicts <id>` shows the
  same list without approving. An exception never conflicts with the craft rule it overrides.
- **Auto-promote.** Only when the owner asks for it: `lessons.py settings auto_promote=true`.
  Proposed lessons then become rules at once (on `match` or `merge`), without the conflict
  check; say that to the owner. `lessons.py settings` prints the current value.
- An approved rule applies to every video in its scope: fix every occurrence in the current
  project's videos, not only the noted time.

## 5. Exceptions (a style that differs from a taste craft rule)
When a style rightly does something a **taste** craft rule forbids (a deliberately linear,
mechanical style), record the exception with its reason instead of a clashing style rule:
```
python scripts/lessons.py exception --style <s> --craft <craft rule id> --measurable "<what this style does instead>" --reason "<why this style differs>"
```
It is a style-layer candidate: approve it like any lesson (with `--one-note` on the owner's yes,
since it has one note). `rules` then marks the craft rule as overridden for that style. A
**quality** craft rule cannot be overridden: tell the owner and fix the edit instead.

## 6. Where rules are written
`approve` (and `edit` on a rule) renders the rule list of that layer and target:
- **style**: between `<!-- style-lab:learned-rules:start <slug> -->` and
  `<!-- style-lab:learned-rules:end <slug> -->` in the style's STYLE.md and its style skill (the
  pair sits in the "Review-round rules" section, `references/style-skill-template.md`). One
  block per style: two styles sharing a pack (a style and its English twin) each keep their
  own block, and a new block goes right after the last one. An empty old pair without a slug is
  taken over. Hand-written text outside the markers is never changed. Each file is backed up once a day
  (`<file>.bak-YYYYMMDD`); the write is all or nothing (a failure restores every file and saves
  nothing); `styles.json` gets the style's `rules_count` and `version` + 1. A missing style
  skill file is a warning (STYLE.md is still written); a missing STYLE.md or damaged markers
  stop the write: fix by hand, then `approve` again.
- **craft**: `craft-rules.learned.md` in the store (whole file generated).
- **client**: `clients/<slug>/lessons.md` in the store (whole file generated).
If a message says the rule files and the registry are out of step:
`python scripts/lessons.py render --layer craft|style|client [--target <s|c>]`.
Never edit learned rules between the markers by hand; use `edit` on the rule.

## 7. Using the rules: before planning and before building
At B2 (before the plan) and again at B4 (before building), read every rule that applies:
```
python scripts/lessons.py rules --style <s> --client <c>        # --json for the fields
```
It lists craft, then style, then client rules by number; a taste craft rule a style overrides
is marked in the `overridden` column, with the exception's reason below. Apply them in the
precedence of section 2 and cite a learned rule in the plan by its lesson id ("L0007"), never
by its number: STYLE.md files have hand-numbered rules too, so "style rule 5" is ambiguous.
With no client: `--style <s>` alone.

## 8. Taste: log every pick, read it before recommending
At every choice the owner makes, log the pick and what he passed over:
```
python scripts/taste.py log --client <c> --style <s> --step plan|frames|motion|palette|idea --picked <option id> --rejected <id> --rejected <id> --project <P> --tags picked=<tag>,<tag> rejected=<tag>,<tag> [--note ".."]
```
- `plan`: each beat's pick at B3 (one call per beat); `frames` / `motion`: frame or motion
  variants he chose between; `palette`: the palette pick at stage C4; `idea`: an idea he took
  or turned down. Leave out `--client` when the video has none.
- Tags are short, reusable words for what the option was (`warm`, `3d`, `paper`, `neon`,
  `full-screen`, `ease-out`), so they add up across projects.
Before marking **ההמלצה שלי** (B2 options, C4 palettes, any choice board):
```
python scripts/taste.py summary --client <c> --style <s>        # --last N, --json
```
`prefers` = tags picked 3+ times and at least twice as often as rejected; `avoids` = tags
rejected 2+ times and never picked; `rejected_twice` = option ids turned down in 2+ projects.
Lean the recommendation toward `prefers`, away from `avoids`, and never recommend a
`rejected_twice` option. When taste decided a recommendation, say so in one line ("המלצתי על
זה כי בפרויקטים הקודמים בחרת רקעים חמים ודחית ניאון").

## 9. Improving the skill itself, and the end of a project
- Whenever a step of this skill fails, a check is missing or a page confuses the owner, note it:
  `python scripts/lessons.py feedback add --text "<what went wrong, what would fix it>" --step <B4 etc.> [--project <P>]`.
- At the end of a project (after the final):
  1. `lessons.py list --status proposed` (and `--status candidate`): offer the open ones for
     approval (section 4).
  2. `python scripts/lessons.py feedback list`: offer the top items as **concrete** edits to the
     skill (which file, which step, the new wording). Never edit the skill's files without the
     owner's yes; after an item is done, `lessons.py feedback done <n>`.
  3. Always offer new ideas too: new beats worth keeping for the ideas bank, palette directions
     for the client, style variants worth training.
