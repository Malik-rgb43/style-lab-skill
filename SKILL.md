---
name: style-lab
description: Train Claude on any number of named video editing styles from reference videos, then edit new clips in the style the user picks. Two phases taken step by step from a proven workflow. TRAIN: about 20 reference videos plus a visual style board (a Claude artifact in place of Figma), a word-by-word and frame-by-frame breakdown, a library of every animation (still, frame strip, clip, notes), packaged as a new style skill. EDIT: millisecond transcript, silences cut to 30 ms, an interactive edit-plan document with three recommended designs per beat, a human plan, then a HyperFrames build, numbered revisions and sound design. Use this whenever the user wants to create, define, copy, learn or train an editing style or look, build a style board or style guide for videos, turn reference reels into a style skill, or edit a talking-head or short-form clip "in the style of" a trained style. Hebrew triggers - סגנון עריכה, ליצור סגנון, ללמד את Claude סגנון, סקיל סגנון, בורד סגנון, לוח השראה, ריפרנסים, תערוך בסגנון, מסמך תכנון עריכה, סטייל לסרטונים.
---

# style-lab

A complete, repeatable way to (A) teach Claude an editing style and (B) edit clips in it.
It reproduces, step by step, the workflow in Joseph's video "How to Edit VIRAL Talking Head
Videos Using ONLY Claude! (New Minimal Style)" (YouTube, 2026-10-31). Two deliberate
changes, both the user's choice: the style board lives in a Claude artifact instead of Figma,
and the edit is built in **HyperFrames** instead of Remotion (the video's engine). Every
prompt the video types is in `references/source-workflow.md`, verbatim, with its timestamp;
where a prompt says "Remotion", run it with HyperFrames. A third default: every style keeps
the reference's layout and motion but uses the brand fonts and colours
(`references/brand-defaults.md`) unless the user says otherwise.

The core idea, in the video's words: **the reference videos are how Claude learns the logic
behind the animations, and the board is how Claude learns the design.** Without both, you
get generic "AI slop" edits. And: if you don't understand what makes a style good, AI will
only help you make bad videos faster. So the human's taste sits at two checkpoints, the
board (phase A) and the edit plan (phase B), and Claude does the heavy lifting around them.

## How to talk to the user
Explain each step in plain language before doing it (what we are doing, why, what the user
must do now), one step at a time, and wait at the two human checkpoints. The user may be
learning this workflow to teach it, so name the step ("Step A2 — the style board") every time.
Write to the user in their language (Hebrew if they write Hebrew).

## Your styles (several named styles side by side)
style-lab holds any number of styles. Each has a **name the user chooses** (Hebrew is fine,
e.g. "מינימל פרימיום"), a slug (latin, e.g. `minimal-premium`), a one-line description, its
own style pack and its own style skill `style-<slug>`. All are listed in one registry,
`~/.claude/style-lab/styles.json`, managed with `scripts/styles.py`:
```
python scripts/styles.py list                       # the styles you can pick from
python scripts/styles.py add <slug> --name "<name>" --pack <project>/styles/<slug> --description "..."
python scripts/styles.py show <slug or name>
python scripts/styles.py set <slug> name="<new name>" status=ready aliases="מינימל,פרימיום"
python scripts/styles.py check                      # every pack and skill still exists
```
Status: `training` while phase A runs, `ready` once A5 is done (only ready styles are offered
for editing), `archived` to hide one without deleting anything. `remove` deletes only the
registry entry, never the files. Each style also records `fonts` and `colors`: `brand` (the
default, `references/brand-defaults.md`) or `reference` (keep the reference's own).
And a `theme`: `dark` (default, the site's palette), `light` (the white option) or `both`; the
user can also ask for one video in the other theme ("make this one white") in B0.

**Which phase?** "new style / ליצור סגנון / ללמד סגנון" → phase A for a new style.
"edit / make a video / תערוך / תכין סרטון" → phase B with a chosen style. "change / rename /
improve style X" → `styles.py set` and, for rule changes, edit that style's STYLE.md and skill.

## Where things live
```
<project>/styles/<style-slug>/          ← one folder per style (the "style pack")
  refs/            the reference videos (local only, never published or redistributed)
  analysis/        per-video breakdowns: words.json, sheets, notes
  board/           board.json + board.html (the published style board artifact) + thumbs/
  library/         V01-A06 <title>/{still.png, strip.jpg, clip.mp4, notes.md} + index.json
  animations.json  the list Claude writes during the breakdown (input to the library script)
  STYLE.md         the trained style: numbered rules, tokens, categories, motion, sound
<project>/edits/<clip-slug>/            ← one folder per edit
  raw/  words.json  edit_ready.mp4  edit_ready.words.json  plan.json  plan.html
  hyperframes/  assets/  renders/v1.mp4 …  revisions.md
~/.claude/skills/style-<slug>/SKILL.md  ← the generated style skill (points to the pack)
~/.claude/style-lab/styles.json         ← the registry of all styles (name, slug, status, paths)
```
Scripts (run with `python`, all local and free): `scripts/styles.py`, `scripts/transcribe_words.py`,
`scripts/cut_silences.py`, `scripts/build_reference_library.py`. Templates:
`assets/board-template.html`, `assets/plan-template.html`.

---

## Phase A — Train a style (build the style pack)

### A1. Collect the reference videos
Goal: about 20 of the creator's best-edited videos in exactly the style wanted (the video
picks "20 of the most premium edited videos" from one creator's Instagram).
- First ask what to **call the style** (the name the user will pick it by) and for a one-line
  description; derive a slug and register it right away:
  `python scripts/styles.py add <slug> --name "<name>" --pack <project>/styles/<slug> --description "..."`
  (status `training`). Ask too whether to keep the brand fonts and colours (default) or the
  reference's own; record it with `--fonts` / `--colors`.
- Ask the user for the creator and where the videos are. The user downloads
  them (or gives links they are allowed to fetch); keep them in `refs/` as local study copies.
  They are never published, uploaded or put in an artifact.
- Name them `V01.mp4 … V20.mp4` and keep a `refs/sources.md` (V01 → original link/title).
- Fewer than ~12 videos gives a thin library; say so, and continue if the user wants.

### A2. The style board (the most important part of training)
The video's human does this alone in Figma: watch every video, screenshot every scene they
like (~130 pieces), sort them into the style's categories, then zoom into one example per
category and break it down: **what makes it up? the fonts, the colours, the sizes, the
patterns** — and for every scene ask **why this design, why these colours, why these effects,
why these animations**. Here Claude does the legwork and the user supplies the taste:
1. Run the frame analysis on every reference (`video-analysis` skill / `tools/analyze.py`
   if the editing toolkit is installed, otherwise extract one frame every 0.5 s with ffmpeg)
   and read the contact sheets.
2. Propose the categories. Start from the ten in `references/example-style-minimal-premium.md`
   only as a checklist; the style's own categories win (merge, rename, add).
3. Pick candidate inspiration frames (aim 100–150), export them as 9:16 JPG thumbs
   (~360 px wide) into `board/thumbs/`, and choose one hero frame per category to annotate.
4. Fill `board.json` (schema in `references/board-and-plan-spec.md`): tokens (fonts,
   colours with hex picked from frames, textures, shapes, sizing, layout), and per category
   the tile list, one annotated frame with 3–5 callouts (each a *measurable* rule like "the
   light sans line spacing is 75 % of the bold word"), and draft answers to the four "why"
   questions.
5. Build `board.html` from `assets/board-template.html` (replace the JSON block) and publish
   it as an artifact with the thumbs as supporting files (`files`), so the user can scan it,
   rewrite the "why" answers, and press **Copy board notes** to paste them back.
6. **Checkpoint 1 — stop and wait.** The user reviews the board: wrong categories, missing
   scenes, rules that are not true. Apply their notes to `board.json` and republish. Do not
   start A3 until the user says the board is right; the board is the design half of the
   training and a wrong board trains a wrong style.

### A3. The training breakdown (the prompt the video uses)
Run exactly what the video's training prompt asks (verbatim in `references/source-workflow.md`):
for **every** reference video,
1. a word-by-word transcript (`scripts/transcribe_words.py`, language from the user or `auto`);
2. a frame-by-frame breakdown that finds every frame with an animation, and for each
   animation studies three things:
   - **placement** — where it sits in the script: which words trigger it, how long after the
     word, what kind of line gets which kind of scene (announce vs. describe, number,
     person, story, punchline);
   - **design** — how the frame is composed and what elements make it up (ground, texture,
     type pair and sizes, colour, line work, illustration style, position);
   - **logic** — how long it lasts and what it feels like: frame counts, distances, easing
     ("slow-fast-slow", "snaps", "wipes letter by letter in 15 f"), camera moves.
   Read the frames to measure; never guess durations. One worked example of the precision
   wanted is V01-A06 in `references/example-style-minimal-premium.md`.
3. Then read the board and reconcile: the board says *what* the design is and *why*; the
   videos say *when* and *how it moves*. Where they disagree, ask the user.
Write each animation into `animations.json` (id `V<video>-A<n>`, start/end seconds,
keyframe, title, category, scene type, words, notes). Do one video at a time and save as you
go; with many videos, split the work across subagents (one video each, same schema).

### A4. Build the reference library
```
python scripts/build_reference_library.py styles/<slug>/animations.json --out styles/<slug>/library
```
Every animation becomes a folder with `still.png`, `strip.jpg` (every frame labelled with
frame number and ms), `clip.mp4` and `notes.md`, plus `library/index.json`. The video's run
produced 225 animations from 20 videos. Open a few strips and check the boundaries are right
(the animation starts and ends inside the clip); fix `animations.json` and rerun with
`--force` for any that are off. Add the library tiles' stills/clips/strips to the board.

### A5. Package the style as a skill
Write `STYLE.md` and generate the style skill from `references/style-skill-template.md`:
- **numbered rules** (§1, §2 …) so plans and revisions can cite them ("rule 13");
- tokens (fonts, colours, textures, shapes, sizing, layout). **Fonts and colours default to the
  brand** (`references/brand-defaults.md`: Karantina 700 / Heebo / Consolas, violet accent):
  map the reference's type pair and accent onto them and keep its sizes, ratios and colour
  discipline; record the reference's originals next to them. Use the reference's own fonts or
  colours only if the user asks for that style; each category with its design rule, motion recipe, placement rule
  and 2–3 library refs; the planning patterns found in A3 (e.g. "announce on face, describe
  with a visual"); the sound profile (music feel, which actions get which sound);
- the absolute path to the style pack and library.
Optionally (recommended) build the style's recurring parts as a HyperFrames component kit in
`<pack>/kit/` (layout, captions, headline, anchor list, accents), verified with `hyperframes check`
and snapshots, so every edit starts from tested blocks.
Save it to `~/.claude/skills/style-<slug>/SKILL.md` (its description names the style's display
name, so "edit in <name>" finds it), then mark it ready:
`python scripts/styles.py set <slug> status=ready board_url=<board artifact URL>`.
Tell the user the style's name and that it now appears in the list when they make a video.
That skill is the style's "brain"; editing (phase B) always loads it.

---

## Phase B — Edit a clip in a trained style

Run on the strongest model and effort available (the video uses Opus 5.5, "Extra"). The
starting prompt the video uses is in `references/source-workflow.md`; if the user just says
"edit this in <style>", follow the same steps.

### B0. Choose the style
Run `python scripts/styles.py list`. If the user named a style (by name, slug or alias),
use it; if they did not, **ask which one** with the ready styles as options (name + one-line
description; with a single ready style, say which one you will use). No ready style yet →
say so and offer phase A. Load the chosen style's skill (`style-<slug>`) and work in its
pack. Write the choice into `edits/<clip>/plan.json` (`style`) so revisions use the same one.
To compare, the user may ask for the same clip in two styles: run B1 once and B2 onward per
style, in `edits/<clip>/<style-slug>/`.

### B1. Transcript and silence cut (≈15 min in the video, all Claude)
1. Load the style skill. 2. `scripts/transcribe_words.py raw.mp4 --out words.json` (ms-level).
3. `scripts/cut_silences.py raw.mp4 --words words.json --out edit_ready.mp4 --pad-ms 30`.
   Everything after this point uses `edit_ready.mp4` and `edit_ready.words.json`.
4. Show the user the low-confidence words (names, numbers, brands) and fix them before planning.

### B2. The edit-plan document
Split the script into **beats** (one idea or one line each; the video's 57 s clip has 15).
For each beat recommend **three designs pulled from the style's library**, the way the creator
would place them (the placement logic learned in A3), and mark one as **My pick**. Each option
carries: title, category · scene type (A-roll + captions / Overlay on A-roll / Cut-away scene),
a concrete description of what is on screen, and `Ref V##-A##`. Each beat also gets
"Seen in the reels" (why this fits, citing the reference videos and rule numbers) and a
camera note. Fill `plan.json`, build `plan.html` from `assets/plan-template.html`, publish
it with a 540p proxy of `edit_ready.mp4` as `clip.mp4` and the option stills as `thumbs/`.
Full spec: `references/board-and-plan-spec.md`.

### B3. **Checkpoint 2 — the human plans the edit (most important step of the edit)**
"Claude can build the animations easily, but without a high-quality edit plan they won't come
together." AI left to plan alone is not creative enough, so the user plans: picks a design per
beat or writes their own note, then presses **Copy Plan** and pastes it back. Offer the
video's planning patterns as advice (`references/source-workflow.md` → "How the plan was
made"): open on the speaker so the viewer connects; make the first visual a strong hook;
when a new idea is announced, stay on the face; when it is described, show it; a person →
character callout; a story → b-roll; balance the scenes so the video flows. Wait for the plan.

### B4. Build (≈60 min in the video, all Claude)
1. Read the pasted plan; for every chosen design open its library folder (clip, strip, notes)
   and write down the exact design to copy.
2. Make an **asset list** (every painting, photo, b-roll, icon) with a source per item, and
   show it. Fetch with the browser or local files: public-domain art (Wikimedia/Wikipedia,
   check the licence on the file page), stock b-roll (Pexels/Pixabay), photos the user owns.
   If something can't be found, ask the user for it rather than inventing it.
3. Build in **HyperFrames** (load the `hyperframes` skill first; it routes to
   `hyperframes-core` for the composition contract and `hyperframes-animation` /
   `hyperframes-keyframes` for motion). One 1080×1920 composition in `hyperframes/`, with
   `edit_ready.mp4` as the A-roll track. Fonts and colours: the style's tokens, which are
   the brand defaults unless the style says otherwise (`references/brand-defaults.md`, with
   the Google Fonts URL for Karantina and Heebo).
   If the style pack has a `kit/` (a component generator such as `kit/gokit.py`, see its README),
   build from its components first and hand-build only what it does not cover.
   Order, as in the video: A-roll captions for the whole video first with the style's caption
   rule, timed from `edit_ready.words.json`; then each scene in plan order as its own
   sub-composition, following its reference's notes (durations in frames, easing, textures,
   camera moves). Validate the composition before rendering.
4. Render the full video (1080×1920) to `renders/v1.mp4` with the HyperFrames CLI (if the
   editing toolkit is installed, its `render-qa-delivery` skill runs the cheap-first render and
   frame QA) and send it. Say plainly that a first draft is about 60–70 % of the way there and
   that the user's notes finish it.

### B5. Revisions (≈10 min)
Ask for numbered notes with timestamps (the video's six notes are the model: "Rough cut
between the graph and the next shot. Hold the graph text until 0:06, then cut."). Apply every
note, log them in `revisions.md`, render `v2`, then ask: **"Is the video ready for sound
design, or are there more revisions?"**

### B6. Sound design (≈15 min)
Match the sound to the reference videos (re-listen to their audio analysis): a cinematic music
bed under the voice; a click for each text appearance; a highlighter/marker sound for graph and
line draws; whooshes for transitions and camera moves. Place each effect on its exact frame
(the video placed 74 effects on 57 s). Keep the voice on top. Only use music and effects the
user is licensed to use (their library, royalty-free sources); never claim a track is cleared
if you don't know. Render the final and report what was placed.

## Time budget the video reports (for setting expectations)
Cuts + planning document 15 min · human planning 15 min · building the edit 60 min ·
revisions 10 min · sound design + music 15 min. "Next time: half the time."

## Guardrails
- Reference videos are study material: local only, never redistributed, never placed in a
  published artifact (stills/thumbs on a private board are fine; full clips stay local unless
  the user asks).
- Board and plan artifacts are private until the user shares them.
- Claims of the source video (prices, income) are not part of this skill.
- If a step's tool is missing (ffmpeg, faster-whisper, HyperFrames, a browser), say which step is
  blocked and what to install; don't silently skip a step.

## References
- `references/source-workflow.md` — the source video, step by step with timestamps and every prompt verbatim. Read before running either phase the first time.
- `references/board-and-plan-spec.md` — `board.json` and `plan.json` schemas, publishing the two artifacts, what each control does.
- `references/style-skill-template.md` — the template for the generated style skill (A5).
- `references/brand-defaults.md` — the brand fonts (Karantina 700, Heebo, Consolas) and colour tokens (violet accent, light and dark), and how to map a reference's type pair and palette onto them. Read at A2 and A5, and before building in B4.
- `references/example-style-minimal-premium.md` — the worked example: the ten categories and rules of the minimal premium style taught in the video.
