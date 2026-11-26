---
name: style-lab
description: Learns a video editing style from reference videos or a description, keeps styles and clients (a brand profile with logo, fonts and a palette built from one colour) in one global store, and edits footage in a chosen style for a chosen client with an edit plan and review rounds. Starts every session with a connections check. Use when the user wants to train, save, learn or apply a reusable editing style, or add a client and its palette: style-lab, editing style, client, palette, סגנון עריכה, תלמד את הסגנון, תאמן סגנון, לקוח, לקוח חדש, פלטת צבעים. For a one-off "make mine like this reference" use reference-style-matching instead.
---

# style-lab

## When to use (full trigger list)
Use when the user wants to create, define, copy, learn or train an editing style or look from
reference videos (links or files) or from a written description; build a style board or style
guide; turn reference reels into a reusable style; or edit a talking-head or short-form clip
"in the style of" a trained style, then fix it in review rounds. Also for: "make my videos look
like this creator's", "save this look so every video matches", "the edit looks generic / AI".
Hebrew: סגנון עריכה, ליצור סגנון, ללמד את Claude סגנון, בורד סגנון, לוח השראה, ריפרנסים,
תערוך בסגנון, תוכנית עריכה, סבב הערות, סטייל לסרטונים.

**סקיל = הוראות קבועות ל־Claude.** This skill is part of the "וידאו עם AI" course skills by
Gabriel Gelman (Apache-2.0). It is an advanced bonus skill: it assumes Claude Code with
ffmpeg and Python, and works best with HyperFrames installed.

Language: always talk to the user in Hebrew unless they write in another language. Explain each
step in plain words before doing it (what we do, why, what the user does now), one step at a
time, and name the step ("שלב A2: בורד הסגנון"). All pages the user sees (board, plan, review
room) are in the course design: RTL Hebrew, Karantina / Heebo, white-black-violet.

Inspiration: the two-phase workflow (train a style from ~20 reference videos and a board, then
edit with a human-made plan) was inspired by Joseph's YouTube video on editing talking-head
videos with Claude (https://youtu.be/g9xoKm5DEeU). Nothing from the video is copied here; the
board, plan and review pages, the scripts, the rules and the review loop are our own, rebuilt
and tested on a batch of 13 Reels.

## The idea in one paragraph
Reference videos teach Claude the *logic* behind the animations (when, how long, how they
move); the board teaches it the *design* (fonts, colours, sizes, why). Without both, edits look
generic. The human's taste sits at three checkpoints: **the board** (phase A), **the edit plan**
(phase B) and **the review rounds**. Claude does the heavy lifting in between. If you do not
know what makes a style good, AI only helps you make bad videos faster, so every rule is
measured and every design choice gets a "why".

## Defaults (say them in one line at the start, change on request)
- **Brand:** the course brand (`references/brand-defaults.md`: Karantina 700 headlines, Heebo
  body at -0.04em, Consolas for code, white / black / violet). The reference's layout, motion
  and structure are kept; its fonts and colours are swapped for the brand. The user can keep the
  reference's own look for a style, or put their own brand in `~/.claude/style-lab/brand.md`.
  A chosen client's palette and fonts replace the brand for that edit (Clients below).
- **Build engine:** HyperFrames (HTML to video). Without it: `references/tools-and-fallbacks.md`.
- **Transcription:** local faster-whisper; Hebrew with a local ivrit-ai model. Cloud (RunPod)
  only with the user's OK, because it costs money.
- **Images / 3D:** optional, only when a beat needs them, and only with the user's OK if paid.
- **Format:** 9:16, 1080x1920, 30 fps, unless the user says otherwise. Never assume the caption
  language: ask (or no captions). What is installed comes from Stage 0, never from a guess.

## Stage 0: connections check (every session, before anything else)
"שלב 0: בדיקת חיבורים", once at the start of every session that uses this skill.
1. `python scripts/doctor.py` checks local tools in parallel (ffmpeg, Python packages, torch,
   Node + HyperFrames, Blender, yt-dlp, gh; env keys such as RunPod for presence only) and writes
   `~/.claude/style-lab/capabilities.json`. `--deep` also probes torch CUDA (slow; for the cut).
2. Record what your tool list and skills show (Higgsfield, Pexels, Pixabay, Pinterest, Figma,
   Blender MCP, After Effects / Premiere, Metricool, Playwright, image models, video-analysis
   skills), one call per note: `doctor.py session --set higgsfield=ok --note "paid credits"`.
3. Show one short Hebrew table: per stage what will be used, what is missing and how to add it,
   what costs money. Later steps pick tools only from this map; a missing tool blocks its step
   (say what to install, never download without a yes). See `references/tools-and-fallbacks.md`.

## The global store: styles and clients
Everything the user makes lives in `~/.claude/style-lab/` (`STYLE_LAB_HOME` overrides it), so
updating the skill never wipes it. Layout and every field: `references/store-schema.md`.
**Styles.** Each has a **name the user chooses** (Hebrew is fine), a latin slug (made from the
name if not given), a description, a pack and a style skill `style-<slug>`, in `styles.json`:
```
python scripts/styles.py {list | show <slug or name> | check | migrate}   # one per call; migrate = v1 -> v2 + dated .bak
python scripts/styles.py add [<slug>] --name "<name>" --pack <folder> --description "..." [--kind trained|described|derived] [--aspect 9:16|16:9|1:1|4:5]
python scripts/styles.py set <slug> name="<new name>" status=ready aliases="..." theme=light version=2
```
Status `training` (phase A), `ready` (after A5; only these are offered), `archived`; `remove`
drops the entry only. Also `fonts`/`colors` (`brand`|`reference`), `theme`, `kind`, `aspect`,
`version` (bump on rule changes), `rules_count`. New packs: `~/.claude/style-lab/styles/<slug>/`.
**Clients** (a brand any style can wear): name, logo, fonts, brand colours, a picked palette
(פלטת צבעים), theme, dos and don'ts, linked styles (one default); the style gives layout and
motion, the client colours, fonts and logo. "לקוח חדש", a new brand, a logo, a colour or a
website → `references/stage-c-client.md`; palettes: `references/palette.md`.
```
python scripts/clients.py {list | show <client>}                         # one per call
python scripts/clients.py add --name "<name>" --color "#RRGGBB" [--industry ..] [--tone ..] [--theme dark|light|both]
python scripts/clients.py palette <client> [--pick monochrome|analogous|accent]   # 3 proposals, save one
python scripts/clients.py link <client> <style> --default
```
**Which phase?** "new style / ליצור סגנון / ללמד סגנון" → phase A. "edit / make a video /
תערוך" → phase B. "לקוח חדש / פלטת צבעים" → stage C. "change / rename / improve style X" →
`styles.py set`, and for rule changes edit that style's STYLE.md and skill.
**Where things live:**
```
<pack>/  refs/ sources.md  analysis/  board/ (board.json board.html thumbs/)  animations.json
         library/<V01-A06 title>/  STYLE.md (numbered rules, tokens, motion, sound)  kit/
<project>/edits/<clip>/  raw/ words.json edit_ready.mp4 edit_ready.words.json plan.json
         plan.html build/ assets/ renders/ revisions.md SOURCES.md
~/.claude/skills/style-<slug>/SKILL.md   the generated style skill
```
`refs/` videos are local study copies, never published. Scripts (local, free): `doctor.py`,
`styles.py`, `clients.py`, `palette.py`, `store.py`, `transcribe_words.py`, `cut_silences.py`,
`verify_cut.py`, `word_edges.py`, `build_reference_library.py`; `transcribe_cloud_ivrit.py` is
paid. Pages: `assets/{board,plan,review}-template.html`.

---

## Phase A: train a style (build the style pack)

### A1. Collect the references
- Ask what to **call the style** and for a one-line description; register it right away
  (`styles.py add ...`, status `training`). Ask brand look (default) or the reference's own.
- Ask where the references are. **Videos:** the user gives files or links they are allowed to
  download; aim for ~20 of the best-edited videos in exactly this style; fewer than ~12 gives a
  thin library (say so, continue if the user wants). Name them `V01.mp4 …`, list each in
  `refs/sources.md`. They stay local: never published, uploaded or put in an artifact.
- **Description only** (no videos): interview the user (feel, grounds, type, motion, captions,
  sound, what is banned), turn every adjective into a number, and collect a few stills or
  screenshots they like. Skip A3-A4's frame work; the board shows tokens, rules and sketches
  instead of reference frames, and the library starts empty and grows from approved edits.

### A2. The style board (the most important part of training)
1. Analyse every reference: frames every 0.5 s with ffmpeg (or a video-analysis tool if
   installed), contact sheets, audio loudness. Read the sheets.
2. Propose the categories (caption system, layout, B-roll types, graphics, transitions, CTA …);
   the style's own categories win. `references/example-style-3d-paper.md` shows a finished one.
3. Pick 100-150 inspiration frames as 9:16 thumbs (~360 px wide) into `board/thumbs/`, and one
   hero frame per category to annotate.
4. Fill `board.json` (schema: `references/board-and-plan-spec.md`): tokens (fonts, colours with
   hex picked from frames, textures, shapes, sizing, layout) and per category the tiles, one
   annotated frame with 3-5 **measurable** callouts, and draft answers to the four "why"
   questions (design, colours, effects, animations).
5. Build `board.html` from `assets/board-template.html` (replace the JSON block) and publish it
   as an artifact with the thumbs as supporting files. The user rewrites the "why" answers and
   presses **העתקת ההערות לבורד** to paste them back.
6. **Checkpoint 1: stop and wait.** Apply the user's notes, republish. Do not start A3 until the
   user says the board is right; a wrong board trains a wrong style.

### A3. The breakdown
For **every** reference: a word-level transcript (`transcribe_words.py`, language from the user
or `auto`) and a frame-by-frame pass that finds every animation and records three things:
- **placement**: which words trigger it, how long after the word, which kind of line gets
  which kind of scene (announce vs describe, number, person, story, punchline);
- **design**: ground, texture, type pair and sizes, colour, line work, illustration, position;
- **logic**: duration in frames, distances, easing, camera moves. Measure on frames; never guess.
Then reconcile with the board (board = what and why; videos = when and how it moves); ask the
user where they disagree. Write each animation to `animations.json` (`V<nn>-A<n>`, start/end,
keyframe, title, category, scene type, words, notes). One video at a time, saving as you go;
with many videos, split the work across subagents (one video each, same schema).

### A4. The reference library
`python scripts/build_reference_library.py styles/<slug>/animations.json --out styles/<slug>/library`
Every animation becomes a folder (still, labelled frame strip, clip, notes) plus `index.json`.
Open a few strips, fix wrong boundaries in `animations.json`, rerun with `--force`.

### A5. Package the style
Write `STYLE.md` and the style skill from `references/style-skill-template.md`: **numbered,
measured rules** (so a plan or a note can say "rule 7"); tokens mapped onto the brand (keep the
reference's sizes, ratios and colour discipline; record its originals); each category with
design, motion, placement and 2-3 library refs; the planning patterns; the sound profile; a
"way of thinking" section; an empty "Review-round rules" section. Start every style from the
general rules in `references/craft-rules.md` and write only what differs. Recommended: build
the recurring parts (captions, headline, speaker layout, CTA) as a tested component kit in
`kit/`. Save the skill to `~/.claude/skills/style-<slug>/SKILL.md`, then
`styles.py set <slug> status=ready board_url=<board artifact URL>`. Tell the user the style's
name and that a new session will list it.

---

## Phase B: edit a clip in a trained style
Use the strongest model and effort available.

### B0. Choose the client and the style
`clients.py list`, `styles.py list`. Ask who the video is for: a client (its default linked style
is the first suggestion; a new client: stage C first) or none (brand defaults). Then the style:
the one the user named, else ask with the ready styles as options (one: say which); none ready:
offer phase A. Load `style-<slug>`, read `clients.py show <client>` (palette, fonts, logo, dos /
don'ts) and write both into `plan.json` so every round uses the same client and style.

### B1. Transcript and cut
1. `transcribe_words.py raw.mp4 --out words.json` (Hebrew: `--language he --model <ivrit-ai folder>`).
2. Show the low-confidence words (names, numbers, brands); fix them before planning.
3. Choose takes with the user if there are several (`--keep keep.json`), then
   `cut_silences.py raw.mp4 --words words.json --out edit_ready.mp4 --pad-ms 120`
   (30 ms only for a deliberately punchy style). Joins get a 15 ms audio fade.
4. Re-listen to every join without VAD (`transcribe_words.py edit_ready.mp4 --start .. --end ..
   --no-vad --no-context`), check word endings (`word_edges.py --at <t>`), then
   `verify_cut.py`. Hidden repeats and clipped endings are found here, not by the owner.
Everything after this uses `edit_ready.mp4` and `edit_ready.words.json`.

### B2. The edit plan (show it unasked)
Split the script into **beats** (one idea or line each). For each beat, three designs from the
style's library, placed the way the references place them, one marked **ההמלצה שלי**. Each
option: title, category, scene type (פנים + כתוביות / שכבה מעל הפנים / סצנה מלאה), a concrete
description, `Ref V##-A##`; each beat: why it fits (cite references and rule numbers) and a
camera note. Design each beat with the five questions in `references/craft-rules.md` §1
(metaphor, the real thing working, invisible made visible, a reference for the move, never
repeated). Build `plan.html` from `assets/plan-template.html`, publish with a 540p proxy as
`clip.mp4` and the option stills as `thumbs/`.

### B3. Checkpoint 2: the human plans
The user picks a design per beat or writes their own, presses **העתקת התוכנית** and pastes it
back. Offer the planning patterns as advice: open on the speaker; the hook is one strong beat;
announce on the face, describe with a visual; a person → a callout; a story → B-roll; balance
the scene types. If the user gives full control ("אתה מחליט"), take your picks and say so.

### B4. Build
1. For every chosen design open its library folder and write down the exact design to copy.
2. **Asset list** with a source per item (real UIs, official logos, the tool maker's own
   showcase clips, the user's files, licensed music/SFX, optional generated images or 3D); show
   it; ask before downloading. Log every source in `SOURCES.md`.
3. Build in HyperFrames (load its skill first) or the fallback engine: one 1080x1920
   composition with `edit_ready.mp4` as the speaker track; style tokens; the style's kit first.
   Order: captions for the whole video from the words file, then each beat in plan order.
   **Build 2 beats first**, snapshot, compare with the style's reference frames, adjust, then
   the rest. Follow `references/craft-rules.md` and check `references/pitfalls.md`.
4. Validate (0 errors), snapshot every 0.5 s (0.2 s over dense beats and the CTA), read every
   snapshot, fix. Then render a **draft** (low quality, with music) and open the review page.
   Say plainly that a first draft is 60-70 % of the way.

### B5. Review rounds
Follow `references/review-loop.md`: the review room page (`assets/review-template.html`) opens
unasked after every draft; read every note including unsent drafts; batch all notes → fix → one
draft render per round; **the same note in two rounds becomes a global rule** in the style's
STYLE.md and is fixed in every video. Log rounds in `revisions.md`. Ask after each round: "יש
עוד הערות, או שאפשר לעבור לסאונד ולרינדור הסופי?"

### B6. Sound and final
SFX only on visible events, a file at most twice per video and never twice within 4 s, -15 to
-20 dB under the voice; music instrumental, ducked under the voice, final about -14 LUFS. Only
sound the user is licensed to use. **Final render only on the owner's explicit OK**, then a last
2.5 s contact-sheet check for empty frames and leftovers, and delivery with `SOURCES.md`.

## Guardrails
- Reference videos are study material: local only, never redistributed, never in a published
  artifact (stills on a private board are fine). Copy a style's logic, not anyone's assets.
- Pages start private; the user decides whether to share them.
- Never invent numbers, quotes, results or logos; mock screens are marked "להמחשה".
- No income or growth promises. Mark AI-generated people and voices as AI.
- Ask before anything paid (cloud transcription, image or 3D generation, stock).

## References
- `references/craft-rules.md`: the way of thinking and the rules that survived real review rounds (composition, motion, speaker, captions, cuts, sound, sources, build hygiene). Read before B2 and B4.
- `references/review-loop.md`: drafts, the review room page, reading notes, rounds, global rules, final.
- `references/pitfalls.md`: real bugs and their fixes (tween resets, in-composition cuts, short full-face moves, SFX fatigue …). Read before B4.
- `references/store-schema.md`: the store, `styles.json` v2, `client.json`, `capabilities.json`.
- `references/stage-c-client.md` (a client in chat) and `references/palette.md` (three palettes, roles, checks).
- `references/tools-and-fallbacks.md`: what each step needs, local vs cloud, building without HyperFrames, optional 3D and image generation.
- `references/board-and-plan-spec.md`: `board.json` and `plan.json` schemas and how to publish the pages.
- `references/style-skill-template.md`: the template for the generated style skill (A5).
- `references/brand-defaults.md`: the course brand tokens (dark and light) and how to map a reference palette onto them.
- `references/example-style-3d-paper.md`: a real, approved style pack after three review rounds, as a worked example.
