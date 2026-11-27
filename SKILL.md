---
name: style-lab
description: Learns a video editing style from reference videos or a description, keeps styles and clients (a brand profile with logo, fonts and a palette built from one colour) in one global store, and edits footage in a chosen style for a chosen client with an edit plan and review rounds, and learns from every note and pick (lessons become craft, style or client rules on the owner's yes). Starts every session with a connections check. Use when the user wants to train, save, learn or apply a reusable editing style, or add a client and its palette: style-lab, editing style, client, palette, סגנון עריכה, תלמד את הסגנון, תאמן סגנון, לקוח, לקוח חדש, פלטת צבעים. For a one-off "make mine like this reference" use reference-style-matching instead.
---

# style-lab

## When to use (full trigger list)
Create, define, copy, learn or train an editing style or look from reference videos (links or
files) or a written description; a style board or guide; reference reels into a reusable style;
edit a talking-head or short-form clip "in the style of" a trained style and fix it in review
rounds; "make my videos look like this creator's", "save this look so every video matches", "the
edit looks generic / AI". Hebrew: סגנון עריכה, ליצור סגנון, ללמד את Claude סגנון, בורד סגנון,
לוח השראה, ריפרנסים, תערוך בסגנון, תוכנית עריכה, סבב הערות, סטייל לסרטונים.

**סקיל = הוראות קבועות ל־Claude.** This skill is part of the "וידאו עם AI" course skills by
Gabriel Gelman (Apache-2.0). It is an advanced bonus skill: it assumes Claude Code with
ffmpeg and Python, and works best with HyperFrames installed.

Language: always talk to the user in Hebrew unless they write in another language. Explain each
step in plain words before doing it (what we do, why, what the user does now), one step at a
time, and name the step ("שלב A2: מדידת הסרטונים"). All pages the user sees (board, plan, review
room) are in the course design: RTL Hebrew, Karantina / Heebo, white-black-violet.

Inspiration: the two-phase workflow (train from ~20 references and a board, then edit with a
human-made plan) came from Joseph's YouTube video on editing talking-head videos with Claude
(https://youtu.be/g9xoKm5DEeU). Nothing is copied from it; pages, scripts, rules and the review
loop are our own, tested on 13 Reels.

## The idea in one paragraph
Reference videos teach Claude the *logic* behind the animations (when, how long, how they
move); the board teaches it the *design* (fonts, colours, sizes, why). Without both, edits look
generic. The human's taste sits at three checkpoints: **the board** (phase A), **the edit plan**
(phase B) and **the review rounds**, and every note and pick is kept so the next video starts
better. If you do not know what makes a style good, AI only helps you make bad videos faster,
so every rule is measured and every design choice gets a "why".

## Defaults (say them in one line at the start, change on request)
- **Brand:** the course brand (`references/brand-defaults.md`: Karantina 700 headlines, Heebo
  body at -0.04em, Consolas for code, white / black / violet) replaces the reference's fonts and
  colours; its layout, motion and structure stay. The user can keep the reference's own look, put
  their brand in `~/.claude/style-lab/brand.md`, or pick a client, whose palette and fonts win.
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
2. Record what your tool list and skills show (MCPs, image models, video-analysis skills; the
   list to look for is in `references/tools-and-fallbacks.md`), one call per note:
   `doctor.py session --set higgsfield=ok --note "paid credits"`.
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
python scripts/styles.py {require <slug> --name N --why W --level required|recommended [--fallback F] | unrequire <slug> --name N}
python scripts/styles.py set <slug> name="<new name>" status=ready aliases="..." theme=light version=2
```
Status `training` (phase A), `ready` (after A7, the proof; only these are offered), `archived`; `remove`
drops the entry only. Other fields (`fonts`/`colors`, `theme`, `kind`, `aspect`, `version`,
`rules_count`): `references/store-schema.md`. New packs: `~/.claude/style-lab/styles/<slug>/`.
**Clients** (a brand any style can wear): name, logo, fonts, brand colours, a picked palette
(פלטת צבעים), theme, dos and don'ts, linked styles (one default); the style gives layout and
motion, the client colours, fonts and logo. "לקוח חדש", a new brand, a logo, a colour or a
website → `references/stage-c-client.md`; palettes: `references/palette.md`. Commands
(`clients.py list | show | add | set | palette | link ...`, every flag): `references/stage-c-client.md`.
**Which phase?** "new style / ליצור סגנון / ללמד סגנון" → A; "edit / make a video / תערוך" →
B; "לקוח חדש / פלטת צבעים" → C; "change / rename / improve style X" → `styles.py set`, hand rules
in its STYLE.md and skill (learned rules only through `lessons.py`, `references/learning-loop.md`).
**Where things live** (style pack and project folders): `references/store-schema.md`. `refs/`
videos are local study copies, never published. Scripts (local, free): every `scripts/*.py`
(`cut_silences.py`, `verify_cut.py` = fallback only); `transcribe_cloud_ivrit.py` is paid. Pages: `assets/{board,plan,review}-template.html`.

## Phase A: train a style (build the style pack)

The full procedure, with every command, is `references/stage-a-train.md`. In short:

1. **A0 sources.** Videos, a profile link, this conversation, a HyperFrames project (code beats
   video: `code_tokens.py`), or a description. Show "מה לקחתי מהשיחה" for a yes. Per-layer
   sources (`code_tokens.py layer`) when the owner wants fonts or hierarchy from a brand or project.
2. **A1 inputs.** Register (`styles.py add`, status `training`); `V01.mp4 ...` in `refs/`, ~20 videos,
   family vs contrast; `doctor.py` `[train]` block. References stay local: never published, uploaded
   or put in an artifact.
3. **The style board (checkpoint 1).** Frames every 0.5 s and contact sheets, categories, 100-150
   thumbs, `board.json` (`references/board-and-plan-spec.md`) with measurable callouts and draft
   "why" answers; `board.html` from `assets/board-template.html`, published as an artifact.
   **Stop and wait** for the owner's notes; a wrong board trains a wrong style.
4. **A2 measure every video** (background scripts, a few at a time): `transcribe_words.py`,
   `measure_video.py`, `track_motion.py`, `sound_events.py`, `look.py`; then watch and write
   `events.manual.json` and `creator.json` (`references/creator-mind.md`).
5. **A3 hold-out first** (`prove.py holdout-pick`), then `fingerprint.py`, `laws.py`, `creator.py synth`
   on the training set; `lang.py adapt` when the target language differs.
6. **A4-A5 assets and recipes.** `assets_log.py` (licences, sound libraries, `requires`), one
   reproduction recipe per element kind, `recipe check`; AI recipes get one test after the
   paid-spend gate and the owner's yes.
7. **A6 create.** A principle beside every rule; `creator.py ideate` for visual ideas.
8. **A7 prove.** Hold-out predicted from transcripts only, 3 recreation moments, `prove.py qa`, the
   graph view, cover candidates (`cover.py candidates`, owner picks, `styles.py set-cover`), then
   `styles.py set-status <slug> ready` (gated by proof.json and the recipe tests).
9. **Package.** `STYLE.md` and the style skill from `references/style-skill-template.md`: numbered,
   measured rules, each with its principle; tokens mapped onto the brand; the "Review-round rules"
   marker pair. Optional: a tested component kit in `kit/`. Save the skill to
   `~/.claude/skills/style-<slug>/SKILL.md`; fill `requires`. Tell the user the style's name and that
   a new session will list it.

The animation library (`build_reference_library.py styles/<slug>/animations.json --out styles/<slug>/library`)
still works for styles trained the old way; new styles use the analysis folders.

## Phase B: edit a clip in a trained style
Use the strongest model and effort available.

### B0. Choose the client and the style
`clients.py list`, `styles.py list`. Ask who the video is for: a client (its default linked style
is the first suggestion; a new client: stage C first) or none (brand defaults). Then the style:
the one the user named, else ask with the ready styles as options (one: say which); none ready:
offer phase A. Load `style-<slug>`, read `clients.py show <client>` (palette, fonts, logo, dos /
don'ts) and write both into `plan.json` so every round uses the same client and style.
Run `doctor.py --style <slug>` before planning; a missing `required` connection (exit 4) = stop and ask.

### B1. Transcript and cut (precise pipeline, details in `references/stage-b-edit.md`)
1. `transcribe_words.py raw.mp4 --out words.json` (Hebrew: `--language he --model <ivrit-ai folder>`);
   show low-confidence words and fix them.
2. `align.py raw.mp4 words.json --out words.aligned.json` (forced alignment; exit 3 = a model is
   missing: ask the user before `--allow-download`). Never cut on raw Whisper times.
3. Keep ranges: read the aligned words AND listen without VAD on short windows for hidden takes;
   drop sentence openings repeated in full; prefer the last complete take; write `keep.json`.
4. `cut.py raw.mp4 words.aligned.json keep.json --out <stem> --preset reels|lesson|vsl`
   (reels: Reels, lesson: lessons, vsl: VSLs/talking heads, then `tighten.py <stem> --out <o>`).
5. `check_cut.py <final stem>` (for vsl the tighten output) until clean or every item is
   explained; only then planning, on `<final stem>.mp4` and `<final stem>.words.json`. `cut_silences.py`/`verify_cut.py`: fallback, user must accept.

### B2. The edit plan (show it unasked)
First read the rules that apply: `python scripts/lessons.py rules --style <s> --client <c>`
(`references/learning-loop.md` §7). Split the script into **beats** (one idea or line each).
For each beat, three designs from the style's library, placed the way the references place
them, one marked **ההמלצה שלי**. Each option: title, category, scene type (פנים + כתוביות /
שכבה מעל הפנים / סצנה מלאה), a concrete description, `Ref V##-A##`; each beat: why it fits
(cite references, hand rule numbers, learned rules by lesson id such as L0007) and a camera
note. Design each beat with the five questions in `references/craft-rules.md` §1 (metaphor,
the real thing working, invisible made visible, a reference for the move, never repeated).
Recommendations read
`taste.py summary --client <c> --style <s>`; say when taste decided one (§8). Build `plan.html`
from `assets/plan-template.html`, publish with a 540p proxy as `clip.mp4`, stills as `thumbs/`.

### B3. Checkpoint 2: the human plans
The user picks a design per beat or writes their own, presses **העתקת התוכנית** and pastes it
back. Offer the planning patterns as advice: open on the speaker; the hook is one strong beat;
announce on the face, describe with a visual; a person → a callout; a story → B-roll; balance
the scene types. If the user gives full control ("אתה מחליט"), take your picks and say so.
Log each of the owner's own picks per beat (not yours): `taste.py log --step plan ... --tags` (§8).

### B4. Build
1. Re-read `lessons.py rules --style <s> --client <c>`. For every chosen design open its
   library folder and write down the exact design to copy.
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
Follow `references/review-loop.md` and `references/learning-loop.md`: the review room page
(`assets/review-template.html`) opens unasked after every draft; read every note, unsent drafts
too. Before any fix show one line per note: his words → what I understood (measurable) → where
→ layer (two readings when unsure); save each with `lessons.py add` or `match`. Batch all notes
→ fix → one draft render per round. A lesson seen in 2 rounds or 2 videos is proposed; only the
owner's yes (`lessons.py approve`) makes it a rule, then fix it in every video. Log rounds in
`revisions.md`. After each round ask: "יש עוד הערות, או שאפשר לעבור לסאונד ולרינדור הסופי?"

### B6. Sound and final
Sound per `references/craft-rules.md` (SFX on visible events, a file at most twice and never
twice within 4 s, -15 to -20 dB under the voice; instrumental music ducked, final about -14
LUFS), only sound the user is licensed to use. **Final render only on the owner's explicit
OK**, then a 2.5 s contact-sheet check for empty frames and leftovers, delivery with
`SOURCES.md`. End of project: open lessons for approval, `lessons.py feedback list` as concrete
skill edits (never edited without a yes) and new ideas (learning-loop §9).

## Guardrails
- Reference videos are study material: local only, never redistributed, never in a published
  artifact (stills on a private board are fine). Copy a style's logic, not anyone's assets.
- Pages start private; the user decides whether to share them. Ask before anything paid
  (cloud transcription, image or 3D generation, stock).
- Never invent numbers, quotes, results or logos; mock screens are marked "להמחשה". No income
  or growth promises. Mark AI-generated people and voices as AI.

## References
- `references/stage-a-train.md`: training step by step (A0 sources, measuring, hold-out, laws, assets, recipes, create, prove, what to show the owner). Read before Phase A.
- `references/creator-mind.md`: how to write `creator.json` (the creator's visual ideas, story, decisions, retention, toolchain). Read before A2's watching step.
- `references/craft-rules.md`: the way of thinking and the rules that survived real review rounds (composition, motion, speaker, captions, cuts, sound, sources, build hygiene). Read before B2 and B4.
- `references/review-loop.md`: drafts, the review room page, reading notes, rounds, final.
- `references/learning-loop.md`: notes into lessons, layers, rules on the owner's yes, conflicts, exceptions, taste, skill feedback. Read before B2 and B5.
- `references/pitfalls.md`: real bugs and their fixes (tween resets, in-composition cuts, short full-face moves, SFX fatigue …). Read before B4.
- `references/store-schema.md`: the store, pack and project folders, `styles.json` v2, `client.json`, `capabilities.json`, lessons, taste.
- `references/stage-c-client.md` (a client in chat) and `references/palette.md` (three palettes, roles, checks).
- `references/tools-and-fallbacks.md`: what each step needs, local vs cloud, building without HyperFrames, optional 3D and image generation.
- `references/board-and-plan-spec.md`: `board.json` and `plan.json` schemas and how to publish the pages.
- `references/style-skill-template.md`: the generated style skill (Package step); `references/brand-defaults.md`: the course brand tokens (dark and light) and how to map a reference palette onto them.
- `references/example-style-3d-paper.md`: a real, approved style pack after three review rounds, as a worked example.
