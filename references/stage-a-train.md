# Stage A: train a style, step by step

The commands below are the real CLIs (`python scripts/<name>.py`); check a flag with `--help` before
you trust memory. A **pack** is the style's folder (`styles/<slug>/`, the path is in `styles.py show
<slug>`). Every measurement of reference `<id>` goes to `<pack>/analysis/<id>/`. Reference videos are
private study material: read-only, local, never uploaded, published or put in an artifact.

Exit codes everywhere: 0 ok, 1 a check failed, 2 bad input, 3 a tool or measurement is missing. A
missing optional tool never stops the run; the output lists it under `not_run`, and so do you.

## A0. Before anything: what the style is learned from

Ask once, in one message, where the style comes from. Any mix is allowed:

| Source | What it gives | How |
|---|---|---|
| Reference videos (files or a folder) | timing, motion, look, sound, laws | A1-A4 below |
| A profile link | the videos, downloaded with a yes | `profile_fetch.py` (see its `--help`); then A1 |
| This conversation | decisions, notes, rejected options | the learning loop (`references/learning-loop.md`) |
| A project folder (HyperFrames) | exact type, colour, motion and sound tokens | `code_tokens.py` |
| A description only | adjectives turned into numbers | interview, then a `derived` style |

**From this conversation or a project folder:**
1. List every decision and note about the look in this conversation. Route each one through the
   learning loop (`lessons.py`, layers and evidence as in `references/learning-loop.md`); a one-off
   stays a candidate.
2. A HyperFrames project beats a video of it, because the code holds the exact values:
   `code_tokens.py <project> --out <pack>/tokens.json`.
3. Confirm with measured renders: when the project has a render, run A2 on it. Disagreements
   between code and render are listed, and the code wins.
4. Show **"מה לקחתי מהשיחה"**: a short list (fonts, sizes, colours, motion, sound, rules) and ask for a yes.
5. Fewer than ~5 examples: the style's kind is `derived` (`styles.py set <slug> kind=derived`, or
   `--kind derived` at `styles.py add`); say what is missing (the
   project folder, finals, more videos). Ask for it in one line.

**Per-layer sources.** A style can take a layer from somewhere other than its references, for
example fonts and text hierarchy from a brand or another project while the motion comes from the videos:

    code_tokens.py layer <pack> --set type=code:<project folder> --set colour=code:<project folder>

Sources: `refs` (default, measured), `code` (tokens.json of that project), `brand` or `owner` (no
measurements; the layer's values come from the brand or the owner's words). Set a source only
when the owner asks for it ("the fonts from X", "the hierarchy of project Y"). The fingerprint
then reads that layer from tokens.json and never from the references; `look` is always `refs`.

## A1. Inputs

- Register the style first: `styles.py add ...` (status `training`).
- Name the videos `V01.mp4 ...` in `<pack>/refs/`, one line each in `refs/sources.md` (where it came
  from, the owner's yes to download it).
- Aim for ~20 videos in exactly this style; under ~12 the laws are thin (say so, continue if asked).
- Ask which videos are the style (the family) and which are only contrast. Contrast videos are
  never trained on; they are good hold-out negatives.
- Run the connections check: `doctor.py` and read the `[train]` block. Required: ffmpeg, ffprobe, numpy.
  Recommended: cv2 (faces, text boxes). Optional: librosa, fontTools, PIL, OCR, faster-whisper,
  the YuNet face model. Say in one line what will run and what will not.
- The YuNet face model is a 232,589-byte download from the OpenCV model zoo: name it, ask, then
  `fetch_model.py yunet --yes`. Without it faces use the Haar cascade or mediapipe, or are not run.

## A2. Measure every video

Per video, in this order. The three heavy steps are scripts: run them in the background, a few
videos at a time, each with a log. You do not need subagents for this; only use them if the user
asks for them, and then only for the mechanics, on the cheaper model.

1. Transcript: `transcribe_words.py <video> --out <pack>/analysis/<id>/words.json`, with the
   language from the user (Hebrew: the ivrit-ai model, or the cloud route when allowed).
2. `measure_video.py <video> --out <pack>/analysis/<id>/measure.json --words <words.json>`:
   cuts, cadence, shots, palette, faces, text boxes, loudness, pace, and the frame sheets.
3. `track_motion.py <video> --measure <measure.json> --out <pack>/analysis/<id>/motion.json`:
   every element and camera move with its value and speed curve, ease fit, duration, peak.
4. `sound_events.py <video> --out <pack>/analysis/<id>/sound.json --words <words.json> --measure
   <measure.json> --motion <motion.json>`: SFX with class guesses, music (tempo, beats, ducking,
   sections), voice (loudness, pace, spectral tilt, crest, room rt60, de-ess) and the sync
   offsets between sound, cuts and moves. `--allow-download` only after a yes (stem separation).
5. `look.py <video> --measure <measure.json> --out <pack>/analysis/<id>/look.json`: grain,
   vignette, sharpness and depth of field, tone curve, split toning, halation, chromatic
   aberration, letterbox, flicker, and the text finish (stroke, shadow, glow, fill).

Then **you watch**. Open the sheets of each video and write `<pack>/analysis/<id>/events.manual.json`:
`[{t, type, label, attrs}]` with what only a viewer sees: props, the kind of B-roll, metaphors,
a reveal, a joke, a real person or outlet on screen. One event per moment, with its time. Fix a
wrong automatic count by saying so in `attrs.note`; never edit the measured files.

Then write the creator's mind for each video, `<pack>/analysis/<id>/creator.json`, following
`references/creator-mind.md` (the visual ideas first: the line, the message, how it became a
picture and why). `creator.py check <pack>` validates every file.

## A3. Hold-out first, then the laws

1. Pick the hold-out before you look for patterns, so the proof is honest:
   `prove.py holdout-pick <pack> --n 3` (or `--ids V07,V12` to choose them; never a contrast video
   for the placement score). These ids are never trained on.
2. `fingerprint.py <pack> --exclude <hold-out ids>`: per layer median and spread, traits (a
   rule candidate when seen in at least 30 % of the training videos), near-duplicates, and the
   layer sources. `fingerprint.py <pack> --thumbs` first if duplicates are likely.
3. `laws.py <pack> --exclude <hold-out ids>`: "when X then Y within the window" regularities. A law
   is a rule candidate only with support in 3 or more videos, macro confidence 0.6 or more and
   lift 1.5 or more; the rest are options.
4. `creator.py synth <pack> --exclude <hold-out ids>`: the playbook (what the creator does with
   each kind of line, the principles, what they never do). `creator.py merge-rules <pack>` folds
   near-identical free-text entries.
5. Target language differs from the references (English references, a Hebrew video):
   `lang.py adapt <pack> --to he --from-lang en` writes `adapt-he.json`: timing scaled by speech
   rate (word-anchored laws move with the words, visual ones keep their time), the RTL mirror
   plan, and the font mapping (`--font-dir` for the brand fonts).

## A4. Sound, assets and research

- Sound libraries come first: `assets_log.py sound-lib list`; add the owner's folders with
  `assets_log.py sound-lib add NAME PATH --kind sfx`.
- Log every asset the style needs, with its licence: `assets_log.py add <pack> --kind font|sfx|music|broll|ui|model
  --name N --source S --licence L --why W`. This writes `ASSETS.md`. No licence, no use.
- Fonts in the references: `font_match.py <crop.png> --text "<the words>" --fonts-dir <dir>` gives
  `{bidi, skipped_no_glyphs, results}`. A font without the language's glyphs is skipped, never chosen.
- `assets_log.py requires <pack>` proposes the style's connections; record the real ones with
  `styles.py require <slug> --name N --why W --level required|recommended [--fallback F]`.

## A5. Reproduction recipes (how each element is made again)

One recipe per element kind the style uses (a B-roll kind, a text card, a 3D object, an AI image,
an animated still...):

    assets_log.py recipe add <pack> --kind <kind> --how <ai_image+i2v|ai_video|stock|code|3d|screen>
        --look '<json from look.json>' --motion '<json from motion.json>' --prompt-template "<...>"
        --tool <capability>:<required|recommended>:<fallback> --evidence V03@12.4

- `assets_log.py recipe check <pack>` compares the tools with the capability map. Tell the owner, in
  Hebrew, what to connect: **"כדי לשחזר את <סוג> צריך לחבר <כלי>; בלי זה: <חלופה>."**
- A recipe whose `how` uses AI needs one test generation before the style can be `ready`. The test
  costs money: go through the paid-spend gate and the owner's yes first. Compare the result with the
  references (look numbers and a side-by-side still), change the prompt template, test again until
  it matches, then `--test passed` (states: untested, paid_ok_pending, passed, failed). The owner can waive this explicitly (`styles.py set-status ...
  --waive-recipes --reason "..."`).

## A6. Create (the rules are not the style)

- Beside each law and rule in STYLE.md, write its **principle**: why it works (metaphor, reveal,
  tension, contrast, rhythm). A rule without a reason gets copied badly.
- New plans answer the five design questions of `references/craft-rules.md` section 1, never
  repeat a beat already used for the same client (`ideas.py`), and carry one wild card per option set.
- For a line, `creator.py ideate <pack> --line "<text>" --type <fact|number|claim|emotion|contrast|process|reveal|cta>`
  proposes visual ideas in the creator's way (`--used` keeps the ones already used out).
- The bar: it passes the style's QA (`prove.py qa`) AND feels new next to the references.

## A7. Prove, then ready

1. **Hold-out.** Without watching the hold-out videos, read only their transcripts and the laws,
   and write `<dir>/<id>.events.json` (`[{t, type}]`), and optionally `<id>.structure.json` and
   `<id>.retention.json`. Then `prove.py holdout <pack> --predicted <dir>`. The actual events
   are read from `analysis/<id>/events.json`, or measured from the analysis folder when that file
   is missing.
2. **Recreation.** Build 3 short moments from the rules (never by copying frames), render them,
   and score each one: `prove.py recreation <pack> --moment V05:12.0-15.5 --build <render.mp4>`.
   `--owner-ok` only after the owner said yes on the board.
3. **QA.** `prove.py qa <pack> --build <render.mp4>` flags moves outside the style's spread.
4. **Graph view.** `prove.py graph <ref motion.json> --compare <build motion.json> --out graph.html`:
   value and speed curves, reference over build. Put it on the board.
5. **Cover.** `cover.py candidates <pack> --n 3`; the owner picks one; `styles.py set-cover <slug> <image>`.
6. `styles.py set-status <slug> ready`. It refuses until proof.json passes and every AI recipe has
   a passed test (or a waiver). `--force --reason` exists for the owner only.

Then write STYLE.md and the style skill (`references/style-skill-template.md`).

## What to show the owner, in Hebrew, at each gate

Keep each one to a few lines; numbers, not adjectives.

| Gate | Show | Ask |
|---|---|---|
| A0 | "מה לקחתי מהשיחה": the list | "נכון? חסר משהו?" |
| A1 | how many videos, family vs contrast, what will not run | "אלה הסרטונים של הסגנון?" |
| A2 | per video: cuts/min, moves, SFX, music yes/no, one sheet | only if a count looks wrong |
| A3 | the top laws in plain words with their numbers ("אחרי מספר, תוך 0.4 ש', כרטיס גדול: 11 מתוך 14") | "יש פה משהו שהוא לא הסגנון?" |
| A5 | what to connect per recipe, and the price of one test each | "לאשר בדיקה?" |
| A7 | hold-out F1 and timing error, the 3 moments, the graph, 3 cover frames | "מאשר שהסגנון מוכן? איזה פריים לשער?" |
