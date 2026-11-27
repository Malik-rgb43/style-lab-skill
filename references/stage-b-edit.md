# Stage B1 in detail: the precise cut pipeline

Every cut of a speaker recording (Reels, lessons, VSLs, talking heads) goes through this
pipeline. Never cut on raw Whisper word times with a fixed pad: Whisper word edges are 100-300 ms
off in Hebrew, which leaves long pauses and clips syllables (see `pitfalls.md`). `cut_silences.py`
and `verify_cut.py` remain only as a fallback the user must accept; they print a warning line.

Run `python scripts/doctor.py` first: stage `cut` (Hebrew) or `cut_other_languages` shows which
packages and models are missing. A missing model is never downloaded without the user's yes;
the script exits 3 naming the model, its size and `--allow-download`.

## Steps
1. **Transcribe**: `transcribe_words.py raw.mp4 --out words.json` (Hebrew: ivrit-ai model; cloud
   only if the user asks, `transcribe_cloud_ivrit.py`, paid). Show low-confidence words (names,
   numbers, brands) and fix them. A Latin word in a Hebrew transcript (a brand, a tech word)
   can only be aligned when the per-user lexicon says how it is spoken, written in the
   language's letters: `python scripts/cutlang.py lexicon add --lang he --word <latin word>
   --spelling <how it is spoken, in Hebrew letters>`; `lexicon list --lang he` shows it. Only
   single Latin words are looked up, so a Hebrew or a multi-word key has no effect.
2. **Align**: `python scripts/align.py raw.mp4 words.json --out words.aligned.json
   [--lang he|en|auto] [--chunk-s 30] [--margin-s 0.6] [--from-word N] [--to-word N]
   [--threads N] [--allow-download]`. Forced alignment (Hebrew wav2vec2 CTC, other languages MMS
   on romanised text) plus an energy refine. Exit 3 = a package or model is missing. All keep
   decisions use the aligned times; words with a `flag` (low score, unaligned) are never cut inside.
3. **Choose keep ranges** by reading the aligned words AND listening: re-transcribe short windows
   of the raw file without VAD (`transcribe_words.py raw.mp4 --start .. --end .. --no-vad
   --no-context`) to find hidden takes the transcript merged or skipped. Drop sentence openings
   the speaker repeats in full (false starts) and prefer the last complete take. Scan every
   chapter or section for them, not only the ones you noticed.
4. **Write `keep.json`**: `{"keep": [[i0, i1], ...]}` (inclusive word indices of `words.json`).
   Optional keys:
   - `retime`: `{"<word index>": [start_ms, end_ms]}` pins a word the aligner put on the wrong take.
   - `exclude_ms`: `[[a, b], ...]` source ranges that must never be in the cut (a hidden fragment).
   - `placeholders`: `[{"id", "after_word", "dur_s", "note"[, "tone_ms": [a, b]]}]` black video with
     looped room tone after a word that ends a piece (for an insert that comes later).
   - `fade_out_ms`: `[[src_out_ms, ms], ...]` longer fade for an out-point on a soft voiced tail.
   - `fade_in_ms`: `[[src_in_ms, ms], ...]` the same for an in-point.
   - `set_in_ms` / `set_out_ms`: `[[old_src_ms, new_src_ms], ...]` move one piece's in / out point.
5. **Cut**: `python scripts/cut.py raw.mp4 words.aligned.json keep.json --out <dir>/<stem>
   --preset reels|lesson|vsl`. Cuts land only in the quiet between words, on frame boundaries,
   with 15 ms crossfades. Presets: `reels` (pauses shortened, 1 s end air), `lesson` (pauses
   about 0.3 s, no end air), `vsl` (the reels cut with a 200 ms tail, for talking heads; follow
   it with step 6). Frame rate: the source's own rate when it is a whole number that divides
   48000 (24, 25, 30, 50, 60); otherwise (29.97, a variable rate) the preset's rate (lesson 25,
   reels and vsl 30) with a one-line note. Explicit flags override a preset, and `--fps` always
   wins: `--fps --lead-ms --tail-ms --inside-ms --merge-gap-ms
   --pause-max-ms --pause-keep-ms --sentence-keep-ms --pause-min-cut-ms --pause-snap-ms
   --end-air-ms --sliver-ms --xfade-ms --crf --excl-overlap --no-render`.
   `--sliver-ms` (240) and `--pause-min-cut-ms` (120) are converted to frames at the cut's fps;
   the old `--sliver-frames` and `--pause-min-cut-frames` still work but are deprecated.
   Outputs: `<stem>.mp4`, `<stem>.cuts.json`, `<stem>.words.json` (remapped to the cut timeline),
   and `placeholders.json` when keep.json has placeholders, moved into place together after the
   render. `--no-render` writes only the plan and renames an older `<stem>.mp4` to
   `<stem>.stale.mp4`. A malformed or reversed keep entry exits 2 naming the entry.
   `--end-air-ms 0` switches the end air
   off, also under `--preset reels`. In `cuts.json` read `soft_joins` (a join with no quiet frame
   in reach), `long_gaps_kept` (a gap kept because no quiet frame existed), `dropped_slivers`
   (pieces too short to keep; each lists `lost_words`, and the CLI prints a warning for them: a
   word you wanted is gone, widen the keep range) and `channel_balance_db` (left minus right of
   the source; the output is balanced). For `vsl` the CLI ends with `next: tighten.py <stem>
   --out <stem>.tight, then check_cut.py <stem>.tight`.
6. **VSL / talking head only**: `python scripts/tighten.py <stem> --out <out_stem>
   [--min-pause-ms 250] [--no-placeholders] [--gap-ms 0] [--soft-db 4] [--loud-db 10]
   [--check-db 0.05]` joins without dead air. Flags: `--min-pause-ms` closes every pause at least
   this long; `--no-placeholders` drops placeholders and closes their joins; `--gap-ms` room tone
   left between a voice end and the next start (at least 10 ms is always left, so the crossfade
   sits in room tone); `--soft-db` sound and edges = room tone + this
   (3 slices in a row); `--loud-db` any single slice this far above the room tone is sound (a
   click, a short consonant); `--check-db` edge check tolerance. Outputs
   `<out_stem>.mp4/.cuts.json/.words.json`, moved into place together; cuts.json carries
   `edge_check`, `missing_check`, `added_check`, `edge_bad`, the informational
   `voice_not_died_away` and `sound_in_pause` (joins kept as the cut made them), and per join
   `type` (stack, keep or bridge), `src_out`/`src_in` and `audio_xfade_ms`. Exit 1 = an edge off
   by more than `--check-db` (default 0.05 dB) or sound missing from the output (any `edge_bad`
   item; the files are still written, fix the cut, do not ship it); exit 2 = a malformed
   cuts.json or an mp4 whose frame count does not match it (a stale render: cut again); exit 3 =
   ffmpeg missing. The audit also has an "added" check: exit 1 if any output audio was read from
   outside the kept pieces. `sound_in_pause` lists joins held because soft sound was found in the
   pause. Read `room_noise_warning` (printed as a warning too): when it is set, raise `--soft-db`
   or check the held joins in `sound_in_pause` and `voice_not_died_away`.
7. **Check**: `python scripts/check_cut.py <final stem> [--lang] [--model] [--heard] [--explain]
   [--max-gap-ms 450] [--max-word-ms 1200] [--join-db 12] [--max-balance-db 1.0]
   [--allow-download]`, where the final stem is the tighten output for vsl and the cut stem
   otherwise. It re-transcribes the whole cut without VAD and diffs it against the expected words;
   writes `<stem>.check.words.json` (unless `--heard`) and `<stem>.check.txt`. On a tighten
   output it judges every join (stack, keep, bridge) on its crossfade window, and tighten's
   `edge_bad` items fail the check too. Exit 0 = clean,
   1 = items left, 2 = the mp4 does not match its cuts.json, 3 = missing package, model or
   ffmpeg. `--explain` takes `{"items": [{"match": "text in
   the item", "why": "reason", "t": seconds (optional)}]}`; explained items are listed but no
   longer fail. Fix and re-run until clean, or until every remaining item is explained to the
   user (a mis-heard word that is really there, a deliberate pause).
8. Only then planning (B2), on `<final stem>.mp4` and `<final stem>.words.json`.

## Fallback
If the stack is missing and the user will not install it, say so in Hebrew, name the cost (long
pauses, clipped syllables), and use `cut_silences.py` + `verify_cut.py` only after an explicit yes.
Both print: "fallback: the precise cut pipeline (align.py, cut.py, check_cut.py) was skipped".

## B2-B7: plan, cost gate, frames, motion, picks, build, ideas

The project folder is `<project>/edits/<clip>/` (`project.py init` registers it). Every command below is
`python scripts/<name>.py`; check flags with `--help`. Picks come from the owner; "אתה מחליט" picks are
yours and are never logged to taste.

### B2. The plan (3 concepts per beat)
1. Read the rules that apply: `lessons.py rules --style <s> --client <c>`.
2. Split the script into beats in `plan.json` (`n`, `start`, `end`, `script`, `beat_types` such as
   `list`, `story`, `person`; optionally `message_type` for the creator's way). Times are on the
   edit-ready clip; the beats cover it end to end.
3. `plan.py candidates <project>`: per beat the laws that fire (word triggers: number, date,
   question), banked ideas that fit, the creator's ideas (`creator.py ideate`), and the blocked
   concepts (already shown, rejected by the client, used for the client, picked for another beat).
4. Write 3 options per beat into `options`: one from a law when a rule-candidate law fires
   (`source: law`, `law: "<when>-><then>"`), one from the bank when an idea fits (`source: idea`,
   `idea_id`), and one new invention as the wild card (`source: invention`, `wild: true`). Each
   option answers the five design questions (`design_q`: point, real_thing, invisible, reference,
   new; craft-rules §1) and has `concept_key`, `fit` (0..1, your judgement), `tags`, and `recipe`
   when it needs AI generation.
5. `plan.py finalize <project>` (it runs `check`; fix every line it prints), then the page: the
   Studio when it runs, else `pickpage.py export <project> --step plan --out <html>` and publish it
   with the `files` map it prints. Say in Hebrew when the client's taste decided a recommendation
   ("הטעם של הלקוח הכריע").
6. "עוד רעיונות" for a beat: `plan.py more <project> --beat N`, at most 3 extra sets; after that the
   script refuses and you ask the owner (`--owner-ok-more --owner-words "<his words>"`).

### The cost gate (right after the plan picks, before any frame)
1. For every AI beat: `genprep.py add <project> --beat N --stage frames --recipe <kind> --variant a`
   (and b, c: one item per planned frame variant) and one `--stage build` item. Slots the template
   needs come from `--slot name=value`; words, palette, look and aspect fill themselves.
2. Prices: call the tool's own cost-estimate tool when the capability map has one, else ask the
   owner. Record each with `spend.py price set --tool --model --unit --cost --currency --source`.
3. `spend.py estimate <project>`: show the one Hebrew table for the whole video (base + retry
   allowance, per currency). Wait for the owner's yes, then
   `spend.py approve <project> --batch S01 --owner-words "<his exact message>" --total <the total he saw>`.
4. Loop: `spend.py next <project>` releases one item → you call the MCP/CLI tool with exactly that
   prompt, negative, model, seed and params → save the file under `gen/files/` →
   `spend.py record <project> --item G001 --file gen/files/<f> --cost <c> --model <m> --seed <s>`.
   Never call a paid tool for an item `next` did not release. If the `paid-spend-gate` skill (or an
   equivalent) is in the session, follow it too. Cloud ASR is a paid call as well: `spend.py add`,
   unless `settings.json` has `cloud_asr_preapproved: true` (set only on the owner's own machine,
   on his request).
5. Look check of every generated image: `lookcheck.py <project> --item G001` (metrics against the
   reference frames and the recipe), then LOOK at the image yourself and give your verdict:
   `--claude pass|fail --why "<what you saw>"`. On a fail: `genprep.py refine <project> --item G001
   --add "..." --drop "..." --why "..."`, then `next` again. Past the retry allowance: a new
   estimate and a new yes.

### B3. Frames: 3 real stills per beat
1. `frames.py new-set <project> --beat N` makes `frames/B0N/s1/{a,b,c}/` from the scaffold with the
   client's tokens (`tokens.css`), the beat's words (`words.json`) and a `variant.json` to fill.
2. Build each variant's `index.html`: the style kit first, or a banked idea
   (`ideas.py render <id> --client <c> --out <file>`); without a kit, from the library refs. Fill
   `variant.json` (`title`, `composition`, `treatment`, one `wild: true`). AI beats use only
   `gen/files/` images whose item passed the look check. Details: `references/frames-and-motion.md`.
3. `frames.py check <project> --beat N --set 1` until clean, then `frames.py snap ...`, then the page
   (`pickpage.py export --step frames`). "עוד רעיונות" = `frames.py new-set --beat N --more`.

### B4. Motion
1. Ask once, word for word: "3 גרסאות אנימציה לכל פריים, או להמשיך עם אחת?". Then
   `motion.py mode <project> three|one` (and `--beat N` for each beat the owner names otherwise).
2. Per beat: `motion.py variants <project> --beat N` (copies the picked frame, writes
   `motion-params.js`: entrance, ease, timing, camera from the style's measured motion), make the
   beat's HTML read `motion-params.js` (`window.SL_MOTION`), then `motion.py check`, `motion.py
   render` (2-4 s drafts) and the page (`--step motion`). In mode one there is no render and the
   pick is yours.

### Picks
The Studio (`studio.py`, `references/studio.md`) is the main way to pick when it runs; start it in the
background at the first pick step and give the owner its link. "בחרתי" in the Studio or on a page → `picks.py sync <project>` (logs the owner's picks to taste once,
counts idea picks), then act on its "עוד רעיונות" and own-design lists. A pasted copy text →
`pickpage.py import <project> --text-file <txt>`. A pick in chat → `picks.py pick <project> --step
plan|frames|motion --beat N --id <id>`. `picks.py validate` refuses a frames pick before the plan
pick, a motion pick before the frames pick, and one concept on two beats.

### B5. Build gates and one draft
1. Captions for the whole video first, then `buildcheck.py captions-done <project>`.
2. Per beat in plan order: `buildcheck.py start-beat <project> --beat N`, build it,
   `buildcheck.py beat-done <project> --beat N [--snippet build/bN.html --params params.json]` (a
   snippet with parameters makes the beat eligible for the ideas bank at B7).
3. The first 2 non-a-roll beats: snapshot each at its frame's `at_s` and run
   `buildcheck.py compare <project> --beat N --snapshot <png>` until both pass (SSIM ≥ 0.60, palette
   ΔE ≤ 10). Later beats are refused until then (`--override "<reason>"` only on the owner's say).
4. AI beats: `genprep.py animate <project> --from G00x` (only a still that passed), the cost gate,
   the look check on the clip; apply the recipe's camera curve (`camera_curve.custom_ease`) to the
   camera layer in HyperFrames.
5. `buildcheck.py checklist <project>`, tick each item after checking it (`tick --phase pre --n K`),
   then `buildcheck.py draft <project> --file <mp4>` (one per round), then
   `prove.py qa <pack> --build <draft>`, the post ticks, then `buildcheck.py present <project>`.

### B6. Review rounds
As in `references/review-loop.md` and the learning loop; `buildcheck.py next-round <project>` at the
start of every round.

### B7. Final and the ideas bank
The final render only on the owner's explicit OK. Then `ideas.py from-final <project>` shows the
pre-ticked beats; the owner unticks what he does not want banked;
`ideas.py commit-final <project> --final <mp4> [--untick 2,5]` banks the rest and closes the project.
