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
   --pause-max-ms --pause-keep-ms --sentence-keep-ms --pause-min-cut-frames --pause-snap-ms
   --end-air-ms --sliver-frames --xfade-ms --crf --excl-overlap --no-render`.
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
   left between a voice end and the next start; `--soft-db` sound and edges = room tone + this
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
   output it judges each join on its crossfade window and skips the stack joins with a note
   (tighten's edge check covers them; its `edge_bad` still fails the check). Exit 0 = clean,
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
