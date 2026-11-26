# Tools: the defaults and what to do without them

## The connections check (Stage 0): `scripts/doctor.py`
Never guess what is installed. At the start of every session that uses the skill:
```
python scripts/doctor.py                 # local checks in parallel, prints the table per stage
python scripts/doctor.py --deep          # also probes torch CUDA (slow); use before the cut stage
python scripts/doctor.py --json          # the same map as JSON
python scripts/doctor.py session --set higgsfield=ok --note "MCP, paid credits"
python scripts/doctor.py session --set pexels=ok --set pixabay=ok --note "MCP, free stock"
python scripts/doctor.py session --set figma=missing
python scripts/doctor.py session --clear # forget the session block
```
The local part checks binaries (ffmpeg, ffprobe, node, blender, yt-dlp, gh; Blender also from
`BLENDER_PATH` and the usual install folders), HyperFrames (bounded, non-interactive,
`npx --no`: it never downloads; a package that is not cached yet reads "not installed", and the
first `npx hyperframes` downloads it, so ask the user first), Python
modules (numpy, cv2, torch, torchaudio, transformers, faster_whisper, PIL), the cut models (presence of the cache folder only, never loaded) and env keys (RunPod; presence
only, a value is never printed or stored). Each missing item carries an install hint.

The session part is yours: look at your own tool list and skills and record each MCP server or
skill a step could use (Higgsfield, Pexels, Pixabay, Pinterest, Figma, Blender MCP, After
Effects / Premiere, Metricool for real view data, Playwright or a browser, image models,
video-analysis skills). `--set` can repeat in one call, but one `--note` covers the whole call,
so group by note. The map is saved in `~/.claude/style-lab/capabilities.json` (schema in
`references/store-schema.md`); a later local run keeps the session block.

Stages in the map (`STAGES` in `doctor.py`) and their Hebrew names for the table:
`train` אימון, `train_profile` פרופיל, `cut` חיתוך (Hebrew), `cut_other_languages` חיתוך בשפות אחרות, `frames` פריימים, `build` בנייה, `3d`
תלת-ממד, `cloud_asr` תמלול בענן (paid), `publish` פרסום. Show the user one short Hebrew table:
per stage what will be used, what is missing (and the hint to add it), and what costs money
(cloud ASR, Higgsfield or other generation credits, paid image models, stock). Every later step
picks its tool only from this map. If a step's tool is missing, say which step is blocked and
what to install. Never install or download without the user's yes, and never skip a step
silently.

## Required
- **ffmpeg / ffprobe** on PATH: frames, cuts, proxies, audio. Nothing works without it.
- **Python 3.10+**: the scripts in `scripts/` (standard library, plus faster-whisper and Pillow
  where noted).

## Transcription
| Option | When | How |
|---|---|---|
| Local faster-whisper (default) | always first; free | `pip install faster-whisper`; `scripts/transcribe_words.py --model small` (English) or `large-v3` |
| Local Hebrew model (default for Hebrew) | Hebrew speech | an ivrit-ai CT2 model such as `ivrit-ai/whisper-large-v3-turbo-ct2`, downloaded once with the user's OK, passed with `--model <folder>` |
| Cloud (optional, paid) | long batches, no GPU, and only with the user's OK | `scripts/transcribe_cloud_ivrit.py`: the ivrit.ai RunPod endpoint; reads `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` from the environment only. Say it costs GPU time before running |
Never trust one pass for names, brands and numbers: show the low-confidence words to the user.

## Precise cut stack (stage B1)
Used by `align.py`, `cut.py`, `tighten.py`, `check_cut.py` (see `stage-b-edit.md`). Models are
looked for in the Hugging Face cache (`HF_HOME`) and the torch hub cache (`TORCH_HOME`); a missing
one is never fetched without the user's yes (`--allow-download`, exit 3 otherwise).
| Item | Install / download | Size |
|---|---|---|
| torch, torchaudio | `pip install torch torchaudio` | about 200 MB CPU build, more with CUDA |
| transformers | `pip install transformers` | about 10 MB |
| faster-whisper | `pip install faster-whisper` | small |
| Hebrew aligner `imvladikon/wav2vec2-large-xlsr-53-hebrew` | `align.py ... --allow-download` | about 1.2 GB |
| Hebrew check ASR `ivrit-ai/whisper-large-v3-turbo-ct2` | `check_cut.py ... --allow-download` | about 1.6 GB |
| Other-language aligner (torchaudio MMS_FA) | `align.py --lang en ... --allow-download` | about 1.2 GB |
| Other-language check ASR `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `check_cut.py ... --allow-download` | about 1.6 GB |
Without this stack the only option is the fallback `cut_silences.py` + `verify_cut.py`, which warns
that the precise pipeline was skipped; use it only after the user accepts the loss of precision.

## Build engine
**Default: HyperFrames** (HTML/CSS/JS rendered to video, by HeyGen, Apache-2.0,
https://github.com/heygen-com/hyperframes). Load its skill first if installed; validate with
its `check`, look with `snapshot`, render drafts with `--quality draft`.

Without HyperFrames, keep phases A and B exactly the same (board, plan, rules, review loop);
only step B4 changes:
- **Remotion** (React to video): one composition per beat, the same timings in frames.
- **After Effects / Premiere** (by hand, or through an MCP/scripting bridge if the user has one):
  the plan becomes a shot list with in/out times; captions as a text track.
- **A plain editor (CapCut, DaVinci Resolve):** hand the user the plan as a numbered shot list
  (time, what is on screen, motion with durations and easing, SFX) and the assets folder; review
  the user's render with the same loop.
- **ffmpeg only:** for cut + captions + simple overlays.
The craft rules are engine-agnostic: entrances from off-frame, durations, easing names,
z-order, caption timing.

## Motion helpers
Any tween library works (GSAP is common in HTML). Whatever you use: one curve per move,
duration from distance and weight, lead + uneven reactions, and tweens that touch only the
properties they animate (pitfalls §1).

## 3D (optional)
Blender in background mode (`blender -b -P script.py -- ...`) renders hero objects as
transparent PNG sequences: small renders (512-768 px, 12-24 frames, low samples + denoiser),
real materials, studio HDRI, contact shadow. If several jobs share one machine, run one at a
time (a lock folder created with `mkdir` works on every OS). Use 3D only when the object is
the point (craft-rules §2).

## Generated images (optional)
Any image model the user already has (and pays for) can make an illustration, texture or
sticker that HTML or 3D cannot make well:
- write the prompt to a file in a new empty folder; describe subject, the style's look and
  colours, background (plain `#00ff00` if it will be cut out), size;
- no readable text inside the image; text stays live in the composition;
- cut-outs: key the green, e.g. `ffmpeg -i in.png -vf colorkey=0x00ff00:0.25:0.08 out.png`;
- a short loop: 4-8 frames generated with the previous frame as reference, played as a PNG
  sequence (prefer it over GIF inside a composition);
- log each image in `SOURCES.md` with the model and the prompt.
Ask before any paid generation and say the cost if it is known.

## Speaker cut-out (optional)
A person matte (any background-removal model) gives a sharp cut-out over a clean plate (the
empty room, the median of several frames). If the matte is bad, keep the plain inset; a bad
edge is worse than no breakout.

## Music and SFX
Only what the user is licensed to use: their own library, royalty-free packs, or SFX they
synthesise. Never claim a track is cleared if you do not know.
