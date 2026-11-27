# style-lab

A Claude Code skill that learns a video editing style from reference videos and then edits new
footage in that style, for any client, with a plan, real frames, motion variants and review rounds.
It measures the references (cuts, motion curves, sound, look) and learns from every note you give.

## Requirements
- **Required:** Claude Code, Python 3.10+, ffmpeg and ffprobe on PATH, numpy.
- **Recommended:** Node 18+ and HyperFrames (`npx hyperframes`) for frames, motion and the build;
  OpenCV (`pip install opencv-python`) for faces and on-screen text.
- **For profile links:** yt-dlp (`instaloader` optional, for Instagram without a login).
- **Optional:** faster-whisper, the precise-cut models, librosa, fontTools, Pillow, an OCR engine,
  Blender, image and video generators. Claude checks what you have at the start of every session
  (`python scripts/doctor.py`) and tells you what each missing piece would add.

## Install, update, roll back
```bash
git clone https://github.com/Malik-rgb43/style-lab-skill
cd style-lab-skill
python scripts/install.py              # installs or updates ~/.claude/skills/style-lab
python scripts/install.py --dry-run    # shows what would change, writes nothing
python scripts/install.py --seed my-seed.json   # optional: lexicon words, brand.md, sound libraries
python scripts/install.py --rollback <backup folder printed by the last install>
```
Every install first copies your current skill to a dated folder in `~/.claude/skills-archive/` and
prints the exact rollback command. A private section at the end of your live SKILL.md (heading: see
`install.py --help`) and your private files are kept. Your styles, clients and lessons live in the
store (below), so updating never touches them.

## Quick start
Type one of these in Claude Code:
1. **Train from a folder:** "תאמן סגנון חדש מהסרטונים בתיקייה <the folder with your reference videos>" (or in English:
   "train a new editing style from the videos in this folder").
2. **Train from a profile link:** "תלמד את הסגנון של הפרופיל הזה: https://www.tiktok.com/@example.creator".
   Claude lists and ranks the posts, you pick, you confirm private study, and only then it downloads.
3. **Edit a clip for a client:** "תערוך את הקליפ הזה בסגנון <style> ללקוח <client>".

## Stages
| Stage | What happens | Read |
|---|---|---|
| 0 | Connections check: what is installed, what is connected, what costs money | `references/tools-and-fallbacks.md` |
| A | Train a style: measure the references, laws, the creator's way of thinking, recipes, proof | `references/stage-a-train.md` |
| C | A client: brand profile, logo, fonts, a palette from one colour | `references/stage-c-client.md` |
| B | Edit: precise cut, plan, cost gate, frames, motion, build, review rounds, ideas bank | `references/stage-b-edit.md` |

## Connections
<!-- connections:start -->
| Stage | Needs | Recommended / optional | Paid |
|---|---|---|---|
| train | ffmpeg, ffprobe, numpy | cv2 (recommended), librosa (optional), fontTools (optional), PIL (optional), ocr (optional), faster_whisper (optional), model_yunet (optional) | no |
| train_profile | yt-dlp | instaloader (optional) | no |
| cut | ffmpeg, torch, torchaudio, transformers, faster_whisper, model_wav2vec2_he, model_whisper_he | - | no |
| cut_other_languages | ffmpeg, torch, torchaudio, faster_whisper, model_mms_fa, model_whisper_multi | - | no |
| frames | node, hyperframes | - | no |
| motion | node, hyperframes, ffmpeg, ffprobe | - | no |
| ideas | ffmpeg, ffprobe | - | no |
| build | node, hyperframes, ffmpeg | - | no |
| 3d | blender | - | no |
| cloud_asr | RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID | - | yes |
| publish | gh | - | no |

| Entry | How to add it |
|---|---|
| ffmpeg | winget install Gyan.FFmpeg / brew install ffmpeg |
| ffprobe | winget install Gyan.FFmpeg / brew install ffmpeg |
| numpy | pip install numpy |
| cv2 | pip install opencv-python |
| librosa | pip install librosa (optional: a second onset envelope for the tempo; numpy is used without it) |
| fontTools | pip install fonttools (optional: font matching and glyph coverage checks) |
| PIL | pip install pillow |
| ocr | pip install rapidocr-onnxruntime, or install tesseract (optional: reads on-screen text; boxes are found without it) |
| faster_whisper | pip install faster-whisper |
| model_yunet | YuNet face model, about 230 KB from the OpenCV model zoo: python scripts/fetch_model.py yunet after the user says yes (faces fall back to Haar without it) |
| yt-dlp | winget install yt-dlp.yt-dlp / brew install yt-dlp |
| instaloader | pip install instaloader (only for Instagram when yt-dlp needs a login) |
| torch | pip install torch |
| torchaudio | pip install torchaudio |
| transformers | pip install transformers (about 10 MB; the Hebrew aligner loads through it) |
| model_wav2vec2_he | Hebrew aligner, about 1.3 GB: run align.py with --allow-download after the user says yes |
| model_whisper_he | Hebrew check ASR, about 1.6 GB: run check_cut.py with --allow-download after the user says yes |
| model_mms_fa | aligner for other languages (torchaudio MMS_FA), about 1.2 GB: run align.py --lang en with --allow-download after the user says yes |
| model_whisper_multi | check ASR for other languages (large-v3-turbo), about 1.6 GB: run check_cut.py with --allow-download after the user says yes |
| node | winget install OpenJS.NodeJS.LTS / brew install node |
| hyperframes | npx hyperframes (no global install needed); needs Node 18+ |
| blender | winget install BlenderFoundation.Blender / brew install --cask blender |
| RUNPOD_API_KEY | set the RUNPOD_API_KEY environment variable (paid cloud ASR) |
| RUNPOD_ENDPOINT_ID | set the RUNPOD_ENDPOINT_ID environment variable (paid cloud ASR) |
| gh | winget install GitHub.cli / brew install gh |
<!-- connections:end -->

## Study copies and privacy
Reference videos are downloaded only after you pick them and confirm they are for private study.
They stay on your computer: never published, uploaded or put in an artifact. Nothing paid runs
without one estimate for the whole video and your yes.

## Your data
Styles, clients, lessons, taste, prices and the ideas bank live in `~/.claude/style-lab` (or the
folder in `STYLE_LAB_HOME`), never inside the skill folder. See `references/store-schema.md`.

## Tests
```bash
python -m pytest
```
The tests never use the network and never touch your real store.

## Licence and credit
Apache-2.0 (see `LICENSE`). Part of the "וידאו עם AI" course skills.
