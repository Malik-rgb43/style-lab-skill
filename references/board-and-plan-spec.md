# The two artifacts: style board and edit plan

Both are single HTML pages built from `assets/*-template.html` by replacing the JSON block
(`<script type="application/json" id="board-data">` / `id="plan-data"`) and publishing with
the Artifact tool. Load the `artifact-design` skill before the first publish in a session.
Images and the clip are published as supporting files with `files` (paths relative to the
page). Keep the page's `<title>` as the style or clip name ("Clean Educational Style",
"Four Habits Edit Plan"). Both start private.

## Style board (`board.json`)
```json
{
  "example": false,
  "style": "Clean Educational Style",
  "subtitle": "20 reference videos · 131 scenes · trained 2026-11-02",
  "tokens": {
    "fonts":   [{"name": "Inter", "role": "main text, light + bold", "sample": "really common", "css": "font-family:Inter;font-weight:800"}],
    "colors":  [{"hex": "#1F3FE0", "name": "Azure blue"}],
    "textures": ["…"], "shapes": ["…"], "sizing": ["…"], "layout": ["…"]
  },
  "categories": [
    {"id": "captions", "name": "On-Screen Text (Captions)",
     "rule": "one sentence: the category's design rule",
     "tiles": [{"id": "V01-A01", "label": "common", "thumb": "thumbs/V01-A01.jpg",
                "clip": "", "strip": "thumbs/V01-A01.strip.jpg", "notes": "notes.md text"}],
     "annotations": [{"thumb": "thumbs/V01-A01.jpg",
                      "callouts": [{"x": 50, "y": 63, "label": "Multiple sans light small text"}]}],
     "why": {"design": "", "colors": "", "effects": "", "animations": ""}}
  ]
}
```
- `tiles` are the inspiration pieces (the video had ~130). `thumb` is a 9:16 JPG ~360 px wide.
  `clip` stays empty by default (reference clips are not published); set it only if the user
  asks and the board stays private.
- `annotations[].callouts`: `x`,`y` are percent positions on the frame where the red dot sits;
  `label` is one measurable rule. 3–5 callouts per frame, at least one annotated frame per
  category. Measure on the frame (pixels, ratios), never eyeball a number you can measure.
- `why`: Claude's draft answers to the four questions; the user rewrites them on the board.
- Controls: page list (left) jumps to sections; a tile opens a detail view (still or clip,
  strip, notes); **Copy board notes** copies every "why" answer as Markdown for the user to
  paste back. Edits are kept in the viewer's browser only, so the paste is how they reach
  Claude: apply them to `board.json` and republish to the same URL.

Thumbs: `ffmpeg -ss <t> -i refs/V01.mp4 -frames:v 1 -vf scale=360:-2 -q:v 3 board/thumbs/V01-A01.jpg`
(the library's `still.png` scaled down works too). A publish takes at most 255 files; publish
more in several calls to the same `url`.

## Edit plan (`plan.json`)
```json
{
  "example": false,
  "title": "Four career patterns",
  "clip": {"file": "clip.mp4", "name": "G - Four career patterns - edit-ready cut 9x16 (1080x1920).mp4", "length_s": 57.4},
  "source": "Arthur Brooks × Chris Williamson",
  "style": "Arthur Brooks minimal",
  "beats": [
    {"n": 1, "start": 0.0, "end": 1.9,
     "script": "There's four career patterns, psychologically",
     "seen": "A framework announced in the first line, like V04 … cut away at 1.7 s.",
     "camera": "Camera: Medium shot 0–31.8 s.",
     "pick": "A",
     "options": [
       {"key": "A", "title": "Opening zoom-out + justified number",
        "category": "On-Screen Text (Captions)", "scene_type": "a-roll",
        "desc": "Justified block … (rule 13) …", "ref": "V14-A01", "thumb": "thumbs/V14-A01.jpg"}]}
  ]
}
```
- Times are seconds on the **edit-ready** clip (after the silence cut), from
  `edit_ready.words.json`. Beats cover the clip end to end without gaps.
- `scene_type`: `a-roll` (A-roll + captions), `overlay` (Overlay on A-roll), `cutaway` (Cut-away
  scene). The beat timeline colours each beat by its chosen option's type.
- `pick` is Claude's recommendation; it starts as the chosen option and is shown as **My pick**.
- `seen`: why this fits, citing reference videos (V04, V08) and style rules (rule 13, §10).
- `clip.file`: publish a proxy, not the master: `ffmpeg -i edit_ready.mp4 -vf scale=540:-2 -c:v libx264 -crf 28 -preset fast -c:a aac -b:a 96k -movflags +faststart clip.mp4` (must stay under 15 MB; lower the bitrate for long clips). Each beat's player loops that beat's range.
- Controls: "Use this" chooses an option; the text box takes the user's own plan for the beat;
  **Copy Plan** copies the whole plan as Markdown (`# <title> edit plan`, then per beat: time,
  script, chosen design with category, scene type and ref, the design text, and "My notes").
  The user pastes that back with the build prompt.

## Checks before publishing either page
- The JSON parses (`python -c "import json,re,sys; …"` or load it in the page once).
- Every `thumb` path exists in the `files` map.
- Hebrew text: elements already use `dir="auto"`; check one beat renders right-to-left.
