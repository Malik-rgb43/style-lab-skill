# Frames and motion: the contracts

Read before B3 and B4 (`references/stage-b-edit.md`). The scripts enforce everything here.

## The frame scaffold (`frames.py new-set`)
Each variant folder `frames/B03/s<k>/<a|b|c>/` holds:
- `index.html`, from `assets/frame-scaffold.html`: the style's aspect at 1080 wide, `data-duration` =
  the beat length, `tokens.css` and `gsap.min.js` linked, and `{{BEAT_TEXT}}` where the words go.
- `tokens.css`: CSS variables `--sl-<role>` (background, surface, text, muted, brand, brand-deep, tint,
  accent, highlight, caption-pill, cta), `--sl-label-<key>` and `--sl-font-heading|body|mono`, from the
  client's palette, or from `assets/brand-defaults.json` when there is no client or no palette.
- `words.json`: the beat's words, times from 0.
- `variant.json`: `{id, title, composition, treatment, wild, concept_key, at_s, ai_items, idea_id}`.
  `at_s` is the moment the still is taken (default: the middle of the beat).
- `gsap.min.js`: copied from `$STYLE_LAB_GSAP`, `<store>/vendor/gsap.min.js` or the npx cache. Without
  one, `new-set` warns; put a copy in `<store>/vendor/` once.

HyperFrames runs only through `hf.py`, which points it at an installed Chrome
(`HYPERFRAMES_BROWSER_PATH`) so it never downloads a browser, and never installs a package.

## Variant rules (`frames.py check`)
- Exactly 3 variants, exactly one wild card.
- Pairwise a different `composition` or `treatment`, AND different markup once colours are removed:
  three variants that differ only in colour are refused.
- The beat's real words in every variant; no `{{`, lorem, "sample text", "placeholder", "טקסט לדוגמה".
- Colours only from `tokens.css` (`var(--sl-...)`); any other hex literal is refused (black and white
  are allowed).
- Media only from inside the project, never from a `refs/` folder; an AI beat (its plan pick has a
  recipe whose `how` is AI) needs at least one `gen/files/` image, and every such image must belong to
  a queue item that passed the look check.
- A variant whose signature was already shown for the beat, or rejected by the client, is refused.

## Motion params (`motion.py variants`)
Each motion variant is a copy of the picked frame variant plus `motion-params.js`:
```js
window.SL_MOTION = {"entrance": "rise|slide-up|scale-up|mask-wipe|slide-left|slide-right",
  "ease": {"name": "power2.out", "bezier": [0.215, 0.61, 0.355, 1], "custom_ease": "M0,0 C..."},
  "timing": {"dur_s": 0.47, "delay_s": 0.0, "stagger_s": 0.1},
  "camera": {"prop": "none|scale|x|y", "from": 1.0, "to": 1.08, "ease": "...", "dur_s": 3.0}};
```
The beat's HTML must load it and build its timeline from `window.SL_MOTION` (register the CustomEase
from `custom_ease`); the check refuses an `index.html` that does not reference `motion-params.js`.
- The design must stay identical across the 3 variants (`index.html` + `tokens.css` hash).
- Every ease comes from the style's measured eases (fingerprint `layers.motion.eases`, bezier within
  0.05 RMSE); without a fingerprint the craft defaults are used (expo.out, power4.out, back.out) and
  the CLI says so. Linear only when the style really uses it (share ≥ 30 %).
- Each pair differs in at least 2 of: entrance, ease name, duration (to 0.05 s), camera prop.
- The window is 2.0-4.0 s around the beat: a short beat extends forward, the last beat backward,
  always inside the clip. A draft render outside 2.0-4.0 s is refused.

## Signatures and no repeats
`sig = "<step>:" + sha1(normalised parts)[:12]`:
- plan: (category, concept_key); frames: (concept_key, composition, treatment);
- motion: (entrance, ease name, duration rounded to 0.05, camera prop).
A new set never shows a signature already shown for that beat and step, rejected by the client at that
step, used for the client in another project (plan), or picked for another beat of the video (plan).
An option made from an idea also blocks `idea:<id>`.

## Render budget
Frames are stills (`hyperframes snapshot`), never video renders. Motion variants are draft renders of
2-4 s. "More ideas" allows 3 extra sets per beat and step; a 4th needs the owner's own words
(`--owner-ok-more --owner-words "..."`), which are logged in `shown.json`.
